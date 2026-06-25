from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable, **_: object):  # type: ignore[misc]
        return iterable

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xlzd_resum.config import DEFAULT_FILE_STEMS, FileLoadConfig
from xlzd_resum.io_utils import load_event_collection
from xlzd_resum.theta import add_centered_z_coordinate

FIXED_TPC_R_MAX = 1500.0
FIXED_TPC_Z_MAX = 2000.0
FIXED_TPC_Z_CENTER = 1982.48


DEFAULT_COMPONENT_GROUPS: dict[str, str] = {
    "TPCPMTTop": "top",
    "GXe": "top",
    "TPCPMTTBottom": "bottom",
    "CathodeGrid": "bottom",
    "FieldCage": "side",
    "ICV": "side",
    "OCV": "side",
}

GROUP_TARGETS: dict[str, str] = {
    "top": "z_from_center",
    "bottom": "z_from_center",
    "side": "r",
}


@dataclass(slots=True)
class DepthDataset:
    df: pd.DataFrame
    z_center: float
    r_max: float
    z_max: float
    component_groups: dict[str, str]


@dataclass(slots=True)
class VBLLPosterior:
    mean: np.ndarray
    covariance: np.ndarray
    noise_variance: float
    prior_precision: float


@dataclass(slots=True)
class VBLLRunResult:
    target_name: str
    feature_columns: list[str]
    history: pd.DataFrame
    prediction_df: pd.DataFrame
    metrics: dict[str, float]
    model_dir: Path


@dataclass(slots=True)
class MLPBaselineRunResult:
    target_name: str
    feature_columns: list[str]
    history: pd.DataFrame
    prediction_df: pd.DataFrame
    metrics: dict[str, float]
    model_dir: Path


class MLPBackbone(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: Iterable[int] = (128, 64), dropout: float = 0.1):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for hidden in hidden_dims:
            layers.append(nn.Linear(prev, hidden))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = hidden
        self.network = nn.Sequential(*layers)
        self.output_dim = prev

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class DeterministicMLPRegressor(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: Iterable[int] = (128, 64), dropout: float = 0.1):
        super().__init__()
        self.backbone = MLPBackbone(in_dim=in_dim, hidden_dims=hidden_dims, dropout=dropout)
        self.head = nn.Linear(self.backbone.output_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x)).squeeze(-1)


def find_repo_root(start: Optional[Path] = None) -> Path:
    start = (start or Path.cwd()).resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "prepare_resum_data.py").exists() and (candidate / "README.md").exists():
            return candidate
    raise RuntimeError("Could not find the XLZD repo root.")


def component_group_for(name: str) -> str:
    if name in DEFAULT_COMPONENT_GROUPS:
        return DEFAULT_COMPONENT_GROUPS[name]
    lower = name.lower()
    if "top" in lower or lower == "gxe":
        return "top"
    if "bottom" in lower or "cathode" in lower:
        return "bottom"
    return "side"


def load_depth_dataset(
    data_dir: str | Path,
    *,
    file_stems: Optional[list[str]] = None,
    max_rows_per_component: int = 25000,
) -> DepthDataset:
    config = FileLoadConfig(
        input_dir=Path(data_dir),
        file_stems=file_stems or list(DEFAULT_FILE_STEMS),
        max_rows_per_file=max_rows_per_component,
        concatenate_components=True,
    )
    loaded = load_event_collection(config)
    z_center = float(FIXED_TPC_Z_CENTER)
    df = add_centered_z_coordinate(loaded.concatenated, z_center).copy()
    df["sz_centered"] = df["sz"].to_numpy(dtype=float) - float(z_center)
    df["d_center"] = np.sqrt(df["r"].to_numpy(dtype=float) ** 2 + df["z_from_center"].to_numpy(dtype=float) ** 2)
    df["r_normalized"] = df["r"].to_numpy(dtype=float) / float(FIXED_TPC_R_MAX)
    df["z_from_center_normalized"] = df["z_from_center"].to_numpy(dtype=float) / float(FIXED_TPC_Z_MAX)
    df["d_center_normalized"] = np.sqrt(
        df["r_normalized"].to_numpy(dtype=float) ** 2
        + df["z_from_center_normalized"].to_numpy(dtype=float) ** 2
    )
    df["component_group"] = df["source_component"].map(component_group_for)
    df["group_target"] = df["component_group"].map(GROUP_TARGETS)
    return DepthDataset(
        df=df,
        z_center=float(z_center),
        r_max=float(FIXED_TPC_R_MAX),
        z_max=float(FIXED_TPC_Z_MAX),
        component_groups={name: component_group_for(name) for name in sorted(df["source_component"].unique())},
    )


def component_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["source_component", "component_group"], dropna=False)
        .agg(
            event_count=("global_event_id", "size"),
            sx_mean=("sx", "mean"),
            sy_mean=("sy", "mean"),
            sz_centered_mean=("sz_centered", "mean"),
            r_mean=("r", "mean"),
            z_from_center_mean=("z_from_center", "mean"),
            d_center_mean=("d_center", "mean"),
            d_center_p01=("d_center", lambda s: np.quantile(s, 0.01)),
        )
        .reset_index()
        .sort_values(["component_group", "source_component"])
    )
    return summary


def train_val_split(
    df: pd.DataFrame,
    *,
    target_col: str,
    feature_cols: list[str],
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df, val_df = train_test_split(
        df[feature_cols + [target_col, "source_component", "component_group"]].copy(),
        test_size=test_size,
        random_state=random_state,
        stratify=df["source_component"],
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def build_feature_frame(df: pd.DataFrame, *, include_component: bool = True) -> pd.DataFrame:
    base = df[["sx", "sy", "sz_centered", "s_r", "s_z_from_center"]].copy()
    if include_component:
        component_dummies = pd.get_dummies(df["source_component"], prefix="component")
        return pd.concat([base, component_dummies], axis=1)
    return base


def make_loaders(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    *,
    batch_size: int = 512,
) -> tuple[DataLoader, DataLoader]:
    train_ds = TensorDataset(torch.from_numpy(train_x).float(), torch.from_numpy(train_y).float())
    val_ds = TensorDataset(torch.from_numpy(val_x).float(), torch.from_numpy(val_y).float())
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    return train_loader, val_loader


def train_backbone_regressor(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    *,
    hidden_dims: tuple[int, int] = (128, 64),
    dropout: float = 0.1,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 40,
    batch_size: int = 512,
    device: Optional[str] = None,
) -> tuple[DeterministicMLPRegressor, pd.DataFrame]:
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = DeterministicMLPRegressor(in_dim=train_x.shape[1], hidden_dims=hidden_dims, dropout=dropout).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    train_loader, val_loader = make_loaders(train_x, train_y, val_x, val_y, batch_size=batch_size)
    best_state = None
    best_val = float("inf")
    history_rows: list[dict[str, float]] = []
    epoch_times: list[float] = []

    epoch_bar = tqdm(range(1, epochs + 1), desc="Training epochs")
    for epoch in epoch_bar:
        start = time.perf_counter()
        model.train()
        train_losses: list[float] = []
        batch_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        for xb, yb in batch_bar:
            xb = xb.to(dev)
            yb = yb.to(dev)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
            batch_bar.set_postfix(loss=f"{np.mean(train_losses):.4f}")

        model.eval()
        val_losses: list[float] = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(dev)
                yb = yb.to(dev)
                pred = model(xb)
                val_losses.append(float(loss_fn(pred, yb).detach().cpu()))

        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        val_loss = float(np.mean(val_losses)) if val_losses else float("nan")
        elapsed = time.perf_counter() - start
        epoch_times.append(elapsed)
        eta_seconds = float(np.mean(epoch_times) * (epochs - epoch))
        history_rows.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "epoch_seconds": elapsed,
                "eta_seconds": eta_seconds,
            }
        )
        epoch_bar.set_postfix(train=f"{train_loss:.4f}", val=f"{val_loss:.4f}", eta=f"{eta_seconds/60.0:.1f}m")
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, pd.DataFrame(history_rows)


def _extract_features(model: DeterministicMLPRegressor, x: np.ndarray, device: Optional[str] = None) -> np.ndarray:
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(dev)
    model.eval()
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x), 4096):
            xb = torch.from_numpy(x[start : start + 4096]).float().to(dev)
            h = model.backbone(xb).detach().cpu().numpy()
            outputs.append(h)
    return np.concatenate(outputs, axis=0) if outputs else np.empty((0, model.backbone.output_dim), dtype=np.float32)


def fit_vbll_posterior(
    model: DeterministicMLPRegressor,
    train_x: np.ndarray,
    train_y: np.ndarray,
    *,
    prior_precision: float = 1.0,
    device: Optional[str] = None,
) -> VBLLPosterior:
    features = _extract_features(model, train_x, device=device)
    phi = np.concatenate([np.ones((len(features), 1), dtype=np.float64), features.astype(np.float64)], axis=1)
    y = train_y.astype(np.float64).reshape(-1, 1)
    head_weight = model.head.weight.detach().cpu().numpy().astype(np.float64).reshape(-1, 1)
    head_bias = model.head.bias.detach().cpu().numpy().astype(np.float64).reshape(1, 1)
    mean_init = np.vstack([head_bias, head_weight])
    pred = phi @ mean_init
    residual = y - pred
    noise_variance = float(max(np.mean(residual ** 2), 1e-5))
    beta = 1.0 / noise_variance
    eye = np.eye(phi.shape[1], dtype=np.float64)
    s_inv = prior_precision * eye + beta * (phi.T @ phi)
    covariance = np.linalg.inv(s_inv)
    mean = beta * covariance @ phi.T @ y
    return VBLLPosterior(
        mean=mean.reshape(-1),
        covariance=covariance,
        noise_variance=noise_variance,
        prior_precision=float(prior_precision),
    )


def predict_with_vbll(
    model: DeterministicMLPRegressor,
    posterior: VBLLPosterior,
    x: np.ndarray,
    *,
    y_mean: float,
    y_scale: float,
    device: Optional[str] = None,
) -> tuple[np.ndarray, np.ndarray]:
    features = _extract_features(model, x, device=device)
    phi = np.concatenate([np.ones((len(features), 1), dtype=np.float64), features.astype(np.float64)], axis=1)
    mean_scaled = phi @ posterior.mean.reshape(-1, 1)
    variance_scaled = posterior.noise_variance + np.sum((phi @ posterior.covariance) * phi, axis=1)
    std_scaled = np.sqrt(np.maximum(variance_scaled, 1e-12))
    pred_mean = mean_scaled.reshape(-1) * y_scale + y_mean
    pred_std = std_scaled.reshape(-1) * y_scale
    return pred_mean.astype(np.float64), pred_std.astype(np.float64)


def _infer_hidden_dims_from_state_dict(state_dict: dict[str, torch.Tensor]) -> tuple[int, ...]:
    linear_keys = sorted(
        key for key in state_dict.keys() if key.startswith("backbone.network") and key.endswith(".weight")
    )
    hidden_dims: list[int] = []
    for key in linear_keys:
        weight = state_dict[key]
        hidden_dims.append(int(weight.shape[0]))
    return tuple(hidden_dims)


def predict_checkpoint_on_dataframe(
    df: pd.DataFrame,
    *,
    checkpoint_path: str | Path,
    device: Optional[str] = None,
) -> pd.DataFrame:
    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    state_dict = checkpoint["state_dict"]
    hidden_dims = _infer_hidden_dims_from_state_dict(state_dict)
    feature_columns = list(checkpoint["feature_columns"])
    target_name = str(checkpoint["target_name"])
    x_mean = np.asarray(checkpoint["x_scaler_mean"], dtype=np.float64)
    x_scale = np.asarray(checkpoint["x_scaler_scale"], dtype=np.float64)
    y_mean = float(np.asarray(checkpoint["y_scaler_mean"], dtype=np.float64)[0])
    y_scale = float(np.asarray(checkpoint["y_scaler_scale"], dtype=np.float64)[0])

    model = DeterministicMLPRegressor(
        in_dim=int(len(x_mean)),
        hidden_dims=hidden_dims,
        dropout=0.1,
    )
    model.load_state_dict(state_dict)

    posterior_dict = checkpoint["posterior"]
    posterior = VBLLPosterior(
        mean=np.asarray(posterior_dict["mean"], dtype=np.float64),
        covariance=np.asarray(posterior_dict["covariance"], dtype=np.float64),
        noise_variance=float(posterior_dict["noise_variance"]),
        prior_precision=float(posterior_dict["prior_precision"]),
    )

    feature_frame = build_feature_frame(df, include_component=True)
    merged = pd.concat([df.reset_index(drop=True), feature_frame.reset_index(drop=True)], axis=1)
    missing = [col for col in feature_columns if col not in merged.columns]
    for col in missing:
        merged[col] = 0.0
    x = merged[feature_columns].to_numpy(dtype=np.float64)
    if x.shape[1] != len(x_mean):
        raise RuntimeError(
            f"Checkpoint expects {len(x_mean)} input features after reconstruction, "
            f"but built matrix has shape {x.shape}. "
            "This usually means the training-time feature assembly changed."
        )
    x_scaled = ((x - x_mean) / x_scale).astype(np.float32)
    pred_mean, pred_std = predict_with_vbll(
        model,
        posterior,
        x_scaled,
        y_mean=y_mean,
        y_scale=y_scale,
        device=device,
    )

    out = df.copy()
    out["pred_mean"] = pred_mean
    out["pred_std"] = pred_std
    out["pred_abs_error"] = np.abs(out[target_name].to_numpy(dtype=float) - pred_mean)
    out["pred_target_name"] = target_name
    return out


def build_prediction_frame(
    base_df: pd.DataFrame,
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    target_name: str,
) -> pd.DataFrame:
    out = base_df[["source_component", "component_group"]].copy()
    out[target_name] = y_true
    out["pred_mean"] = y_pred
    out["pred_std"] = y_std
    out["abs_error"] = np.abs(y_true - y_pred)
    out["within_1sigma"] = out["abs_error"] <= out["pred_std"]
    out["within_2sigma"] = out["abs_error"] <= 2.0 * out["pred_std"]
    return out


def regression_metrics(df: pd.DataFrame, *, target_name: str) -> dict[str, float]:
    y_true = df[target_name].to_numpy(dtype=float)
    y_pred = df["pred_mean"].to_numpy(dtype=float)
    y_std = df["pred_std"].to_numpy(dtype=float)
    mse = float(np.mean((y_true - y_pred) ** 2))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(mse))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "coverage_1sigma": float(np.mean(np.abs(y_true - y_pred) <= y_std)),
        "coverage_2sigma": float(np.mean(np.abs(y_true - y_pred) <= 2.0 * y_std)),
        "mean_pred_std": float(np.mean(y_std)),
    }


def run_vbll_regression(
    df: pd.DataFrame,
    *,
    target_name: str,
    model_dir: str | Path,
    include_component: bool = True,
    epochs: int = 40,
    batch_size: int = 512,
    lr: float = 1e-3,
    hidden_dims: tuple[int, int] = (128, 64),
    dropout: float = 0.1,
    random_state: int = 42,
) -> VBLLRunResult:
    feature_frame = build_feature_frame(df, include_component=include_component)
    feature_cols = list(feature_frame.columns)
    merged = pd.concat([df.reset_index(drop=True), feature_frame.reset_index(drop=True)], axis=1)
    train_df, val_df = train_val_split(
        merged,
        target_col=target_name,
        feature_cols=feature_cols,
        random_state=random_state,
    )

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    train_x = x_scaler.fit_transform(train_df[feature_cols].to_numpy(dtype=float)).astype(np.float32)
    val_x = x_scaler.transform(val_df[feature_cols].to_numpy(dtype=float)).astype(np.float32)
    train_y = y_scaler.fit_transform(train_df[[target_name]].to_numpy(dtype=float)).astype(np.float32).reshape(-1)
    val_y = y_scaler.transform(val_df[[target_name]].to_numpy(dtype=float)).astype(np.float32).reshape(-1)

    model, history = train_backbone_regressor(
        train_x,
        train_y,
        val_x,
        val_y,
        hidden_dims=hidden_dims,
        dropout=dropout,
        lr=lr,
        epochs=epochs,
        batch_size=batch_size,
    )
    posterior = fit_vbll_posterior(model, train_x, train_y)
    pred_mean, pred_std = predict_with_vbll(
        model,
        posterior,
        val_x,
        y_mean=float(y_scaler.mean_[0]),
        y_scale=float(y_scaler.scale_[0]),
    )
    prediction_df = build_prediction_frame(
        val_df,
        y_true=val_df[target_name].to_numpy(dtype=float),
        y_pred=pred_mean,
        y_std=pred_std,
        target_name=target_name,
    )
    metrics = regression_metrics(prediction_df, target_name=target_name)

    out_dir = Path(model_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    history.to_csv(out_dir / "history.csv", index=False)
    prediction_df.to_csv(out_dir / "validation_predictions.csv", index=False)
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "posterior": asdict(posterior),
            "x_scaler_mean": x_scaler.mean_.tolist(),
            "x_scaler_scale": x_scaler.scale_.tolist(),
            "y_scaler_mean": y_scaler.mean_.tolist(),
            "y_scaler_scale": y_scaler.scale_.tolist(),
            "feature_columns": feature_cols,
            "target_name": target_name,
        },
        out_dir / "vbll_regressor.pt",
    )
    return VBLLRunResult(
        target_name=target_name,
        feature_columns=feature_cols,
        history=history,
        prediction_df=prediction_df,
        metrics=metrics,
        model_dir=out_dir,
    )


def run_mlp_regression(
    df: pd.DataFrame,
    *,
    target_name: str,
    model_dir: str | Path,
    include_component: bool = True,
    epochs: int = 40,
    batch_size: int = 512,
    lr: float = 1e-3,
    hidden_dims: tuple[int, int] = (128, 64),
    dropout: float = 0.1,
    random_state: int = 42,
) -> MLPBaselineRunResult:
    feature_frame = build_feature_frame(df, include_component=include_component)
    feature_cols = list(feature_frame.columns)
    merged = pd.concat([df.reset_index(drop=True), feature_frame.reset_index(drop=True)], axis=1)
    train_df, val_df = train_val_split(
        merged,
        target_col=target_name,
        feature_cols=feature_cols,
        random_state=random_state,
    )

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    train_x = x_scaler.fit_transform(train_df[feature_cols].to_numpy(dtype=float)).astype(np.float32)
    val_x = x_scaler.transform(val_df[feature_cols].to_numpy(dtype=float)).astype(np.float32)
    train_y = y_scaler.fit_transform(train_df[[target_name]].to_numpy(dtype=float)).astype(np.float32).reshape(-1)
    val_y = y_scaler.transform(val_df[[target_name]].to_numpy(dtype=float)).astype(np.float32).reshape(-1)

    model, history = train_backbone_regressor(
        train_x,
        train_y,
        val_x,
        val_y,
        hidden_dims=hidden_dims,
        dropout=dropout,
        lr=lr,
        epochs=epochs,
        batch_size=batch_size,
    )

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(dev)
    model.eval()
    preds_scaled: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(val_x), 4096):
            xb = torch.from_numpy(val_x[start : start + 4096]).float().to(dev)
            preds_scaled.append(model(xb).detach().cpu().numpy())
    pred_scaled = np.concatenate(preds_scaled, axis=0) if preds_scaled else np.empty((0,), dtype=np.float32)
    pred_mean = pred_scaled.astype(np.float64) * float(y_scaler.scale_[0]) + float(y_scaler.mean_[0])
    pred_std = np.zeros_like(pred_mean, dtype=np.float64)

    prediction_df = build_prediction_frame(
        val_df,
        y_true=val_df[target_name].to_numpy(dtype=float),
        y_pred=pred_mean,
        y_std=pred_std,
        target_name=target_name,
    )
    metrics = regression_metrics(prediction_df, target_name=target_name)

    out_dir = Path(model_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    history.to_csv(out_dir / "history.csv", index=False)
    prediction_df.to_csv(out_dir / "validation_predictions.csv", index=False)
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "x_scaler_mean": x_scaler.mean_.tolist(),
            "x_scaler_scale": x_scaler.scale_.tolist(),
            "y_scaler_mean": y_scaler.mean_.tolist(),
            "y_scaler_scale": y_scaler.scale_.tolist(),
            "feature_columns": feature_cols,
            "target_name": target_name,
        },
        out_dir / "mlp_regressor.pt",
    )
    return MLPBaselineRunResult(
        target_name=target_name,
        feature_columns=feature_cols,
        history=history,
        prediction_df=prediction_df,
        metrics=metrics,
        model_dir=out_dir,
    )


def plot_component_initial_positions(df: pd.DataFrame, *, max_points_per_component: int = 1500) -> plt.Figure:
    components = sorted(df["source_component"].unique())
    ncols = 3
    nrows = int(math.ceil(len(components) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.0 * nrows), squeeze=False)
    rng = np.random.default_rng(42)
    for ax, component in zip(axes.ravel(), components):
        part = df[df["source_component"] == component]
        if len(part) > max_points_per_component:
            idx = rng.choice(len(part), size=max_points_per_component, replace=False)
            part = part.iloc[idx]
        ax.scatter(part["s_r"], part["sz_centered"], s=8, alpha=0.35)
        ax.set_title(f"{component} ({component_group_for(component)})")
        ax.set_xlabel("s_r")
        ax.set_ylabel("sz_centered")
        ax.grid(True, alpha=0.25)
    for ax in axes.ravel()[len(components) :]:
        ax.axis("off")
    fig.suptitle("Initial positions by component", fontsize=15)
    fig.tight_layout()
    return fig


def plot_target_distributions(df: pd.DataFrame, *, target_col: str) -> plt.Figure:
    components = sorted(df["source_component"].unique())
    ncols = 3
    nrows = int(math.ceil(len(components) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.8 * nrows), squeeze=False)
    for ax, component in zip(axes.ravel(), components):
        part = df[df["source_component"] == component]
        ax.hist(part[target_col], bins=40, alpha=0.85)
        ax.set_title(component)
        ax.set_xlabel(target_col)
        ax.set_ylabel("count")
        ax.grid(True, alpha=0.25)
    for ax in axes.ravel()[len(components) :]:
        ax.axis("off")
    fig.suptitle(f"Target distributions by component: {target_col}", fontsize=15)
    fig.tight_layout()
    return fig


def plot_training_history(history: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].plot(history["epoch"], history["train_loss"], label="train")
    axes[0].plot(history["epoch"], history["val_loss"], label="validation")
    axes[0].set_title("Loss by epoch")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("MSE loss")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()
    axes[1].plot(history["epoch"], history["epoch_seconds"], label="epoch seconds")
    axes[1].plot(history["epoch"], history["eta_seconds"], label="eta seconds")
    axes[1].set_title("Runtime tracking")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("seconds")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    return fig


def plot_prediction_parity(prediction_df: pd.DataFrame, *, target_col: str, title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.errorbar(
        prediction_df[target_col],
        prediction_df["pred_mean"],
        yerr=prediction_df["pred_std"],
        fmt="o",
        ms=4,
        alpha=0.35,
        capsize=2,
    )
    lo = min(float(prediction_df[target_col].min()), float(prediction_df["pred_mean"].min()))
    hi = max(float(prediction_df[target_col].max()), float(prediction_df["pred_mean"].max()))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlabel(f"true {target_col}")
    ax.set_ylabel("predicted mean")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def plot_component_error_bars(prediction_df: pd.DataFrame) -> plt.Figure:
    summary = (
        prediction_df.groupby(["source_component", "component_group"], dropna=False)
        .agg(
            mae=("abs_error", "mean"),
            coverage_1sigma=("within_1sigma", "mean"),
            coverage_2sigma=("within_2sigma", "mean"),
            mean_pred_std=("pred_std", "mean"),
        )
        .reset_index()
        .sort_values(["component_group", "source_component"])
    )
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    axes[0].bar(summary["source_component"], summary["mae"])
    axes[0].set_title("MAE by component")
    axes[1].bar(summary["source_component"], summary["coverage_1sigma"], label="1-sigma")
    axes[1].bar(summary["source_component"], summary["coverage_2sigma"], alpha=0.55, label="2-sigma")
    axes[1].set_title("Coverage by component")
    axes[1].legend()
    axes[2].bar(summary["source_component"], summary["mean_pred_std"])
    axes[2].set_title("Mean predicted std by component")
    for ax in axes:
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def plot_component_penetration_ranking(
    df: pd.DataFrame,
    *,
    target_col: str,
    title: str,
) -> plt.Figure:
    summary = (
        df.groupby(["source_component", "component_group"], dropna=False)[target_col]
        .agg(p01=lambda s: np.quantile(s, 0.01), p10=lambda s: np.quantile(s, 0.10), median="median")
        .reset_index()
        .sort_values("p01")
    )
    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = np.arange(len(summary))
    ax.bar(x - 0.25, summary["p01"], width=0.25, label="1% quantile")
    ax.bar(x, summary["p10"], width=0.25, label="10% quantile")
    ax.bar(x + 0.25, summary["median"], width=0.25, label="median")
    ax.set_xticks(x)
    ax.set_xticklabels(summary["source_component"], rotation=45, ha="right")
    ax.set_ylabel(target_col)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_component_depth_heatmap(
    df: pd.DataFrame,
    *,
    target_col: str,
    max_points_per_component: int = 2000,
) -> plt.Figure:
    components = sorted(df["source_component"].unique())
    ncols = 3
    nrows = int(math.ceil(len(components) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.0 * nrows), squeeze=False)
    rng = np.random.default_rng(7)
    for ax, component in zip(axes.ravel(), components):
        part = df[df["source_component"] == component]
        if len(part) > max_points_per_component:
            idx = rng.choice(len(part), size=max_points_per_component, replace=False)
            part = part.iloc[idx]
        scatter = ax.scatter(part["s_r"], part["sz_centered"], c=part[target_col], s=9, alpha=0.55, cmap="viridis_r")
        ax.set_title(component)
        ax.set_xlabel("s_r")
        ax.set_ylabel("sz_centered")
        ax.grid(True, alpha=0.2)
        fig.colorbar(scatter, ax=ax, shrink=0.85)
    for ax in axes.ravel()[len(components) :]:
        ax.axis("off")
    fig.suptitle(f"Initial position colored by {target_col}", fontsize=15)
    fig.tight_layout()
    return fig
