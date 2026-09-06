from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

# -----------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MFGP_CHECKPOINT_FORMAT = "generic_autoregressive_mfgp_linear_v1"


# -----------------------------------------------------------------------------
# Dataclasses
# -----------------------------------------------------------------------------
@dataclass
class MFGPTrainingData:
    """Generic two-fidelity training data for the MF-GP."""

    x_lf: np.ndarray
    y_lf: np.ndarray
    x_hf: np.ndarray
    y_hf: np.ndarray
    y_lf_err: np.ndarray | None = None
    input_names: Sequence[str] | None = None

    def validate(self) -> None:
        self.x_lf = np.asarray(self.x_lf, dtype=float)
        self.y_lf = np.asarray(self.y_lf, dtype=float).reshape(-1)
        self.x_hf = np.asarray(self.x_hf, dtype=float)
        self.y_hf = np.asarray(self.y_hf, dtype=float).reshape(-1)

        if self.x_lf.ndim != 2 or self.x_hf.ndim != 2:
            raise ValueError(f"x_lf and x_hf must both be 2D. Got x_lf={self.x_lf.shape}, x_hf={self.x_hf.shape}")
        if self.x_lf.shape[1] != self.x_hf.shape[1]:
            raise ValueError(f"x_lf and x_hf must have the same feature dimension. Got {self.x_lf.shape[1]} and {self.x_hf.shape[1]}")
        if len(self.x_lf) != len(self.y_lf):
            raise ValueError(f"x_lf/y_lf length mismatch: {len(self.x_lf)} != {len(self.y_lf)}")
        if len(self.x_hf) != len(self.y_hf):
            raise ValueError(f"x_hf/y_hf length mismatch: {len(self.x_hf)} != {len(self.y_hf)}")
        if len(self.x_lf) == 0 or len(self.x_hf) == 0:
            raise ValueError("MF-GP training requires at least one LF point and one HF point")
        if not np.isfinite(self.x_lf).all() or not np.isfinite(self.x_hf).all():
            raise ValueError("MF-GP input arrays contain non-finite values")
        if not np.isfinite(self.y_lf).all() or not np.isfinite(self.y_hf).all():
            raise ValueError("MF-GP target arrays contain non-finite values")

        if self.y_lf_err is not None:
            self.y_lf_err = np.asarray(self.y_lf_err, dtype=float).reshape(-1)
            if len(self.y_lf_err) != len(self.y_lf):
                raise ValueError(f"y_lf_err length mismatch: {len(self.y_lf_err)} != {len(self.y_lf)}")
            if not np.isfinite(self.y_lf_err).all():
                raise ValueError("y_lf_err contains non-finite values")
            if (self.y_lf_err < 0).any():
                raise ValueError("y_lf_err must be non-negative")

        if self.input_names is None:
            self.input_names = [f"x{i}" for i in range(self.x_lf.shape[1])]
        else:
            self.input_names = list(self.input_names)
            if len(self.input_names) != self.x_lf.shape[1]:
                raise ValueError(f"input_names has {len(self.input_names)} entries for {self.x_lf.shape[1]} input dimensions")
            if len(set(self.input_names)) != len(self.input_names):
                raise ValueError("input_names must be unique")


@dataclass(frozen=True)
class MFGPTrainResult:
    model_path: Path
    metadata_json: Path
    metrics_json: Path
    training_prediction_csv: Path
    metrics: dict[str, float]


@dataclass
class MFGPPredictionResults:
    frame: pd.DataFrame
    mean: np.ndarray
    std: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    output_path: Path | None = None
    metrics: dict[str, float] | None = None
    metrics_path: Path | None = None

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
def load_config(path: str | Path) -> dict:
    """Load a TOML or JSON configuration file."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")

    suffix = path.suffix.lower()

    if suffix == ".toml":
        with path.open("rb") as f:
            return tomllib.load(f)

    if suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    raise ValueError(f"Unsupported config format {suffix!r}. Expected .toml or .json")

# -----------------------------------------------------------------------------
# Generic autoregressive MF-GP
# -----------------------------------------------------------------------------
class CleanAutoregressiveMFGP:
    """Two-level autoregressive MF-GP using sklearn Gaussian processes."""
    def __init__(self, 
                 random_state: int = 42,
                 alpha_lf: float = 1e-8, 
                 alpha_hf: float = 1e-8,
                 kernel_noise: float = 1e-5, 
                 kernel_noise_lim: tuple[float] = (1e-10, 1e1)
                ) -> None:
        self.random_state = int(random_state)
        self.alpha_lf = float(alpha_lf)
        self.alpha_hf = float(alpha_hf)
        self.kernel_noise = float(kernel_noise)
        self.kernel_noise_lim = tuple(kernel_noise_lim)
        
        self.x_scaler: Optional[StandardScaler] = None
        self.y_lf_scaler: Optional[StandardScaler] = None
        self.y_d_scaler: Optional[StandardScaler] = None

        self.gp_lf: Optional[GaussianProcessRegressor] = None
        self.gp_d: Optional[GaussianProcessRegressor] = None

        self.rho: Optional[float] = None
        self.x_dim: Optional[int] = None

    def _kernel(self, input_dim: int):
        return ConstantKernel(1.0, (1e-4, 1e4)) * Matern(length_scale=np.ones(input_dim), length_scale_bounds=(1e-3, 1e3), nu=1.5) + WhiteKernel(noise_level=self.kernel_noise, noise_level_bounds=self.kernel_noise_lim)

    def fit(self, x_lf: np.ndarray, y_lf: np.ndarray, x_hf: np.ndarray, y_hf: np.ndarray, verbose: bool = False) -> CleanAutoregressiveMFGP:
        x_lf = np.asarray(x_lf, dtype=float)
        y_lf = np.asarray(y_lf, dtype=float).reshape(-1, 1)
        x_hf = np.asarray(x_hf, dtype=float)
        y_hf = np.asarray(y_hf, dtype=float).reshape(-1, 1)

        if x_lf.ndim != 2 or x_hf.ndim != 2:
            raise ValueError("x_lf and x_hf must be 2D")
        if x_lf.shape[1] != x_hf.shape[1]:
            raise ValueError("x_lf and x_hf must have the same feature dimension")
        if len(x_lf) < 1 or len(x_hf) < 1:
            raise ValueError("Need at least one LF point and one HF point")

        self.x_dim = x_lf.shape[1]

        small_data = min(len(x_lf), len(x_hf)) < 3
        optimizer = None if small_data else "fmin_l_bfgs_b"
        n_restarts = 0 if small_data else 2

        self.x_scaler = StandardScaler().fit(np.vstack([x_lf, x_hf]))
        x_lf_s = self.x_scaler.transform(x_lf)
        x_hf_s = self.x_scaler.transform(x_hf)

        self.y_lf_scaler = StandardScaler().fit(y_lf)
        y_lf_s = self.y_lf_scaler.transform(y_lf).ravel()

        self.gp_lf = GaussianProcessRegressor(
            kernel=self._kernel(self.x_dim),
            alpha=self.alpha_lf,
            normalize_y=False,
            optimizer=optimizer,
            n_restarts_optimizer=n_restarts,
            random_state=self.random_state,
        )

        if verbose:
            print("[fit] Training LF GP...")
        self.gp_lf.fit(x_lf_s, y_lf_s)

        mu_lf_hf_s, _ = self.gp_lf.predict(x_hf_s, return_std=True)
        mu_lf_hf = self.y_lf_scaler.inverse_transform(mu_lf_hf_s.reshape(-1, 1)).ravel()

        y_hf_vec = y_hf.ravel()
        denominator = float(np.dot(mu_lf_hf, mu_lf_hf)) + 1e-12
        self.rho = float(np.dot(mu_lf_hf, y_hf_vec) / denominator)
        if verbose:
            print(f"[fit] Estimated rho={self.rho:.6f}")

        y_d = y_hf_vec - self.rho * mu_lf_hf

        self.y_d_scaler = StandardScaler().fit(y_d.reshape(-1, 1))
        y_d_s = self.y_d_scaler.transform(y_d.reshape(-1, 1)).ravel()

        self.gp_d = GaussianProcessRegressor(
            kernel=self._kernel(self.x_dim),
            alpha=self.alpha_hf,
            normalize_y=False,
            optimizer=optimizer,
            n_restarts_optimizer=n_restarts,
            random_state=self.random_state + 1000,
        )

        if verbose:
            print("[fit] Training HF discrepancy GP...")
        self.gp_d.fit(x_hf_s, y_d_s)
        if verbose:
            print("[fit] GP training complete.")

        return self

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.gp_lf is None or self.gp_d is None or self.x_scaler is None or self.rho is None or self.y_lf_scaler is None or self.y_d_scaler is None:
            raise RuntimeError("Model has not been fitted")

        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.ndim != 2:
            raise ValueError(f"x must be 2D, got shape {x.shape}")
        if self.x_dim is not None and x.shape[1] != self.x_dim:
            raise ValueError(f"Expected {self.x_dim} input dimensions, got {x.shape[1]}")

        x_s = self.x_scaler.transform(x)

        mu_lf_s, std_lf_s = self.gp_lf.predict(x_s, return_std=True)
        mu_d_s, std_d_s = self.gp_d.predict(x_s, return_std=True)

        y_lf_scale = float(self.y_lf_scaler.scale_[0])
        y_d_scale = float(self.y_d_scaler.scale_[0])

        mu_lf = self.y_lf_scaler.inverse_transform(mu_lf_s.reshape(-1, 1)).ravel()
        mu_d = self.y_d_scaler.inverse_transform(mu_d_s.reshape(-1, 1)).ravel()

        std_lf = np.maximum(std_lf_s * y_lf_scale, 1e-12)
        std_d = np.maximum(std_d_s * y_d_scale, 1e-12)

        mu_hf = self.rho * mu_lf + mu_d
        var_hf = (self.rho ** 2) * (std_lf ** 2) + (std_d ** 2)
        std_hf = np.maximum(np.sqrt(np.maximum(var_hf, 0.0)), 1e-12)

        return mu_hf, std_hf, mu_lf, std_lf


# -----------------------------------------------------------------------------
# Prediction intervals, validation, and metrics
# -----------------------------------------------------------------------------
def prediction_interval(mean: np.ndarray, std: np.ndarray, sigma: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return linear-space prediction center, lower bound, and upper bound."""
    mean = np.asarray(mean, dtype=float)
    std = np.maximum(np.asarray(std, dtype=float), 1e-12)
    width = float(sigma)

    lower = mean - width * std
    upper = mean + width * std

    return mean.copy(), lower, upper

def _minimum_point_count(input_dim: int) -> int:
    return max(4, int(input_dim) + 2)

def _validate_unique_input_points(x: np.ndarray, *, label: str, minimum: int, require_axis_span: bool = True) -> None:
    x = np.asarray(x, dtype=float)
    unique = np.unique(x, axis=0)

    if len(unique) < int(minimum):
        raise ValueError(f"MF-GP was not run: {label} has only {len(unique)} unique input points; at least {int(minimum)} are required")

    if require_axis_span and x.shape[1] > 0:
        missing_span = [axis for axis in range(x.shape[1]) if len(np.unique(unique[:, axis])) < 2]

        if missing_span:
            raise ValueError(f"MF-GP was not run: {label} does not span input axes {missing_span}")

def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> dict[str, float]:
    """Metrics for measuring capability of MFGP"""
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    mean = np.asarray(mean, dtype=float).reshape(-1)
    std = np.asarray(std, dtype=float).reshape(-1)

    if not (len(y_true) == len(y_pred) == len(mean) == len(std)):
        raise ValueError("y_true, y_pred, mean, and std must have the same length")

    if len(y_true) == 0:
        raise ValueError("Cannot calculate regression metrics from empty arrays")

    if not (np.isfinite(y_true).all() and np.isfinite(y_pred).all()
            and np.isfinite(mean).all() and np.isfinite(std).all()):
        raise ValueError("Regression metric inputs contain non-finite values")

    residual = y_pred - y_true
    absolute_error = np.abs(residual)
    safe_std = np.maximum(std, 1e-12)
    pull = residual / safe_std

    if len(y_true) >= 2 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        pearson_r = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        pearson_r = float("nan")

    metrics = {
        "n_points": int(len(y_true)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else float("nan"),
        "bias": float(np.mean(residual)),
        "median_absolute_error": float(np.median(absolute_error)),
        "max_absolute_error": float(np.max(absolute_error)),
        "pearson_r": pearson_r,
        "mean_predicted_sigma": float(np.mean(std)),
        "median_predicted_sigma": float(np.median(std)),
        "pull_mean": float(np.mean(pull)),
        "pull_std": float(np.std(pull, ddof=1)) if len(pull) > 1 else float("nan"),
        "rms_pull": float(np.sqrt(np.mean(np.square(pull)))),
    }

    for k in (1, 2, 3):
        _, lower, upper = prediction_interval(mean, std, sigma=float(k))
        metrics[f"coverage_{k}sigma"] = float(np.mean((y_true >= lower) & (y_true <= upper)))

    return metrics

# -----------------------------------------------------------------------------
# Checkpoints
# -----------------------------------------------------------------------------
def save_mfgp_checkpoint(model: CleanAutoregressiveMFGP, model_path: str | Path, *, version: str, input_names: Sequence[str], random_state: int, alpha_lf: float, alpha_hf: float, kernel_noise: float, kernel_noise_lim: tuple[float]) -> Path:
    """Save a fitted linear-space MF-GP and all metadata required for inference."""
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    if model.gp_lf is None or model.gp_d is None:
        raise RuntimeError("Cannot save an MF-GP before model.fit() has completed")

    bundle = {
        "checkpoint_format": MFGP_CHECKPOINT_FORMAT,
        "model": model,
        "version": str(version),
        "input_names": list(input_names),
        "random_state": int(random_state),
        "alpha_lf": float(alpha_lf),
        "alpha_hf": float(alpha_hf),
        "kernel_noise": float(kernel_noise),
        "kernel_noise_lim": tuple(kernel_noise_lim),
    }

    joblib.dump(bundle, model_path)

    return model_path

def load_mfgp_checkpoint(model_path: str | Path) -> dict:
    """Load a fitted linear-space MF-GP checkpoint."""
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"MF-GP checkpoint does not exist: {model_path}")

    bundle = joblib.load(model_path)

    if not isinstance(bundle, dict):
        raise ValueError(f"Invalid MF-GP checkpoint {model_path}: expected a dictionary")

    if bundle.get("checkpoint_format") != MFGP_CHECKPOINT_FORMAT:
        raise ValueError(f"Unsupported MF-GP checkpoint format {bundle.get('checkpoint_format')!r}; expected {MFGP_CHECKPOINT_FORMAT!r}")

    required = {"model", "input_names"}
    missing = sorted(required - set(bundle))

    if missing:
        raise ValueError(f"MF-GP checkpoint {model_path} is missing fields: {missing}")

    model = bundle["model"]

    if not isinstance(model, CleanAutoregressiveMFGP):
        raise TypeError(f"MF-GP checkpoint model has type {type(model).__name__}, expected CleanAutoregressiveMFGP")

    if model.gp_lf is None or model.gp_d is None:
        raise ValueError("MF-GP checkpoint contains an unfitted model")

    return bundle

# -----------------------------------------------------------------------------
# Prediction helpers
# -----------------------------------------------------------------------------
def _predict_in_chunks(model: CleanAutoregressiveMFGP, x: np.ndarray, chunk_size: int, *, progress: bool = True, desc: str = "MF-GP prediction") -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    chunk_size = max(1, int(chunk_size))

    means: list[np.ndarray] = []
    stds: list[np.ndarray] = []

    starts = range(0, len(x), chunk_size)
    iterator = tqdm(starts, total=(len(x) + chunk_size - 1) // chunk_size, desc=desc, unit="chunk", disable=not progress)

    for start in iterator:
        mean, std, _, _ = model.predict(x[start:start + chunk_size])
        means.append(np.asarray(mean, dtype=float))
        stds.append(np.asarray(std, dtype=float))

    if not means:
        return np.empty(0, dtype=float), np.empty(0, dtype=float)

    return np.concatenate(means), np.concatenate(stds)

def _coerce_prediction_input(x: np.ndarray | pd.DataFrame, input_names: Sequence[str]) -> tuple[np.ndarray, pd.DataFrame]:
    input_names = list(input_names)

    if isinstance(x, pd.DataFrame):
        missing = sorted(set(input_names) - set(x.columns))

        if missing:
            raise ValueError(f"Prediction dataframe is missing input columns: {missing}")

        x_array = x[input_names].to_numpy(dtype=float)
        frame = x[input_names].copy().reset_index(drop=True)

    else:
        x_array = np.asarray(x, dtype=float)

        if x_array.ndim == 1:
            x_array = x_array.reshape(1, -1)

        if x_array.ndim != 2 or x_array.shape[1] != len(input_names):
            raise ValueError(f"Expected x shape (N, {len(input_names)}) for {input_names}, got {x_array.shape}")

        frame = pd.DataFrame(x_array, columns=input_names)

    if not np.isfinite(x_array).all():
        raise ValueError("Prediction inputs contain non-finite values")

    return x_array, frame

# -----------------------------------------------------------------------------
# Runners
# -----------------------------------------------------------------------------
def run_mfgp_training(
    config_path: str | Path,
    training_data: MFGPTrainingData
) -> MFGPTrainResult:
    """Train one generic two-level autoregressive MF-GP in linear target space."""

    raw = load_config(config_path)

    run_cfg = raw.get("run", {})
    training_cfg = raw.get("training", {})
    model_cfg = raw.get("model", {})
    prediction_cfg = raw.get("prediction", {})

    training_data.validate()

    output_dir = Path(run_cfg.get("output_dir", "mfgp_outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)

    version = str(run_cfg.get("version", "default"))
    random_state = int(run_cfg.get("random_state", 42))
    verbose = bool(run_cfg.get("verbose", True))

    input_dim = training_data.x_lf.shape[1]
    default_minimum = _minimum_point_count(input_dim)

    minimum_lf = int(training_cfg.get("min_lf_points", default_minimum))
    minimum_hf = int(training_cfg.get("min_hf_points", default_minimum))
    require_axis_span = bool(training_cfg.get("require_axis_span", True))

    _validate_unique_input_points(training_data.x_lf, label="LF training data", minimum=minimum_lf, require_axis_span=require_axis_span)
    _validate_unique_input_points(training_data.x_hf, label="HF training data", minimum=minimum_hf, require_axis_span=require_axis_span)

    alpha_lf_floor = float(model_cfg.get("alpha_lf_floor", 1e-10))
    alpha_hf = float(model_cfg.get("alpha_hf", 1e-10))
    kernel_noise = float(model_cfg.get("kernel_noise", 1e-5))
    kernel_noise_lim = tuple(model_cfg.get("kernel_noise_lim", (1e-10, 1e1)))

    if training_data.y_lf_err is None:
        alpha_lf = alpha_lf_floor
    else:
        lf_sigma = np.maximum(np.asarray(training_data.y_lf_err, dtype=float), 1e-12)
        alpha_lf = max(float(np.mean(np.square(lf_sigma))), alpha_lf_floor)

    model = CleanAutoregressiveMFGP(
        random_state=random_state, 
        alpha_lf=alpha_lf,
        alpha_hf=alpha_hf,
        kernel_noise=kernel_noise,
        kernel_noise_lim=kernel_noise_lim,
    )

    if verbose:
        print(f"=== MF-GP training: {version} ===")
        print(f"[data] LF points={len(training_data.x_lf):,} | HF points={len(training_data.x_hf):,} | input_dim={input_dim}")
        print(f"[target] linear | alpha_lf={alpha_lf:.6g} | alpha_hf={alpha_hf:.6g}")

    model.fit(training_data.x_lf, training_data.y_lf, training_data.x_hf, training_data.y_hf, verbose=verbose)

    model_path = output_dir / f"mfgp_{version}_model.joblib"

    save_mfgp_checkpoint(
        model,
        model_path,
        version=version,
        input_names=training_data.input_names,
        random_state=random_state,
        alpha_lf=alpha_lf,
        alpha_hf=alpha_hf,
        kernel_noise=kernel_noise,
        kernel_noise_lim=kernel_noise_lim,
    )

    chunk_size = int(prediction_cfg.get("chunk_size", 20000))
    progress = bool(prediction_cfg.get("progress", True))

    hf_mean, hf_std = _predict_in_chunks(model, training_data.x_hf, chunk_size, progress=progress, desc="MF-GP training prediction")
    hf_prediction, hf_lower, hf_upper = prediction_interval(hf_mean, hf_std, sigma=1.0)

    metrics = regression_metrics(training_data.y_hf, hf_prediction, hf_mean, hf_std)

    training_prediction_csv = output_dir / f"mfgp_{version}_training_predictions.csv"

    training_frame = pd.DataFrame(training_data.x_hf, columns=training_data.input_names)
    training_frame["y_true"] = training_data.y_hf
    training_frame["mf_prediction"] = hf_prediction
    training_frame["mf_std"] = hf_std
    training_frame["mf_lower_1sigma"] = hf_lower
    training_frame["mf_upper_1sigma"] = hf_upper
    training_frame["residual"] = hf_prediction - training_data.y_hf
    training_frame["absolute_residual"] = np.abs(training_frame["residual"])
    training_frame["within_1sigma"] = (training_data.y_hf >= hf_lower) & (training_data.y_hf <= hf_upper)

    training_frame.to_csv(training_prediction_csv, index=False)

    metrics_json = output_dir / f"mfgp_{version}_metrics.json"
    metrics_json.write_text(json.dumps(metrics, indent=2))

    metadata = {
        "version": version,
        "checkpoint_format": MFGP_CHECKPOINT_FORMAT,
        "checkpoint_path": str(model_path),
        "config_path": str(Path(config_path)),
        "input_names": list(training_data.input_names),
        "input_dim": int(input_dim),
        "target_space": "linear",
        "n_lf_points": int(len(training_data.x_lf)),
        "n_hf_points": int(len(training_data.x_hf)),
        "minimum_lf_points": minimum_lf,
        "minimum_hf_points": minimum_hf,
        "rho": float(model.rho),
        "alpha_lf": float(alpha_lf),
        "alpha_hf": float(alpha_hf),
        "kernel_noise": float(kernel_noise),
        "kernel_noise_lim": tuple(kernel_noise_lim),
        "lf_kernel": str(model.gp_lf.kernel_) if model.gp_lf is not None else None,
        "hf_discrepancy_kernel": str(model.gp_d.kernel_) if model.gp_d is not None else None,
    }

    metadata_json = output_dir / f"mfgp_{version}_model.json"
    metadata_json.write_text(json.dumps(metadata, indent=2))

    if verbose:
        print(f"[saved] model: {model_path}")
        print(f"[saved] training predictions: {training_prediction_csv}")
        print(f"[saved] metrics: {metrics_json}")
        print(f"[done] RMSE={metrics['rmse']:.6g} | MAE={metrics['mae']:.6g} | 1σ coverage={metrics['coverage_1sigma']:.3f}")

    return MFGPTrainResult(
        model_path=model_path,
        metadata_json=metadata_json,
        metrics_json=metrics_json,
        training_prediction_csv=training_prediction_csv,
        metrics=metrics,
    )


def run_mfgp_prediction(
    config_path: str | Path, 
    *, model_path: str | Path,
    x: np.ndarray | pd.DataFrame, 
    y_true: np.ndarray | None = None, 
    output_path: str | Path | None = None, 
    metrics_path: str | Path | None = None, 
    sigma: float | None = None
) -> MFGPPredictionResults:
    """Run a saved linear-space MF-GP on arbitrary input points without re-fitting."""
    raw = load_config(config_path)

    run_cfg = raw.get("run", {})
    prediction_cfg = raw.get("prediction", {})

    bundle = load_mfgp_checkpoint(model_path)
    model: CleanAutoregressiveMFGP = bundle["model"]
    input_names = list(bundle["input_names"])

    x_array, frame = _coerce_prediction_input(x, input_names)

    sigma_value = float(prediction_cfg.get("sigma", 1.0) if sigma is None else sigma)
    chunk_size = int(prediction_cfg.get("chunk_size", 20000))
    progress = bool(prediction_cfg.get("progress", True))

    mean, std = _predict_in_chunks(model, x_array, chunk_size, progress=progress, desc="MF-GP prediction")
    mean, lower, upper = prediction_interval(mean, std, sigma=sigma_value)

    frame["mf_prediction"] = mean
    frame["mf_std"] = std
    frame[f"mf_lower_{sigma_value:g}sigma"] = lower
    frame[f"mf_upper_{sigma_value:g}sigma"] = upper

    metrics = None
    resolved_metrics_path = None

    if y_true is not None:
        y_true = np.asarray(y_true, dtype=float).reshape(-1)

        if len(y_true) != len(frame):
            raise ValueError(f"y_true length mismatch: {len(y_true)} != {len(frame)}")

        if not np.isfinite(y_true).all():
            raise ValueError("y_true contains non-finite values")

        metrics = regression_metrics(y_true, mean, mean, std)

        frame["y_true"] = y_true
        frame["residual"] = mean - y_true
        frame["absolute_residual"] = np.abs(frame["residual"])

        _, one_sigma_lower, one_sigma_upper = prediction_interval(mean, std, sigma=1.0)
        frame["within_1sigma"] = (y_true >= one_sigma_lower) & (y_true <= one_sigma_upper)

    resolved_output_path = None

    if output_path is not None:
        resolved_output_path = Path(output_path)

    elif bool(prediction_cfg.get("save_predictions", True)):
        output_dir = Path(run_cfg.get("output_dir", "mfgp_outputs"))
        output_dir.mkdir(parents=True, exist_ok=True)

        version = str(run_cfg.get("version", bundle.get("version", "default")))
        resolved_output_path = output_dir / f"mfgp_{version}_predictions.csv"

    if resolved_output_path is not None:
        resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(resolved_output_path, index=False)

    if metrics is not None:
        if metrics_path is not None:
            resolved_metrics_path = Path(metrics_path)

        elif resolved_output_path is not None:
            resolved_metrics_path = resolved_output_path.with_name(f"{resolved_output_path.stem}_metrics.json")

        if resolved_metrics_path is not None:
            resolved_metrics_path.parent.mkdir(parents=True, exist_ok=True)
            resolved_metrics_path.write_text(json.dumps(metrics, indent=2))

    return MFGPPredictionResults(
        frame=frame,
        mean=mean,
        std=std,
        lower=lower,
        upper=upper,
        output_path=resolved_output_path,
        metrics=metrics,
        metrics_path=resolved_metrics_path,
    )
