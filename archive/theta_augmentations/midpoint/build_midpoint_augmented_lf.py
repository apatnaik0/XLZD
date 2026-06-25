#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch


def find_repo_root(start: Optional[Path] = None) -> Path:
    start = (start or Path.cwd()).resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "prepare_resum_data.py").exists() and (candidate / "README.md").exists():
            return candidate
    raise RuntimeError("Could not find the XLZD repo root.")


REPO_ROOT = find_repo_root()
SRC_RUN_CNP = REPO_ROOT / "src" / "run_cnp"
LF_AUG_DIR = REPO_ROOT / "lf_augmentations"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_RUN_CNP) not in sys.path:
    sys.path.insert(0, str(SRC_RUN_CNP))
if str(LF_AUG_DIR) not in sys.path:
    sys.path.insert(0, str(LF_AUG_DIR))

from cnp_clean_pipeline import load_model_checkpoint, load_runtime_config, set_seed  # noqa: E402
from lf_augmentation import default_paths as default_lf_paths  # noqa: E402
from lf_augmentation import write_mfgp_config_variant  # noqa: E402
from xlzd_resum.theta import ThetaRegion, theta_mask  # noqa: E402


@dataclass(slots=True)
class MidpointPaths:
    repo_root: Path
    config_path: Path
    model_path: Path
    train_dir: Path
    training_cnp_csv: Path
    validation_cnp_csv: Path
    artifact_dir: Path


def default_paths(config_path: str | Path | None = None, model_path: str | Path | None = None) -> MidpointPaths:
    lf_paths = default_lf_paths(config_path)
    runtime = load_runtime_config(lf_paths.config_path, seed=42)
    resolved_model = Path(model_path).expanduser().resolve() if model_path else (
        runtime.out_dir / f"cnp_{runtime.version}_model_{runtime.epochs}epochs.pth"
    ).resolve()
    artifact_dir = REPO_ROOT / "theta_augmentations" / "midpoint" / "artifacts"
    return MidpointPaths(
        repo_root=REPO_ROOT,
        config_path=lf_paths.config_path,
        model_path=resolved_model,
        train_dir=runtime.train_dir.resolve(),
        training_cnp_csv=lf_paths.training_cnp_csv,
        validation_cnp_csv=lf_paths.validation_cnp_csv,
        artifact_dir=artifact_dir,
    )


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file type: {path}")


def _list_block_files(train_dir: Path) -> List[Path]:
    files = sorted([*train_dir.glob("*.csv"), *train_dir.glob("*.parquet")])
    if not files:
        raise FileNotFoundError(f"No training block files found in {train_dir}")
    return files


def _build_source_table_map(train_dir: Path) -> Dict[str, Path]:
    tables = _list_block_files(train_dir)
    mapping: Dict[str, Path] = {}
    for path in tables:
        mapping[path.name] = path
        mapping[path.with_suffix(".h5").name] = path
    return mapping


def _required_block_columns() -> List[str]:
    return ["R_max", "Z_max", "r", "z_from_center", "s_r", "s_z_from_center", "inside_theta"]


def _predict_trial_for_theta(
    df_block: pd.DataFrame,
    *,
    source_name: str,
    iteration: int,
    theta_r: float,
    theta_z: float,
    model,
    context_ratio: float,
    mc_samples: int,
    chunk_size: int,
    seed: int,
) -> Tuple[Dict[str, float | int | str], pd.DataFrame]:
    theta = ThetaRegion(Z_max=float(theta_z), R_max=float(theta_r))
    inside = theta_mask(df_block, theta).astype(np.float32).to_numpy().reshape(-1, 1)

    x_np = np.column_stack(
        [
            np.full(len(df_block), float(theta_r), dtype=np.float32),
            np.full(len(df_block), float(theta_z), dtype=np.float32),
            df_block["s_r"].to_numpy(dtype=np.float32),
            df_block["s_z_from_center"].to_numpy(dtype=np.float32),
        ]
    ).astype(np.float32)

    n = len(df_block)
    rng = np.random.default_rng(seed)
    n_context = max(2, int(context_ratio * n))
    n_context = min(n_context, n - 1)
    c_idx = rng.choice(n, size=n_context, replace=False)
    context_mask = np.zeros(n, dtype=np.uint8)
    context_mask[c_idx] = 1

    dev = next(model.parameters()).device
    context_x = torch.from_numpy(x_np[c_idx]).to(dev)
    context_y = torch.from_numpy(inside[c_idx]).to(dev)

    mu_parts: List[np.ndarray] = []
    std_parts: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, n, chunk_size):
            end = min(n, start + chunk_size)
            tx = torch.from_numpy(x_np[start:end]).to(dev)
            mu_t, std_t = model.predict_proba_mc(context_x, context_y, tx, mc_samples=mc_samples)
            mu_parts.append(mu_t.cpu().numpy())
            std_parts.append(std_t.cpu().numpy())

    mu = np.vstack(mu_parts).reshape(-1, 1)
    std = np.vstack(std_parts).reshape(-1, 1)

    per_signal = pd.DataFrame(
        {
            "iteration": np.full(n, int(iteration), dtype=np.int32),
            "fidelity": np.zeros(n, dtype=np.int32),
            "source_file": [source_name] * n,
            "event_index": np.arange(n, dtype=np.int64),
            "n_samples_file": np.full(n, n, dtype=np.int32),
            "is_context": context_mask.astype(np.uint8),
            "R_max": np.full(n, float(theta_r), dtype=float),
            "Z_max": np.full(n, float(theta_z), dtype=float),
            "s_r": df_block["s_r"].to_numpy(dtype=float),
            "s_z_from_center": df_block["s_z_from_center"].to_numpy(dtype=float),
            "y_raw": inside.ravel().astype(float),
            "y_cnp": mu.ravel().astype(float),
            "y_cnp_err": std.ravel().astype(float),
        }
    )

    row = {
        "iteration": int(iteration),
        "fidelity": 0,
        "n_samples": int(n),
        "R_max": float(theta_r),
        "Z_max": float(theta_z),
        "y_cnp": float(mu.mean()),
        "y_cnp_err": float(np.sqrt(np.mean(np.square(std)))),
        "y_raw": float(inside.mean()),
        "log_prop": np.nan,
        "bce": np.nan,
        "source_file": source_name,
    }
    return row, per_signal


def _normalized_distance(a: np.ndarray, b: np.ndarray, theta_max: Sequence[float]) -> float:
    scale = np.asarray(theta_max, dtype=float)
    return float(np.sqrt(np.sum(np.square((a - b) / scale))))


def _build_midpoint_pairs(
    theta_df: pd.DataFrame,
    *,
    theta_max: Sequence[float],
    nearest_k: int = 1,
    min_norm_dist: float = 0.05,
    max_norm_dist: float = 0.25,
) -> List[Dict[str, object]]:
    points = theta_df[["R_max", "Z_max"]].to_numpy(dtype=float)
    source_files = theta_df["source_file"].astype(str).tolist()
    pair_records: List[Dict[str, object]] = []
    seen_pairs: set[Tuple[int, int]] = set()
    seen_midpoints: set[Tuple[float, float]] = set()

    for i in range(len(points)):
        dists = []
        for j in range(len(points)):
            if i == j:
                continue
            d = _normalized_distance(points[i], points[j], theta_max)
            dists.append((d, j))
        dists.sort(key=lambda x: x[0])
        used = 0
        for d, j in dists:
            if d < min_norm_dist or d > max_norm_dist:
                continue
            pair = (min(i, j), max(i, j))
            if pair in seen_pairs:
                continue
            midpoint = (points[i] + points[j]) / 2.0
            midpoint_key = (round(float(midpoint[0]), 6), round(float(midpoint[1]), 6))
            if midpoint_key in seen_midpoints:
                continue
            seen_pairs.add(pair)
            seen_midpoints.add(midpoint_key)
            pair_records.append(
                {
                    "idx_a": int(pair[0]),
                    "idx_b": int(pair[1]),
                    "source_file_a": source_files[pair[0]],
                    "source_file_b": source_files[pair[1]],
                    "R_max_a": float(points[pair[0], 0]),
                    "Z_max_a": float(points[pair[0], 1]),
                    "R_max_b": float(points[pair[1], 0]),
                    "Z_max_b": float(points[pair[1], 1]),
                    "R_max_mid": float(midpoint[0]),
                    "Z_max_mid": float(midpoint[1]),
                    "normalized_distance": float(d),
                }
            )
            used += 1
            if used >= int(nearest_k):
                break
    return pair_records


def _combine_with_base_training(base_training_df: pd.DataFrame, new_lf_df: pd.DataFrame, *, keep_original_lf: bool = True) -> pd.DataFrame:
    hf = base_training_df[base_training_df["fidelity"].astype(int) == 1].copy()
    lf_original = base_training_df[base_training_df["fidelity"].astype(int) == 0].copy()
    combined_lf = pd.concat([lf_original, new_lf_df], ignore_index=True, sort=False) if keep_original_lf else new_lf_df.copy()
    expected_cols = list(base_training_df.columns)
    for col in expected_cols:
        if col not in combined_lf.columns:
            combined_lf[col] = np.nan
        if col not in hf.columns:
            hf[col] = np.nan
    combined = pd.concat([combined_lf[expected_cols], hf[expected_cols]], ignore_index=True, sort=False)
    combined["fidelity"] = combined["fidelity"].astype(int)
    combined["iteration"] = combined["iteration"].astype(int)
    return combined


def build_midpoint_outputs(
    *,
    config_path: str | Path | None = None,
    model_path: str | Path | None = None,
    nearest_k: int = 1,
    min_norm_dist: float = 0.05,
    max_norm_dist: float = 0.25,
    mc_samples: int = 30,
    chunk_size: int = 20000,
    keep_original_lf: bool = True,
    random_state: int = 42,
    device: Optional[str] = None,
) -> Dict[str, Path]:
    paths = default_paths(config_path, model_path)
    paths.artifact_dir.mkdir(parents=True, exist_ok=True)

    runtime = load_runtime_config(paths.config_path, seed=random_state)
    set_seed(random_state)
    model = load_model_checkpoint(paths.model_path, device=device)

    base_training_df = pd.read_csv(paths.training_cnp_csv)
    file_map = _build_source_table_map(paths.train_dir)

    original_theta_df = base_training_df[base_training_df["fidelity"].astype(int) == 0][["source_file", "R_max", "Z_max"]].copy()
    pair_records = _build_midpoint_pairs(
        original_theta_df,
        theta_max=runtime.theta_max,
        nearest_k=nearest_k,
        min_norm_dist=min_norm_dist,
        max_norm_dist=max_norm_dist,
    )
    if not pair_records:
        raise RuntimeError("No valid midpoint pairs were found with the current nearest-neighbor distance filter.")

    all_trial_rows: List[Dict[str, float | int | str]] = []
    all_per_signal_parts: List[pd.DataFrame] = []
    manifest_rows: List[Dict[str, float | int | str]] = []

    for pair_idx, pair in enumerate(pair_records):
        path_a = file_map[str(pair["source_file_a"])]
        df_block = _read_table(path_a)
        missing = [c for c in _required_block_columns() if c not in df_block.columns]
        if missing:
            raise ValueError(f"{path_a} is missing required columns: {missing}")

        midpoint_r = float(pair["R_max_mid"])
        midpoint_z = float(pair["Z_max_mid"])
        synthetic_name = f"{Path(path_a).stem}__midpoint_{pair_idx:03d}"

        row, per_signal = _predict_trial_for_theta(
            df_block,
            source_name=synthetic_name,
            iteration=0,
            theta_r=midpoint_r,
            theta_z=midpoint_z,
            model=model,
            context_ratio=runtime.context_ratio,
            mc_samples=mc_samples,
            chunk_size=chunk_size,
            seed=random_state + pair_idx * 4099,
        )
        all_trial_rows.append(row)
        all_per_signal_parts.append(per_signal)
        manifest_rows.append(
            {
                **pair,
                "source_file_synthetic": synthetic_name,
                "n_samples": int(len(df_block)),
                "y_raw_mid": float(row["y_raw"]),
                "y_cnp_mid": float(row["y_cnp"]),
                "y_cnp_err_mid": float(row["y_cnp_err"]),
            }
        )

    synthetic_trials_df = pd.DataFrame(all_trial_rows)
    per_signal_df = pd.concat(all_per_signal_parts, ignore_index=True) if all_per_signal_parts else pd.DataFrame()
    manifest_df = pd.DataFrame(manifest_rows)
    augmented_training_df = _combine_with_base_training(base_training_df, synthetic_trials_df, keep_original_lf=keep_original_lf)

    out = {
        "synthetic_trials_csv": paths.artifact_dir / "midpoint_lf_trials.csv",
        "synthetic_per_signal_csv": paths.artifact_dir / "midpoint_per_signal.csv",
        "midpoint_manifest_csv": paths.artifact_dir / "midpoint_theta_manifest.csv",
        "augmented_training_csv": paths.artifact_dir / "cnp_augmented_midpoint_training.csv",
        "baseline_config": paths.artifact_dir / "settings_baseline.yaml",
        "midpoint_config": paths.artifact_dir / "settings_midpoint.yaml",
        "summary_json": paths.artifact_dir / "midpoint_summary.json",
    }
    synthetic_trials_df.to_csv(out["synthetic_trials_csv"], index=False)
    per_signal_df.to_csv(out["synthetic_per_signal_csv"], index=False)
    manifest_df.to_csv(out["midpoint_manifest_csv"], index=False)
    augmented_training_df.to_csv(out["augmented_training_csv"], index=False)

    write_mfgp_config_variant(
        paths.config_path,
        output_path=out["baseline_config"],
        version_suffix="baseline",
        out_dir_mfgp=paths.artifact_dir / "mfgp_baseline",
    )
    write_mfgp_config_variant(
        paths.config_path,
        output_path=out["midpoint_config"],
        version_suffix="midpoint",
        out_dir_mfgp=paths.artifact_dir / "mfgp_midpoint",
    )

    summary = {
        "config_path": str(paths.config_path),
        "model_path": str(paths.model_path),
        "train_dir": str(paths.train_dir),
        "base_training_cnp_csv": str(paths.training_cnp_csv),
        "validation_cnp_csv": str(paths.validation_cnp_csv),
        "n_original_lf_trials": int((base_training_df["fidelity"].astype(int) == 0).sum()),
        "n_midpoint_lf_trials": int(len(synthetic_trials_df)),
        "nearest_k": int(nearest_k),
        "min_norm_dist": float(min_norm_dist),
        "max_norm_dist": float(max_norm_dist),
        "keep_original_lf": bool(keep_original_lf),
    }
    out["summary_json"].write_text(json.dumps(summary, indent=2))
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create LF midpoint-theta augmentation for MF-GP.")
    p.add_argument("--config", type=str, default=None, help="CNP config path. Defaults to minibatch config.")
    p.add_argument("--model-path", type=str, default=None, help="CNP checkpoint path. Defaults to minibatch model.")
    p.add_argument("--nearest-k", type=int, default=1, help="Nearest neighbors per theta to consider for midpoint pairs.")
    p.add_argument("--min-norm-dist", type=float, default=0.05, help="Minimum normalized theta distance to allow.")
    p.add_argument("--max-norm-dist", type=float, default=0.25, help="Maximum normalized theta distance to allow.")
    p.add_argument("--mc-samples", type=int, default=30, help="MC dropout samples for CNP prediction.")
    p.add_argument("--chunk-size", type=int, default=20000, help="Prediction chunk size.")
    p.add_argument("--random-state", type=int, default=42, help="Random seed.")
    p.add_argument("--device", type=str, default=None, help="Torch device override.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = build_midpoint_outputs(
        config_path=args.config,
        model_path=args.model_path,
        nearest_k=args.nearest_k,
        min_norm_dist=args.min_norm_dist,
        max_norm_dist=args.max_norm_dist,
        mc_samples=args.mc_samples,
        chunk_size=args.chunk_size,
        random_state=args.random_state,
        device=args.device,
    )
    print("Artifacts:")
    for key, path in out.items():
        print(f"- {key}: {path}")


if __name__ == "__main__":
    main()
