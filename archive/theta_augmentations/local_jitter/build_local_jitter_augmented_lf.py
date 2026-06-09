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
import yaml


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
class LocalJitterPaths:
    repo_root: Path
    config_path: Path
    model_path: Path
    train_dir: Path
    training_cnp_csv: Path
    validation_cnp_csv: Path
    artifact_dir: Path


def default_paths(config_path: str | Path | None = None, model_path: str | Path | None = None) -> LocalJitterPaths:
    lf_paths = default_lf_paths(config_path)
    runtime = load_runtime_config(lf_paths.config_path, seed=42)
    resolved_model = Path(model_path).expanduser().resolve() if model_path else (
        runtime.out_dir / f"cnp_{runtime.version}_model_{runtime.epochs}epochs.pth"
    ).resolve()
    artifact_dir = REPO_ROOT / "theta_augmentations" / "local_jitter" / "artifacts"
    return LocalJitterPaths(
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


def _sample_local_jitter_theta(
    original_r: float,
    original_z: float,
    *,
    rng: np.random.Generator,
    theta_min: Sequence[float],
    theta_max: Sequence[float],
    r_jitter_max: float,
    z_jitter_max: float,
    existing_keys: set[Tuple[float, float]],
    rounding_decimals: int = 6,
    max_tries: int = 100,
) -> Tuple[float, float]:
    for _ in range(max_tries):
        new_r = float(np.clip(original_r + rng.uniform(-r_jitter_max, r_jitter_max), theta_min[0], theta_max[0]))
        new_z = float(np.clip(original_z + rng.uniform(-z_jitter_max, z_jitter_max), theta_min[1], theta_max[1]))
        if abs(new_r - original_r) < 1e-9 and abs(new_z - original_z) < 1e-9:
            continue
        key = (round(new_r, rounding_decimals), round(new_z, rounding_decimals))
        if key in existing_keys:
            continue
        existing_keys.add(key)
        return new_r, new_z
    raise RuntimeError(
        f"Could not sample a unique jitter theta near ({original_r:.6f}, {original_z:.6f}) "
        f"after {max_tries} attempts."
    )


def _predict_trial_for_theta(
    df_block: pd.DataFrame,
    *,
    file_path: Path,
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
            "source_file": [file_path.name] * n,
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
        "source_file": file_path.name,
    }
    return row, per_signal


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


def build_local_jitter_outputs(
    *,
    config_path: str | Path | None = None,
    model_path: str | Path | None = None,
    n_variants_per_theta: int = 1,
    r_jitter_max: float = 25.0,
    z_jitter_max: float = 35.0,
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

    all_trial_rows: List[Dict[str, float | int | str]] = []
    all_per_signal_parts: List[pd.DataFrame] = []
    manifest_rows: List[Dict[str, float | int | str]] = []

    existing_keys = set(
        (round(float(r), 6), round(float(z), 6))
        for r, z in base_training_df.loc[base_training_df["fidelity"].astype(int) == 0, ["R_max", "Z_max"]].drop_duplicates().itertuples(index=False, name=None)
    )

    lf_source_names = (
        base_training_df.loc[base_training_df["fidelity"].astype(int) == 0, "source_file"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    for file_idx, source_name in enumerate(lf_source_names):
        if source_name not in file_map:
            raise KeyError(
                f"Could not resolve LF source_file '{source_name}' to an original CSV/parquet table in {paths.train_dir}"
            )
        file_path = file_map[source_name]
        df_block = _read_table(file_path)
        missing = [c for c in _required_block_columns() if c not in df_block.columns]
        if missing:
            raise ValueError(f"{file_path} is missing required columns: {missing}")

        original_r = float(df_block["R_max"].iloc[0])
        original_z = float(df_block["Z_max"].iloc[0])
        iteration = 0

        for variant_idx in range(int(n_variants_per_theta)):
            theta_r, theta_z = _sample_local_jitter_theta(
                original_r,
                original_z,
                rng=np.random.default_rng(random_state + file_idx * 1009 + variant_idx),
                theta_min=runtime.theta_min,
                theta_max=runtime.theta_max,
                r_jitter_max=r_jitter_max,
                z_jitter_max=z_jitter_max,
                existing_keys=existing_keys,
            )
            row, per_signal = _predict_trial_for_theta(
                df_block,
                file_path=file_path,
                iteration=iteration,
                theta_r=theta_r,
                theta_z=theta_z,
                model=model,
                context_ratio=runtime.context_ratio,
                mc_samples=mc_samples,
                chunk_size=chunk_size,
                seed=random_state + file_idx * 1009 + variant_idx,
            )
            synthetic_name = f"{file_path.stem}__jitter_{variant_idx:02d}"
            row["source_file"] = synthetic_name
            per_signal["source_file"] = synthetic_name
            all_trial_rows.append(row)
            all_per_signal_parts.append(per_signal)
            manifest_rows.append(
                {
                    "source_file_original": file_path.name,
                    "source_file_synthetic": synthetic_name,
                    "R_max_original": original_r,
                    "Z_max_original": original_z,
                    "R_max_synthetic": theta_r,
                    "Z_max_synthetic": theta_z,
                    "delta_R_max": theta_r - original_r,
                    "delta_Z_max": theta_z - original_z,
                    "n_samples": int(len(df_block)),
                    "y_raw_synthetic": float(row["y_raw"]),
                    "y_cnp_synthetic": float(row["y_cnp"]),
                    "y_cnp_err_synthetic": float(row["y_cnp_err"]),
                }
            )

    synthetic_trials_df = pd.DataFrame(all_trial_rows)
    per_signal_df = pd.concat(all_per_signal_parts, ignore_index=True) if all_per_signal_parts else pd.DataFrame()
    manifest_df = pd.DataFrame(manifest_rows)
    augmented_training_df = _combine_with_base_training(base_training_df, synthetic_trials_df, keep_original_lf=keep_original_lf)

    out = {
        "synthetic_trials_csv": paths.artifact_dir / "local_jitter_lf_trials.csv",
        "synthetic_per_signal_csv": paths.artifact_dir / "local_jitter_per_signal.csv",
        "theta_manifest_csv": paths.artifact_dir / "local_jitter_theta_manifest.csv",
        "augmented_training_csv": paths.artifact_dir / "cnp_augmented_local_jitter_training.csv",
        "baseline_config": paths.artifact_dir / "settings_baseline.yaml",
        "local_jitter_config": paths.artifact_dir / "settings_local_jitter.yaml",
        "summary_json": paths.artifact_dir / "local_jitter_summary.json",
    }
    synthetic_trials_df.to_csv(out["synthetic_trials_csv"], index=False)
    per_signal_df.to_csv(out["synthetic_per_signal_csv"], index=False)
    manifest_df.to_csv(out["theta_manifest_csv"], index=False)
    augmented_training_df.to_csv(out["augmented_training_csv"], index=False)

    write_mfgp_config_variant(
        paths.config_path,
        output_path=out["baseline_config"],
        version_suffix="baseline",
        out_dir_mfgp=paths.artifact_dir / "mfgp_baseline",
    )
    write_mfgp_config_variant(
        paths.config_path,
        output_path=out["local_jitter_config"],
        version_suffix="localjitter",
        out_dir_mfgp=paths.artifact_dir / "mfgp_local_jitter",
    )

    summary = {
        "config_path": str(paths.config_path),
        "model_path": str(paths.model_path),
        "train_dir": str(paths.train_dir),
        "base_training_cnp_csv": str(paths.training_cnp_csv),
        "validation_cnp_csv": str(paths.validation_cnp_csv),
        "n_original_lf_trials": int((base_training_df["fidelity"].astype(int) == 0).sum()),
        "n_synthetic_lf_trials": int(len(synthetic_trials_df)),
        "n_variants_per_theta": int(n_variants_per_theta),
        "keep_original_lf": bool(keep_original_lf),
        "r_jitter_max": float(r_jitter_max),
        "z_jitter_max": float(z_jitter_max),
    }
    out["summary_json"].write_text(json.dumps(summary, indent=2))
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create LF synthetic-theta local-jitter augmentation for MF-GP.")
    p.add_argument("--config", type=str, default=None, help="CNP config path. Defaults to minibatch config.")
    p.add_argument("--model-path", type=str, default=None, help="CNP checkpoint path. Defaults to minibatch model.")
    p.add_argument("--n-variants-per-theta", type=int, default=1, help="Synthetic jitter thetas per original LF theta.")
    p.add_argument("--r-jitter-max", type=float, default=25.0, help="Max absolute jitter applied to R_max.")
    p.add_argument("--z-jitter-max", type=float, default=35.0, help="Max absolute jitter applied to Z_max.")
    p.add_argument("--mc-samples", type=int, default=30, help="MC dropout samples for CNP prediction.")
    p.add_argument("--chunk-size", type=int, default=20000, help="Prediction chunk size.")
    p.add_argument("--random-state", type=int, default=42, help="Random seed.")
    p.add_argument("--device", type=str, default=None, help="Torch device override.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = build_local_jitter_outputs(
        config_path=args.config,
        model_path=args.model_path,
        n_variants_per_theta=args.n_variants_per_theta,
        r_jitter_max=args.r_jitter_max,
        z_jitter_max=args.z_jitter_max,
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
