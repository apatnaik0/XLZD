#!/usr/bin/env python3
"""
validate_cnp_mfgp.py

End-to-end holdout validation for the XLZD CNP -> MF-GP pipeline.

This script is intended for detector geometries that were NOT used to train the
CNP/MF-GP models.

Pipeline
--------
1. Read validation_data/validation_manifest.csv:
       filename,R,Z,z_center,fidelity
2. Read each raw event file containing:
       sx, sy, sz, x, y, z
3. Convert the raw events into the categorical H5 format expected by the CNP:
       theta        = [detector_R, detector_Z]
       phi          = [s_r, s_z_from_center]
       target_shell = zero-based truth shell from endpoint (x,y,z)
4. Load the trained CNP AND its fixed inference context from the saved .pth.
   Validation labels are never used as CNP context.
5. Run the self-contained CNP checkpoint on every held-out event.
6. Aggregate CNP predictions into the long-form CSV expected by the MF-GP.
7. Load the saved MF-GP checkpoint and evaluate it on the held-out validation geometries.
   No MF-GP fitting occurs in this validation script.
8. Save CNP metrics, MF-GP metrics, validation predictions, and a combined
   per-geometry summary.

Important
---------
Both CNP and MF-GP are loaded from saved model checkpoints. No training or
re-fitting occurs in this validation script.

Example
-------
python validate_cnp_mfgp.py \
    --raw-dir /path/to/validation_data \
    --cnp-model /path/to/cnp_model.pth \
    --mfgp-model /path/to/mfgp_model.joblib \
    --output-dir /path/to/holdout_validation
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm


# -----------------------------------------------------------------------------
# Imports from the existing project
# -----------------------------------------------------------------------------

try:
    from cnp_mfgp.cnp_clean_pipeline import (
        H5EventPool,
        load_cnp_inference_checkpoint,
        set_seed,
    )
except ImportError:
    from cnp_clean_pipeline import (
        H5EventPool,
        load_cnp_inference_checkpoint,
        set_seed,
    )

try:
    from cnp_mfgp.mfgp_clean_pipeline import (
        MFGPRuntimeConfig,
        _TRANSFORM_TITLES,
        _plot_mean_std_heatmaps,
        _plot_validation_hf_points,
        _plot_validation_parity_pointwise,
        _positive_prediction_grid,
        _predict_in_chunks,
        _prediction_interval_in_output_space,
        evaluate_mfgp_checkpoint,
        load_mfgp_checkpoint,
    )
except ImportError:
    from mfgp_clean_pipeline import (
        MFGPRuntimeConfig,
        _TRANSFORM_TITLES,
        _plot_mean_std_heatmaps,
        _plot_validation_hf_points,
        _plot_validation_parity_pointwise,
        _positive_prediction_grid,
        _predict_in_chunks,
        _prediction_interval_in_output_space,
        evaluate_mfgp_checkpoint,
        load_mfgp_checkpoint,
    )


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

VALIDATION_MANIFEST_NAME = "validation_manifest.csv"
MANIFEST_COLUMNS = ["filename", "R", "Z", "z_center", "fidelity"]
RAW_COLUMNS = ["sx", "sy", "sz", "x", "y", "z"]

R_THETA_NAMES = {
    "r",
    "r_max",
    "detector_r",
    "detector_radius",
    "r_shell",
    "r_boundary",
    "radius",
}

Z_THETA_NAMES = {
    "z",
    "z_max",
    "detector_z",
    "detector_height",
    "z_shell",
    "z_boundary",
    "height",
}


# -----------------------------------------------------------------------------
# Data containers
# -----------------------------------------------------------------------------

@dataclass
class PreparedValidationData:
    h5_dir: Path
    manifest: pd.DataFrame
    preparation_summary: pd.DataFrame


@dataclass
class CNPValidationResult:
    prediction_csv: Path
    geometry_metrics_csv: Path
    overall_metrics_json: Path
    parity_plot: Path
    shell_error_plot: Path


@dataclass
class MFGPValidationResult:
    metrics_json: Path
    validation_prediction_csv: Path
    combined_summary_csv: Path
    grid_csv: Path
    mean_std_plot: Path
    validation_point_plot: Path
    validation_parity_plot: Path


# -----------------------------------------------------------------------------
# General helpers
# -----------------------------------------------------------------------------

def _numeric_frame(df: pd.DataFrame, columns: Sequence[str], context: str) -> pd.DataFrame:
    out = df.copy()
    for column in columns:
        out[column] = pd.to_numeric(out[column], errors="coerce")

    bad = ~np.isfinite(out[list(columns)].to_numpy(dtype=float)).all(axis=1)
    if bad.any():
        n_bad = int(bad.sum())
        print(f"[warn] {context}: dropping {n_bad:,} rows with missing/non-finite required values")
        out = out.loc[~bad].copy()

    return out.reset_index(drop=True)


def _read_raw_event_file(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
    elif suffix in {".h5", ".hdf5"}:
        # This is for pandas-style raw event HDF5 files, not the CNP block H5.
        try:
            df = pd.read_hdf(path)
        except (ValueError, KeyError) as exc:
            raise ValueError(
                f"{path}: could not read as a pandas raw-event HDF5 file. "
                "Validation source files should contain tabular sx,sy,sz,x,y,z columns."
            ) from exc
    else:
        raise ValueError(
            f"Unsupported validation event file type {path.suffix!r}: {path}. "
            "Supported: .csv, .parquet/.pq, pandas .h5/.hdf5."
        )

    missing = sorted(set(RAW_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"{path.name}: missing required raw columns {missing}")

    return _numeric_frame(df, RAW_COLUMNS, path.name)


def load_validation_manifest(raw_dir: str | Path) -> tuple[pd.DataFrame, Path]:
    """
    Load validation_manifest.csv from raw_dir.

    Expected layout:

        raw_dir/
        ├── validation_manifest.csv
        ├── geometry_A.csv
        ├── geometry_B.csv
        └── ...

    Every filename in the manifest is resolved from raw_dir.
    """
    base_dir = Path(raw_dir).expanduser().resolve()

    if not base_dir.exists():
        raise FileNotFoundError(
            f"Validation data directory does not exist: {base_dir}"
        )
    if not base_dir.is_dir():
        raise NotADirectoryError(
            f"--raw-dir must point to a directory: {base_dir}"
        )

    manifest_path = base_dir / VALIDATION_MANIFEST_NAME
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Validation manifest not found: {manifest_path}\n"
            f"Expected {VALIDATION_MANIFEST_NAME!r} inside --raw-dir."
        )

    manifest = pd.read_csv(
        manifest_path,
        na_values=["", "None", "none", "NULL", "null", "NaN", "nan"],
        keep_default_na=True,
    )

    missing = sorted(set(MANIFEST_COLUMNS) - set(manifest.columns))
    if missing:
        raise ValueError(
            f"Validation manifest is missing required columns {missing}. "
            f"Expected at least {MANIFEST_COLUMNS}."
        )

    manifest = manifest.copy()
    manifest["filename"] = manifest["filename"].astype(str).str.strip()

    for column in ["R", "Z", "z_center", "fidelity"]:
        manifest[column] = pd.to_numeric(
            manifest[column], errors="coerce"
        )

    if manifest[MANIFEST_COLUMNS].isna().any().any():
        bad_rows = manifest.index[
            manifest[MANIFEST_COLUMNS].isna().any(axis=1)
        ].tolist()
        raise ValueError(
            "Validation manifest has missing/non-numeric required "
            f"values at rows {bad_rows}"
        )

    manifest["R"] = manifest["R"].astype(float)
    manifest["Z"] = manifest["Z"].astype(float)
    manifest["z_center"] = manifest["z_center"].astype(float)
    manifest["fidelity"] = manifest["fidelity"].astype(int)

    invalid_fidelity = ~manifest["fidelity"].isin([0, 1])
    if invalid_fidelity.any():
        bad = manifest.loc[invalid_fidelity, "fidelity"].tolist()
        raise ValueError(
            f"fidelity must be exactly 0 or 1; found {bad}"
        )

    if (manifest["R"] <= 0).any() or (manifest["Z"] <= 0).any():
        raise ValueError(
            "Manifest detector R and Z values must be positive."
        )

    for filename in manifest["filename"]:
        # Validation files are always expected directly inside raw_dir.
        file_path = base_dir / Path(filename).name
        if not file_path.exists():
            raise FileNotFoundError(
                f"Manifest entry points to missing file: {file_path}"
            )

    return manifest.reset_index(drop=True), base_dir

def shell_boundary_scales(n_shells: int, scale_power: float) -> np.ndarray:
    if n_shells <= 0:
        raise ValueError("n_shells must be positive")
    if scale_power <= 0:
        raise ValueError("scale_power must be positive")

    fraction = np.arange(1, n_shells + 1, dtype=float) / float(n_shells)
    return np.power(fraction, float(scale_power))


def assign_target_shell(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    detector_R: float,
    detector_Z: float,
    z_center: float,
    n_shells: int,
    scale_power: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (target_shell, valid_mask).

    Shell boundaries match the preprocessing convention:
        R_i = R * (i / n_shells)^scale_power
        Z_i = Z * (i / n_shells)^scale_power

    The target is determined from the ENDPOINT:
        r = sqrt(x^2 + y^2)
        z_from_center = abs(z - z_center)

    target_shell is zero-based.
    """
    r = np.sqrt(np.square(x) + np.square(y))
    z_from_center = np.abs(z - float(z_center))

    radial_fraction = r / float(detector_R)
    axial_fraction = z_from_center / float(detector_Z)
    required_scale = np.maximum(radial_fraction, axial_fraction)

    scales = shell_boundary_scales(n_shells, scale_power)
    target = np.searchsorted(scales, required_scale, side="left").astype(np.int64)

    valid = (
        np.isfinite(required_scale)
        & (required_scale >= 0.0)
        & (required_scale <= 1.0 + 1e-12)
        & (target >= 0)
        & (target < n_shells)
    )

    return target, valid


def _theta_matrix(
    n_rows: int,
    theta_headers: Sequence[str],
    detector_R: float,
    detector_Z: float,
) -> np.ndarray:
    columns: list[np.ndarray] = []

    for name in theta_headers:
        key = str(name).strip().lower()

        if key in R_THETA_NAMES:
            value = float(detector_R)
        elif key in Z_THETA_NAMES:
            value = float(detector_Z)
        else:
            raise ValueError(
                f"Cannot derive theta header {name!r}. "
                "This validator currently knows detector-radius and detector-Z theta features."
            )

        columns.append(np.full(n_rows, value, dtype=np.float32))

    return np.column_stack(columns).astype(np.float32)


def _phi_matrix(
    df: pd.DataFrame,
    phi_headers: Sequence[str],
    z_center: float,
) -> np.ndarray:
    derived: dict[str, np.ndarray] = {
        "s_r": np.sqrt(
            np.square(df["sx"].to_numpy(dtype=float))
            + np.square(df["sy"].to_numpy(dtype=float))
        ),
        "s_z_from_center": np.abs(df["sz"].to_numpy(dtype=float) - float(z_center)),
    }

    columns: list[np.ndarray] = []
    for name in phi_headers:
        if name in df.columns:
            values = pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=float)
        elif name in derived:
            values = derived[name]
        else:
            raise ValueError(
                f"Cannot derive CNP phi feature {name!r}. "
                "Available raw columns are sx,sy,sz,x,y,z and derived features "
                "s_r, s_z_from_center."
            )
        columns.append(values.astype(np.float32))

    phi = np.column_stack(columns).astype(np.float32)
    if not np.isfinite(phi).all():
        raise ValueError("Derived CNP phi matrix contains non-finite values.")

    return phi


def write_cnp_h5(
    output_path: Path,
    *,
    theta: np.ndarray,
    phi: np.ndarray,
    target_shell: np.ndarray,
    theta_headers: Sequence[str],
    phi_headers: Sequence[str],
    fidelity: int,
    source_file: str,
    detector_R: float,
    detector_Z: float,
    z_center: float,
    original_event_index: np.ndarray,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    theta = np.asarray(theta, dtype=np.float32)
    phi = np.asarray(phi, dtype=np.float32)
    target_shell = np.asarray(target_shell, dtype=np.int64).reshape(-1)

    n = len(target_shell)
    if len(theta) != n or len(phi) != n:
        raise ValueError("theta/phi/target_shell row mismatch while writing validation H5")

    with h5py.File(output_path, "w") as f:
        f.create_dataset("theta", data=theta, compression="gzip", compression_opts=4)
        f.create_dataset("phi", data=phi, compression="gzip", compression_opts=4)
        f.create_dataset("target_shell", data=target_shell, compression="gzip", compression_opts=4)

        f.create_dataset("theta_labels", data=np.asarray(theta_headers, dtype="S"))
        f.create_dataset("phi_labels", data=np.asarray(phi_headers, dtype="S"))
        f.create_dataset("target_headers", data=np.asarray(["target_shell"], dtype="S"))

        meta = f.create_group("meta")
        meta.create_dataset("event_index", data=np.arange(n, dtype=np.int64), compression="gzip")
        meta.create_dataset(
            "original_event_id",
            data=np.asarray(original_event_index, dtype=np.int64),
            compression="gzip",
        )
        meta.create_dataset(
            "shell_index",
            data=target_shell.astype(np.int64) + 1,
            compression="gzip",
        )
        meta.create_dataset(
            "fidelity",
            data=np.full(n, int(fidelity), dtype=np.int32),
            compression="gzip",
        )
        meta.create_dataset(
            "source_file",
            data=np.asarray([source_file] * n, dtype="S"),
            compression="gzip",
        )
        meta.create_dataset(
            "split_name",
            data=np.asarray(["external_validation"] * n, dtype="S"),
            compression="gzip",
        )
        meta.create_dataset(
            "detector_R",
            data=np.full(n, float(detector_R), dtype=np.float32),
            compression="gzip",
        )
        meta.create_dataset(
            "detector_Z",
            data=np.full(n, float(detector_Z), dtype=np.float32),
            compression="gzip",
        )
        meta.create_dataset(
            "detector_z_center",
            data=np.full(n, float(z_center), dtype=np.float32),
            compression="gzip",
        )


def _cnp_spec_from_checkpoint(metadata: dict) -> dict[str, object]:
    """Extract and validate the raw-data preprocessing contract from the CNP checkpoint."""
    required = {
        "theta_headers",
        "phi_headers",
        "target_headers",
        "n_shells",
        "scale_power",
    }
    missing = sorted(required - set(metadata))
    if missing:
        raise ValueError(
            "The CNP model is not a portable validation checkpoint. "
            f"Missing metadata: {missing}. Re-save/retrain it with the updated "
            "cnp_clean_pipeline.py."
        )

    return {
        "theta_headers": list(metadata["theta_headers"]),
        "phi_headers": list(metadata["phi_headers"]),
        "target_headers": list(metadata["target_headers"]),
        "n_shells": int(metadata["n_shells"]),
        "scale_power": float(metadata["scale_power"]),
    }


# -----------------------------------------------------------------------------
# Step 1: raw validation data -> CNP H5
# -----------------------------------------------------------------------------

def prepare_validation_h5(
    *,
    raw_dir: str | Path,
    cnp_metadata: dict,
    output_dir: str | Path,
    overwrite: bool,
) -> PreparedValidationData:
    spec = _cnp_spec_from_checkpoint(cnp_metadata)
    manifest, base_dir = load_validation_manifest(raw_dir)

    h5_dir = Path(output_dir).expanduser().resolve() / "validation_h5"
    if h5_dir.exists() and overwrite:
        shutil.rmtree(h5_dir)
    h5_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, object]] = []

    for row_index, row in manifest.iterrows():
        filename = str(row["filename"])
        input_path = base_dir / Path(filename).name

        detector_R = float(row["R"])
        detector_Z = float(row["Z"])
        z_center = float(row["z_center"])
        fidelity = int(row["fidelity"])

        print(
            f"[prepare] {filename} | R={detector_R:g}, Z={detector_Z:g}, "
            f"z_center={z_center:g}, fidelity={fidelity}"
        )

        raw = _read_raw_event_file(input_path)
        raw["__original_event_index"] = np.arange(len(raw), dtype=np.int64)

        target_shell, valid = assign_target_shell(
            raw["x"].to_numpy(dtype=float),
            raw["y"].to_numpy(dtype=float),
            raw["z"].to_numpy(dtype=float),
            detector_R=detector_R,
            detector_Z=detector_Z,
            z_center=z_center,
            n_shells=int(spec["n_shells"]),
            scale_power=float(spec["scale_power"]),
        )

        n_input = len(raw)
        n_dropped = int((~valid).sum())

        raw_valid = raw.loc[valid].reset_index(drop=True)
        target_valid = target_shell[valid]

        if len(raw_valid) == 0:
            raise ValueError(
                f"{filename}: no events remained after endpoint shell assignment. "
                "Check R, Z, z_center, and endpoint coordinates."
            )

        theta = _theta_matrix(
            len(raw_valid),
            spec["theta_headers"],
            detector_R,
            detector_Z,
        )
        phi = _phi_matrix(
            raw_valid,
            spec["phi_headers"],
            z_center,
        )

        safe_stem = input_path.stem.replace(" ", "_")
        output_path = h5_dir / f"{row_index:04d}_{safe_stem}_validation.h5"

        write_cnp_h5(
            output_path,
            theta=theta,
            phi=phi,
            target_shell=target_valid,
            theta_headers=spec["theta_headers"],
            phi_headers=spec["phi_headers"],
            fidelity=fidelity,
            source_file=filename,
            detector_R=detector_R,
            detector_Z=detector_Z,
            z_center=z_center,
            original_event_index=raw_valid["__original_event_index"].to_numpy(dtype=np.int64),
        )

        summary_rows.append(
            {
                "manifest_row": row_index,
                "filename": filename,
                "fidelity": fidelity,
                "R": detector_R,
                "Z": detector_Z,
                "z_center": z_center,
                "input_events": n_input,
                "saved_events": len(raw_valid),
                "dropped_outside_detector": n_dropped,
                "h5_file": str(output_path),
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary_path = Path(output_dir).expanduser().resolve() / "preparation_summary.csv"
    summary.to_csv(summary_path, index=False)

    return PreparedValidationData(
        h5_dir=h5_dir,
        manifest=manifest,
        preparation_summary=summary,
    )


# -----------------------------------------------------------------------------
# Step 2: strict-holdout CNP validation prediction
# -----------------------------------------------------------------------------


def _js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)

    p = np.maximum(p, 0.0) + eps
    q = np.maximum(q, 0.0) + eps
    p = p / p.sum()
    q = q / q.sum()

    m = 0.5 * (p + q)
    return float(
        0.5 * np.sum(p * np.log(p / m))
        + 0.5 * np.sum(q * np.log(q / m))
    )


def _decode_first(value: np.ndarray | object, default: str) -> str:
    arr = np.asarray(value).reshape(-1)
    if arr.size == 0:
        return default
    item = arr[0]
    if isinstance(item, (bytes, np.bytes_)):
        return item.decode("utf-8")
    return str(item)


def run_cnp_holdout_validation(
    *,
    model: torch.nn.Module,
    context_x: torch.Tensor,
    context_y: torch.Tensor,
    cnp_metadata: dict,
    validation_h5_dir: str | Path,
    output_dir: str | Path,
    mc_samples: int,
    chunk_size: int,
    iteration: int,
    seed: int,
) -> CNPValidationResult:
    spec = _cnp_spec_from_checkpoint(cnp_metadata)
    set_seed(seed)
    dev = next(model.parameters()).device

    if context_x.shape[1] != int(cnp_metadata["x_dim"]):
        raise ValueError(
            f"Saved context x_dim={context_x.shape[1]} does not match "
            f"saved CNP x_dim={cnp_metadata['x_dim']}"
        )
    if int(cnp_metadata["y_dim"]) != int(spec["n_shells"]):
        raise ValueError(
            f"Saved CNP y_dim={cnp_metadata['y_dim']} does not match "
            f"n_shells={spec['n_shells']}"
        )

    validation_pool = H5EventPool(
        validation_h5_dir,
        theta_headers=spec["theta_headers"],
        phi_headers=spec["phi_headers"],
        target_headers=spec["target_headers"],
        n_shells=int(spec["n_shells"]),
        seed=seed + 10000,
        cache_files=False,
    )

    output_dir = Path(output_dir).expanduser().resolve()
    cnp_out_dir = output_dir / "cnp"
    cnp_out_dir.mkdir(parents=True, exist_ok=True)

    aggregate_rows: list[dict[str, object]] = []
    event_metric_rows: list[dict[str, object]] = []

    theta_dim = len(spec["theta_headers"])
    shell_indices = np.arange(1, int(spec["n_shells"]) + 1, dtype=np.int64)

    files = list(validation_pool.iter_file_data())
    for file_path, x_np, target_shell_np, meta in tqdm(
        files,
        total=len(files),
        desc="CNP holdout validation",
        unit="file",
    ):
        n_events = len(target_shell_np)
        if n_events == 0:
            continue

        theta_values = x_np[0, :theta_dim].astype(float)
        fidelity_values = np.asarray(meta["fidelity"]).reshape(-1)
        fidelity = int(fidelity_values[0])
        source_file = _decode_first(meta.get("source_file", np.asarray([file_path.name])), file_path.name)

        prob_sum = np.zeros(int(spec["n_shells"]), dtype=np.float64)
        std_sq_sum = np.zeros(int(spec["n_shells"]), dtype=np.float64)
        truth_counts = np.zeros(int(spec["n_shells"]), dtype=np.int64)

        top1_correct = 0
        abs_shell_error_sum = 0.0
        p_true_sum = 0.0

        for start in range(0, n_events, max(1, int(chunk_size))):
            end = min(n_events, start + max(1, int(chunk_size)))
            target_x = torch.from_numpy(x_np[start:end]).to(dev)
            truth_chunk = target_shell_np[start:end].astype(np.int64)

            with torch.no_grad():
                prob_t, std_t = model.predict_proba_mc(
                    context_x,
                    context_y,
                    target_x,
                    mc_samples=int(mc_samples),
                )

            probs = prob_t.detach().cpu().numpy()
            stds = std_t.detach().cpu().numpy()

            pred_chunk = np.argmax(probs, axis=1).astype(np.int64)
            rows = np.arange(len(truth_chunk))

            top1_correct += int(np.sum(pred_chunk == truth_chunk))
            abs_shell_error_sum += float(np.abs(pred_chunk - truth_chunk).sum())
            p_true_sum += float(probs[rows, truth_chunk].sum())

            prob_sum += probs.sum(axis=0)
            std_sq_sum += np.square(stds).sum(axis=0)
            truth_counts += np.bincount(truth_chunk, minlength=int(spec["n_shells"]))

        y_cnp = prob_sum / float(n_events)
        y_cnp_err = np.sqrt(std_sq_sum / float(n_events))
        y_raw = truth_counts.astype(float) / float(n_events)

        eps = 1e-6
        p = np.clip(y_cnp, eps, 1.0 - eps)
        y = np.clip(y_raw, 0.0, 1.0)
        log_prop = y * np.log(p) + (1.0 - y) * np.log(1.0 - p)
        bce = -log_prop

        base = {
            "iteration": int(iteration),
            "fidelity": fidelity,
            "n_samples": int(n_events),
            "source_file": source_file,
        }
        for theta_name, theta_value in zip(spec["theta_headers"], theta_values):
            base[theta_name] = float(theta_value)

        for shell_i in range(int(spec["n_shells"])):
            aggregate_rows.append(
                {
                    **base,
                    "shell_index": int(shell_indices[shell_i]),
                    "y_cnp": float(y_cnp[shell_i]),
                    "y_cnp_err": float(y_cnp_err[shell_i]),
                    "y_raw": float(y_raw[shell_i]),
                    "log_prop": float(log_prop[shell_i]),
                    "bce": float(bce[shell_i]),
                }
            )

        event_metric_rows.append(
            {
                **base,
                "top1_accuracy": top1_correct / float(n_events),
                "mean_abs_shell_error": abs_shell_error_sum / float(n_events),
                "mean_true_shell_probability": p_true_sum / float(n_events),
                "distribution_mae": float(np.mean(np.abs(y_cnp - y_raw))),
                "distribution_rmse": float(np.sqrt(np.mean(np.square(y_cnp - y_raw)))),
                "js_divergence": _js_divergence(y_raw, y_cnp),
            }
        )

    if not aggregate_rows:
        raise RuntimeError("CNP holdout validation produced no predictions.")

    long_df = pd.DataFrame(aggregate_rows)

    # If multiple source files share the exact same geometry/fidelity, combine them
    # with event-count weighting so the MF-GP sees one shell distribution per point.
    group_cols = ["iteration", "fidelity", *spec["theta_headers"], "shell_index"]

    long_df["_y_cnp_sum"] = long_df["y_cnp"] * long_df["n_samples"]
    long_df["_y_raw_sum"] = long_df["y_raw"] * long_df["n_samples"]
    long_df["_err_sq_sum"] = np.square(long_df["y_cnp_err"]) * long_df["n_samples"]

    grouped = (
        long_df.groupby(group_cols, as_index=False)
        .agg(
            n_samples=("n_samples", "sum"),
            y_cnp_sum=("_y_cnp_sum", "sum"),
            y_raw_sum=("_y_raw_sum", "sum"),
            err_sq_sum=("_err_sq_sum", "sum"),
            source_file=("source_file", lambda s: ";".join(sorted(set(map(str, s))))),
        )
    )

    grouped["y_cnp"] = grouped["y_cnp_sum"] / grouped["n_samples"]
    grouped["y_raw"] = grouped["y_raw_sum"] / grouped["n_samples"]
    grouped["y_cnp_err"] = np.sqrt(grouped["err_sq_sum"] / grouped["n_samples"])

    p = np.clip(grouped["y_cnp"].to_numpy(dtype=float), 1e-6, 1.0 - 1e-6)
    y = np.clip(grouped["y_raw"].to_numpy(dtype=float), 0.0, 1.0)
    grouped["log_prop"] = y * np.log(p) + (1.0 - y) * np.log(1.0 - p)
    grouped["bce"] = -grouped["log_prop"]

    grouped = grouped[
        [
            "iteration",
            "fidelity",
            "n_samples",
            *spec["theta_headers"],
            "shell_index",
            "y_cnp",
            "y_cnp_err",
            "y_raw",
            "log_prop",
            "bce",
            "source_file",
        ]
    ].sort_values([*spec["theta_headers"], "fidelity", "shell_index"])

    prediction_csv = cnp_out_dir / "cnp_holdout_shell_distributions.csv"
    grouped.to_csv(prediction_csv, index=False)

    event_metrics = pd.DataFrame(event_metric_rows)

    # Combine duplicate geometry rows using event-count weighting.
    geom_group = ["fidelity", *spec["theta_headers"]]
    weighted_cols = [
        "top1_accuracy",
        "mean_abs_shell_error",
        "mean_true_shell_probability",
        "distribution_mae",
        "distribution_rmse",
        "js_divergence",
    ]
    for col in weighted_cols:
        event_metrics[f"_{col}_weighted"] = event_metrics[col] * event_metrics["n_samples"]

    agg_spec: dict[str, tuple[str, str | callable]] = {
        "n_samples": ("n_samples", "sum"),
        "source_file": ("source_file", lambda s: ";".join(sorted(set(map(str, s))))),
    }
    for col in weighted_cols:
        agg_spec[f"_{col}_sum"] = (f"_{col}_weighted", "sum")

    geometry_metrics = event_metrics.groupby(geom_group, as_index=False).agg(**agg_spec)

    for col in weighted_cols:
        geometry_metrics[col] = geometry_metrics[f"_{col}_sum"] / geometry_metrics["n_samples"]
        geometry_metrics.drop(columns=f"_{col}_sum", inplace=True)

    geometry_metrics_csv = cnp_out_dir / "cnp_holdout_geometry_metrics.csv"
    geometry_metrics.to_csv(geometry_metrics_csv, index=False)

    # Overall event-level metrics, weighted by number of events in each file/geometry.
    total_n = float(event_metrics["n_samples"].sum())
    overall = {
        "n_validation_events": int(total_n),
        "n_validation_files": int(len(event_metrics)),
        "n_unique_geometries": int(
            geometry_metrics[list(spec["theta_headers"])].drop_duplicates().shape[0]
        ),
        "top1_accuracy": float(
            np.sum(event_metrics["top1_accuracy"] * event_metrics["n_samples"]) / total_n
        ),
        "mean_abs_shell_error": float(
            np.sum(event_metrics["mean_abs_shell_error"] * event_metrics["n_samples"]) / total_n
        ),
        "mean_true_shell_probability": float(
            np.sum(event_metrics["mean_true_shell_probability"] * event_metrics["n_samples"]) / total_n
        ),
        "mean_geometry_distribution_mae": float(geometry_metrics["distribution_mae"].mean()),
        "mean_geometry_distribution_rmse": float(geometry_metrics["distribution_rmse"].mean()),
        "mean_geometry_js_divergence": float(geometry_metrics["js_divergence"].mean()),
        "context_source": "embedded_in_cnp_checkpoint",
        "context_size": int(len(context_x)),
        "mc_samples": int(mc_samples),
    }

    overall_metrics_json = cnp_out_dir / "cnp_holdout_overall_metrics.json"
    overall_metrics_json.write_text(json.dumps(overall, indent=2))

    # Plot aggregate shell occupation parity.
    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    ax.scatter(grouped["y_raw"], grouped["y_cnp"], s=14, alpha=0.6)
    lo = float(min(grouped["y_raw"].min(), grouped["y_cnp"].min()))
    hi = float(max(grouped["y_raw"].max(), grouped["y_cnp"].max()))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlabel("Simulation truth shell occupation")
    ax.set_ylabel("CNP predicted shell occupation")
    ax.set_title("Held-out CNP shell-distribution parity")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    parity_plot = cnp_out_dir / "cnp_holdout_shell_parity.png"
    fig.savefig(parity_plot, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.hist(
        event_metrics["mean_abs_shell_error"],
        bins=min(30, max(5, len(event_metrics))),
        alpha=0.8,
    )
    ax.set_xlabel("Mean |predicted shell - true shell| per validation file")
    ax.set_ylabel("Count")
    ax.set_title("Held-out CNP shell-index error")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    shell_error_plot = cnp_out_dir / "cnp_holdout_shell_error.png"
    fig.savefig(shell_error_plot, dpi=180)
    plt.close(fig)

    print(f"[CNP] predictions -> {prediction_csv}")
    print(f"[CNP] geometry metrics -> {geometry_metrics_csv}")
    print(f"[CNP] overall metrics -> {overall_metrics_json}")

    return CNPValidationResult(
        prediction_csv=prediction_csv,
        geometry_metrics_csv=geometry_metrics_csv,
        overall_metrics_json=overall_metrics_json,
        parity_plot=parity_plot,
        shell_error_plot=shell_error_plot,
    )


def _saved_mfgp_training_theta(bundle: dict) -> tuple[np.ndarray, np.ndarray]:
    """Recover LF/HF detector-theta points stored inside the fitted sklearn GPs."""
    model = bundle["model"]
    if model.x_scaler is None or model.gp_lf is None or model.gp_d is None:
        raise ValueError(
            "Saved MF-GP does not contain the fitted scaler/GP state needed "
            "to reconstruct its plotting domain."
        )

    x_lf = model.x_scaler.inverse_transform(
        np.asarray(model.gp_lf.X_train_, dtype=float)
    )
    x_hf = model.x_scaler.inverse_transform(
        np.asarray(model.gp_d.X_train_, dtype=float)
    )
    return np.asarray(x_lf, dtype=float), np.asarray(x_hf, dtype=float)


def _saved_mfgp_plot_runtime(
    *,
    bundle: dict,
    model_path: str | Path,
    output_dir: Path,
    x_lf: np.ndarray,
    x_hf: np.ndarray,
) -> MFGPRuntimeConfig:
    """Build the runtime object required by the normal MF-GP grid helper."""
    theta_headers = list(bundle["theta_headers"])
    observed = np.vstack([x_lf, x_hf]).astype(float)

    saved_min = bundle.get("theta_min")
    saved_max = bundle.get("theta_max")

    if (
        isinstance(saved_min, (list, tuple))
        and len(saved_min) == len(theta_headers)
        and np.all(np.isfinite(np.asarray(saved_min, dtype=float)))
    ):
        theta_min = [float(x) for x in saved_min]
    else:
        theta_min = np.nanmin(observed, axis=0).astype(float).tolist()

    if (
        isinstance(saved_max, (list, tuple))
        and len(saved_max) == len(theta_headers)
        and np.all(np.isfinite(np.asarray(saved_max, dtype=float)))
    ):
        theta_max = [float(x) for x in saved_max]
    else:
        theta_max = np.nanmax(observed, axis=0).astype(float).tolist()

    return MFGPRuntimeConfig(
        config_path=Path(model_path).expanduser().resolve(),
        version=str(bundle.get("version", "saved_model")),
        sim_type="saved_model_validation",
        theta_headers=theta_headers,
        theta_min=theta_min,
        theta_max=theta_max,
        n_shells=max(int(bundle.get("shell_n", 1)), 1),
        pca_components=0.995,
        pca_epsilon=1e-8,
        distribution_mc_samples=500,
        out_dir_cnp=output_dir,
        out_dir_mfgp=output_dir,
    )


# -----------------------------------------------------------------------------
# Step 4: MF-GP validation using the existing pipeline
# -----------------------------------------------------------------------------

def run_mfgp_validation(
    *,
    mfgp_model_path: str | Path,
    validation_cnp_csv: str | Path,
    output_dir: str | Path,
    predict_chunk_size: int,
    grid_points_per_axis: int = 120,
) -> MFGPValidationResult:
    """Evaluate a saved MF-GP and make the same three plots as run_clean_mfgp()."""
    output_dir = Path(output_dir).expanduser().resolve()
    mfgp_out = output_dir / "mfgp"
    mfgp_out.mkdir(parents=True, exist_ok=True)

    bundle = load_mfgp_checkpoint(mfgp_model_path)
    model = bundle["model"]
    x_cols = list(bundle["theta_headers"])
    transform_mode = str(bundle["target_transform"])
    experiment_title = _TRANSFORM_TITLES.get(
        transform_mode,
        f"MF-GP ({transform_mode})",
    )
    tag = f"{bundle['version']}_{transform_mode}"
    iteration = int(bundle["iteration"])
    use_log_hf = bool(bundle["use_log_hf"])

    # Held-out validation predictions + metrics.
    validation_prediction_csv = (
        mfgp_out
        / f"mfgp_{tag}_savedmodel_validation_predictions_iter{iteration}.csv"
    )
    metrics_json = (
        mfgp_out
        / f"mfgp_{tag}_savedmodel_validation_metrics_iter{iteration}.json"
    )

    predictions, metrics = evaluate_mfgp_checkpoint(
        mfgp_model_path,
        validation_cnp_csv,
        output_csv=validation_prediction_csv,
        metrics_json=metrics_json,
        chunk_size=int(predict_chunk_size),
    )

    combined_summary_csv = mfgp_out / "mfgp_holdout_validation_summary.csv"
    predictions.to_csv(combined_summary_csv, index=False)

    x_val = predictions[x_cols].to_numpy(dtype=float)
    y_val = predictions["y_raw"].to_numpy(dtype=float)
    val_mean_model = predictions["mf_mean_model_space"].to_numpy(dtype=float)
    val_std_model = predictions["mf_std_model_space"].to_numpy(dtype=float)
    validation_for_plot = predictions[x_cols + ["y_raw"]].copy()

    # Same validation point plot as normal training.
    validation_point_plot = (
        mfgp_out / f"mfgp_{tag}_validation_hf_points_iter{iteration}.png"
    )
    _plot_validation_hf_points(
        validation_for_plot,
        x_cols,
        val_mean_model,
        val_std_model,
        validation_point_plot,
        use_log_hf=use_log_hf,
        title=f"{experiment_title}: validation HF truth vs prediction",
    )

    # Same validation parity plot as normal training.
    validation_parity_plot = (
        mfgp_out / f"mfgp_{tag}_validation_parity_iter{iteration}.png"
    )
    _plot_validation_parity_pointwise(
        y_val,
        val_mean_model,
        val_std_model,
        validation_parity_plot,
        use_log_hf=use_log_hf,
        title=experiment_title,
    )

    # Same mean/uncertainty heatmap as normal training.
    x_lf_train, x_hf_train = _saved_mfgp_training_theta(bundle)
    plot_runtime = _saved_mfgp_plot_runtime(
        bundle=bundle,
        model_path=mfgp_model_path,
        output_dir=mfgp_out,
        x_lf=x_lf_train,
        x_hf=x_hf_train,
    )
    grid_xy = _positive_prediction_grid(
        plot_runtime,
        x_lf_train,
        x_hf_train,
        int(grid_points_per_axis),
    )

    grid_mean_model, grid_std_model = _predict_in_chunks(
        model,
        grid_xy,
        int(predict_chunk_size),
    )
    grid_pred, grid_lower, grid_upper = _prediction_interval_in_output_space(
        grid_mean_model,
        grid_std_model,
        use_log_hf=use_log_hf,
    )
    grid_uncertainty = 0.5 * np.maximum(grid_upper - grid_lower, 0.0)

    grid_csv = mfgp_out / f"mfgp_{tag}_savedmodel_grid_iter{iteration}.csv"
    grid_output = pd.DataFrame(grid_xy, columns=x_cols)
    grid_output["mf_mean_model_space"] = grid_mean_model
    grid_output["mf_std_model_space"] = grid_std_model
    grid_output["mf_prediction"] = grid_pred
    grid_output["mf_lower_1sigma"] = grid_lower
    grid_output["mf_upper_1sigma"] = grid_upper
    grid_output["mf_uncertainty_1sigma"] = grid_uncertainty
    grid_output.to_csv(grid_csv, index=False)

    mean_std_plot = (
        mfgp_out / f"mfgp_{tag}_mean_uncertainty_positive_iter{iteration}.png"
    )
    _plot_mean_std_heatmaps(
        grid_xy,
        grid_pred,
        grid_uncertainty,
        x_cols,
        x_val,
        mean_std_plot,
        title=f"{experiment_title}: MF-GP prediction over R≥0, Z≥0",
    )

    print(f"[MF-GP] loaded checkpoint -> {Path(mfgp_model_path).resolve()}")
    print(f"[MF-GP] metrics -> {metrics_json}")
    print(f"[MF-GP] validation predictions -> {validation_prediction_csv}")
    print(f"[MF-GP] prediction grid -> {grid_csv}")
    print(
        "[MF-GP] three standard plots -> "
        f"{mean_std_plot}, {validation_point_plot}, {validation_parity_plot}"
    )

    return MFGPValidationResult(
        metrics_json=metrics_json,
        validation_prediction_csv=validation_prediction_csv,
        combined_summary_csv=combined_summary_csv,
        grid_csv=grid_csv,
        mean_std_plot=mean_std_plot,
        validation_point_plot=validation_point_plot,
        validation_parity_plot=validation_parity_plot,
    )


# -----------------------------------------------------------------------------
# Step 5: combined report
# -----------------------------------------------------------------------------

def make_combined_report(
    *,
    cnp_result: CNPValidationResult,
    mfgp_result: MFGPValidationResult,
    output_dir: str | Path,
) -> Path:
    output_dir = Path(output_dir).expanduser().resolve()

    cnp_metrics = json.loads(cnp_result.overall_metrics_json.read_text())
    mfgp_metrics = json.loads(mfgp_result.metrics_json.read_text())

    report = {
        "cnp_holdout": cnp_metrics,
        "mfgp_holdout": mfgp_metrics.get("validation"),
        "mfgp_training_reference": mfgp_metrics.get("training"),
        "artifacts": {
            "cnp_shell_distribution_csv": str(cnp_result.prediction_csv),
            "cnp_geometry_metrics_csv": str(cnp_result.geometry_metrics_csv),
            "cnp_shell_parity_plot": str(cnp_result.parity_plot),
            "cnp_shell_error_plot": str(cnp_result.shell_error_plot),
            "mfgp_metrics_json": str(mfgp_result.metrics_json),
            "mfgp_validation_prediction_csv": str(mfgp_result.validation_prediction_csv),
            "mfgp_grid_csv": str(mfgp_result.grid_csv),
            "mfgp_mean_uncertainty_plot": str(mfgp_result.mean_std_plot),
            "mfgp_validation_point_plot": str(mfgp_result.validation_point_plot),
            "mfgp_validation_parity_plot": str(mfgp_result.validation_parity_plot),
        },
    }

    report_path = output_dir / "holdout_validation_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    return report_path


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Strict holdout validation for the trained CNP -> MF-GP detector pipeline."
    )

    p.add_argument(
        "--raw-dir",
        type=Path,
        required=True,
        help="Directory containing validation_manifest.csv and all validation event files.",
    )
    p.add_argument("--cnp-model", type=Path, required=True)

    p.add_argument(
        "--mfgp-model",
        type=Path,
        required=True,
        help="Saved MF-GP .joblib checkpoint produced by mfgp_clean_pipeline.py.",
    )

    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--mc-samples", type=int, default=30)
    p.add_argument("--chunk-size", type=int, default=20000)
    p.add_argument(
        "--grid-points",
        type=int,
        default=120,
        help="MF-GP dense prediction-grid points per R/Z axis.",
    )
    p.add_argument("--iteration", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None)
    p.add_argument(
        "--no-overwrite-h5",
        action="store_true",
        help="Do not clear output_dir/validation_h5 before preparation.",
    )
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 90)
    print("STRICT CNP -> MF-GP HOLDOUT VALIDATION")
    print("=" * 90)
    print(f"Validation data:   {args.raw_dir}")
    print(f"Manifest:          {args.raw_dir / VALIDATION_MANIFEST_NAME}")
    print(f"CNP checkpoint:    {args.cnp_model}")
    print("CNP context:       embedded in checkpoint")
    print(f"MF-GP checkpoint:  {args.mfgp_model}")
    print(f"Output directory:  {output_dir}")
    print("=" * 90)

    # Load the complete CNP inference bundle once. No YAML or training data is needed.
    cnp_model, context_x, context_y, cnp_metadata = load_cnp_inference_checkpoint(
        args.cnp_model,
        device=args.device,
    )
    print(f"Embedded context:  {len(context_x):,} events")

    prepared = prepare_validation_h5(
        raw_dir=args.raw_dir,
        cnp_metadata=cnp_metadata,
        output_dir=output_dir,
        overwrite=not args.no_overwrite_h5,
    )

    if not (prepared.manifest["fidelity"] == 1).any():
        raise ValueError(
            "The validation manifest contains no fidelity=1 rows. "
            "The current MF-GP validation path compares against high-fidelity truth, "
            "so at least one validation geometry must have fidelity=1."
        )

    cnp_result = run_cnp_holdout_validation(
        model=cnp_model,
        context_x=context_x,
        context_y=context_y,
        cnp_metadata=cnp_metadata,
        validation_h5_dir=prepared.h5_dir,
        output_dir=output_dir,
        mc_samples=args.mc_samples,
        chunk_size=args.chunk_size,
        iteration=args.iteration,
        seed=args.seed,
    )

    mfgp_result = run_mfgp_validation(
        mfgp_model_path=args.mfgp_model,
        validation_cnp_csv=cnp_result.prediction_csv,
        output_dir=output_dir,
        predict_chunk_size=args.chunk_size,
        grid_points_per_axis=args.grid_points,
    )

    report_path = make_combined_report(
        cnp_result=cnp_result,
        mfgp_result=mfgp_result,
        output_dir=output_dir,
    )


    print("\n" + "=" * 90)
    print("VALIDATION COMPLETE")
    print("=" * 90)
    print(f"Preparation summary: {output_dir / 'preparation_summary.csv'}")
    print(f"CNP predictions:     {cnp_result.prediction_csv}")
    print(f"CNP metrics:         {cnp_result.overall_metrics_json}")
    print(f"MF-GP metrics:       {mfgp_result.metrics_json}")
    print(f"MF-GP grid:          {mfgp_result.grid_csv}")
    print(f"MF-GP mean/std plot: {mfgp_result.mean_std_plot}")
    print(f"MF-GP points plot:   {mfgp_result.validation_point_plot}")
    print(f"MF-GP parity plot:   {mfgp_result.validation_parity_plot}")
    print(f"Combined report:     {report_path}")
    print("=" * 90)


if __name__ == "__main__":
    main()
