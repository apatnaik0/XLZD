"""
This file is used for emulating a number of fake event sources and format them in the same strucutre to use an already trained model to predict on those emulated points
"""
from __future__ import annotations

from dataclasses import replace
from inspect import signature
from pathlib import Path
from typing import Mapping, Optional, Sequence
import os
import shutil
import sys

import numpy as np
import pandas as pd

def find_repo_root(start: Path | None = None) -> Path:
    """Find the repository root from a notebook or script location."""
    start = (start or Path.cwd()).resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "PROJECT_EXPERIMENT_GUIDE.md").exists() and (candidate / "README.md").exists():
            return candidate
    raise RuntimeError("Could not find the XLZD repo root from the current working directory.")


REPO_ROOT = find_repo_root()
os.chdir(REPO_ROOT)
CNP_MFGP_ROOT = REPO_ROOT / "cnp_mfgp"
for path in (REPO_ROOT, CNP_MFGP_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from cnp_clean_pipeline import CNPRuntimeConfig, PredictResult, predict_cnp
from prepare_cnp_mfgp_data import TARGET_COLUMN, write_h5_class_block

from common.config import ShellConfig
from common.geometry import shell_boundaries

###====================================###

### Geometry

###====================================###

def sample_cylinder_region(
    n: int,
    r_min: float,
    r_max: float,
    z_min: float,
    z_max: float,
    seed: int | None = None,
) -> np.ndarray:
    """Sample points uniformly in cylindrical volume.

    Returns an ``(n, 3)`` array with columns ``x, y, z``.
    """
    if n < 0:
        raise ValueError("n must be non-negative.")
    if r_min < 0 or r_max <= r_min:
        raise ValueError("Require 0 <= r_min < r_max.")
    if z_max <= z_min:
        raise ValueError("Require z_min < z_max.")

    rng = np.random.default_rng(seed)
    n = int(n)
    if n == 0:
        return np.empty((0, 3), dtype=float)

    r = np.sqrt(rng.uniform(float(r_min) ** 2, float(r_max) ** 2, size=n))
    theta = rng.uniform(0.0, 2.0 * np.pi, size=n)
    z = rng.uniform(float(z_min), float(z_max), size=n)

    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.column_stack([x, y, z])

###====================================###

### Emulation

###====================================###

def emulate_detector_points(
    n_points: int,
    radius: float,
    height: float,
    width: float,
    E0: float = 2447.0,
    ETPC: float = 2447.0,
    seed: int | None = None,
    include_caps: bool = False,
    z_min: float = 0.0,
) -> pd.DataFrame:
    """Generate synthetic source points just outside a cylindrical detector.

    Parameters
    ----------
    n_points:
        Number of synthetic source points to generate.
    radius:
        Detector radius.  Side-wall points are sampled with
        ``radius <= s_r <= radius + width``.
    height:
        Detector full z-height in the same coordinates as the raw ``sz`` column.
        The side-wall region spans ``z_min <= sz <= z_min + height``.
    width:
        Thickness of the outside sampling skin.
    include_caps:
        If False, sample only around the cylindrical side wall.  If True, also
        sample top and bottom cap skins and allocate points by approximate volume.
    z_min:
        Lower detector z coordinate.  Use ``0.0`` for data stored from bottom to
        top, or ``-Z_max`` for centered coordinates.

    Returns
    -------
    pd.DataFrame
        Columns: ``sx, sy, sz, E0, ETPC, x, y, z``.  The endpoint columns are
        placeholders so the output resembles the raw event CSV schema.
    """
    if n_points <= 0:
        raise ValueError("n_points must be positive.")
    if radius <= 0 or height <= 0 or width <= 0:
        raise ValueError("radius, height, and width must be positive.")

    rng = np.random.default_rng(seed)
    outer_radius = float(radius) + float(width)
    z0 = float(z_min)
    z1 = z0 + float(height)

    if include_caps:
        side_vol = np.pi * (outer_radius**2 - float(radius) ** 2) * float(height)
        cap_vol = np.pi * outer_radius**2 * float(width)
        total_vol = side_vol + 2.0 * cap_vol

        n_side = int(round(n_points * side_vol / total_vol))
        n_top = int(round(n_points * cap_vol / total_vol))
        n_bottom = int(n_points) - n_side - n_top

        parts = [
            sample_cylinder_region(n_side, radius, outer_radius, z0, z1, seed=rng.integers(0, 2**32 - 1)),
            sample_cylinder_region(n_bottom, 0.0, outer_radius, z0 - width, z0, seed=rng.integers(0, 2**32 - 1)),
            sample_cylinder_region(n_top, 0.0, outer_radius, z1, z1 + width, seed=rng.integers(0, 2**32 - 1)),
        ]
        points = np.vstack([p for p in parts if len(p)])
    else:
        points = sample_cylinder_region(
            int(n_points),
            r_min=float(radius),
            r_max=outer_radius,
            z_min=z0,
            z_max=z1,
            seed=seed,
        )

    rng.shuffle(points)

    return pd.DataFrame(
        {
            "sx": points[:, 0],
            "sy": points[:, 1],
            "sz": points[:, 2],
            "E0": float(E0),
            "ETPC": float(ETPC),
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
        }
    )

def prepare_emulated_csv_for_cnp_predict(
    emulated_csv: str | Path,
    runtime: CNPRuntimeConfig,
    shell_cfg: ShellConfig,
    h5_block_size: int | None = 100_000,
    output_h5_dir: str | Path | None = None,
    theta_values: Mapping[str, float] | None = None,
    z_center: float | None = None,
    overwrite: bool = True,
) -> Path:
    """Convert an emulated source-point CSV into prediction H5 blocks.

    The H5 files match the categorical CNP format:

    - ``theta`` uses ``runtime.theta_headers``
    - ``phi`` uses ``runtime.phi_headers``
    - ``target_shell`` is a dummy valid class label because the CNP loader
      requires labels.  These labels are marked as unlabeled again after
      prediction, so do not use the generated MFGP aggregate as truth.
    """
    emulated_csv = Path(emulated_csv).expanduser().resolve()
    if not emulated_csv.exists():
        raise FileNotFoundError(f"Emulated CSV does not exist: {emulated_csv}")

    if output_h5_dir is None:
        output_h5_dir = emulated_csv.parent / f"{emulated_csv.stem}_cnp_h5"
    output_h5_dir = Path(output_h5_dir).expanduser().resolve()

    if output_h5_dir.exists() and overwrite:
        shutil.rmtree(output_h5_dir)
    output_h5_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(emulated_csv)
    required_source = {"sx", "sy", "sz"}
    missing_source = sorted(required_source - set(df.columns))
    if missing_source:
        raise ValueError(f"Emulated CSV is missing required source columns: {missing_source}")

    center = _resolve_z_center(df, shell_cfg, z_center)
    df = _add_phi_columns(df, runtime.phi_headers, center)

    if h5_block_size is None:
        h5_block_size = len(df)
    h5_block_size = int(h5_block_size)
    if h5_block_size <= 0:
        raise ValueError("h5_block_size must be positive or None.")

    for block_index, start in enumerate(range(0, len(df), h5_block_size)):
        block_df = df.iloc[start : start + h5_block_size].copy()
        n = len(block_df)
        if n == 0:
            continue

        output_path = output_h5_dir / f"{emulated_csv.stem}_block{block_index:04d}_event_classes.h5"

        theta = _build_theta_matrix(
            n_rows=n,
            theta_headers=runtime.theta_headers,
            shell_cfg=shell_cfg,
            theta_values=theta_values,
        )
        phi = block_df[list(runtime.phi_headers)].to_numpy(dtype=np.float32)

        # Dummy valid class label. H5EventPool requires target_shell in
        # [0, n_shells - 1]. We mark prediction CSVs as unlabeled afterward.
        target_shell = np.zeros(n, dtype=np.int64)

        event_index = np.arange(start, start + n, dtype=np.int64)
        source_file = np.asarray([emulated_csv.name] * n, dtype="S")

        meta = {
            "event_index": event_index,
            "original_event_id": event_index,
            "shell_index": np.full(n, -1, dtype=np.int64),
            "source_file": source_file,
            "source_fidelity": np.full(n, -1, dtype=np.int32),
            "split_name": np.asarray(["emulation"] * n, dtype="S"),
            "output_fidelity": np.asarray(["emulation"] * n, dtype="S"),
            "detector_R": np.full(n, float(shell_cfg.R_max), dtype=np.float32),
            "detector_Z": np.full(n, float(shell_cfg.Z_max), dtype=np.float32),
            "detector_z_center": np.full(n, float(center), dtype=np.float32),
        }

        write_h5_class_block(
            output_path=output_path,
            theta=theta,
            phi=phi,
            target_shell=target_shell,
            theta_headers=runtime.theta_headers,
            phi_headers=runtime.phi_headers,
            meta=meta,
        )

    return output_h5_dir

###====================================###

### H5 Helpers

###====================================###

def _resolve_z_center(df: pd.DataFrame, shell_cfg: ShellConfig, z_center: float | None) -> float:
    if z_center is not None:
        return float(z_center)
    if shell_cfg.z_center is not None:
        return float(shell_cfg.z_center)
    return float(0.5 * (df["sz"].min() + df["sz"].max()))

def _add_phi_columns(
    df: pd.DataFrame,
    phi_headers: Sequence[str],
    z_center: float,
) -> pd.DataFrame:
    """Ensure the dataframe has all phi columns required by the CNP runtime."""
    out = df.copy()

    if "s_r" in phi_headers and "s_r" not in out.columns:
        if not {"sx", "sy"}.issubset(out.columns):
            raise ValueError("Cannot derive s_r without sx and sy columns.")
        out["s_r"] = np.sqrt(out["sx"].astype(float) ** 2 + out["sy"].astype(float) ** 2)

    if "s_z_from_center" in phi_headers and "s_z_from_center" not in out.columns:
        if "sz" not in out.columns:
            raise ValueError("Cannot derive s_z_from_center without an sz column.")
        out["s_z_from_center"] = np.abs(out["sz"].astype(float) - float(z_center))

    missing = [name for name in phi_headers if name not in out.columns]
    if missing:
        raise ValueError(
            "Emulated CSV is missing required phi columns and they could not be derived: "
            f"{missing}"
        )

    return out

def _default_theta_value(theta_name: str, shell_cfg: ShellConfig) -> float:
    """Map common theta names onto detector/shell geometry values."""
    key = theta_name.strip().lower()
    boundaries = shell_boundaries(shell_cfg)
    outer = boundaries.iloc[-1]

    r_names = {
        "r",
        "r_max",
        "detector_r",
        "detector_radius",
        "r_shell",
        "r_boundary",
        "radius",
    }
    z_names = {
        "z",
        "z_max",
        "detector_z",
        "detector_height",
        "z_shell",
        "z_boundary",
        "height",
    }

    if key in r_names:
        if shell_cfg.R_max is None:
            raise ValueError(f"Cannot fill theta column {theta_name!r}: shell_cfg.R_max is None.")
        return float(outer["R_boundary"])

    if key in z_names:
        if shell_cfg.Z_max is None:
            raise ValueError(f"Cannot fill theta column {theta_name!r}: shell_cfg.Z_max is None.")
        return float(outer["Z_boundary"])

    raise ValueError(
        f"Do not know how to fill theta column {theta_name!r}. "
        "Pass theta_values={...} to prepare_emulated_csv_for_cnp_predict."
    )

def _build_theta_matrix(
    n_rows: int,
    theta_headers: Sequence[str],
    shell_cfg: ShellConfig,
    theta_values: Mapping[str, float] | None = None,
) -> np.ndarray:
    theta_values = {} if theta_values is None else dict(theta_values)
    columns: list[np.ndarray] = []

    for name in theta_headers:
        value = theta_values.get(name, _default_theta_value(name, shell_cfg))
        columns.append(np.full(n_rows, float(value), dtype=np.float32))

    return np.column_stack(columns).astype(np.float32)
    
###====================================###

### Prediction

###====================================###

def _predict_cnp_accepts_all_shells() -> bool:
    return "all_shells" in signature(predict_cnp).parameters

def _mark_unlabeled_predictions(result: PredictResult, keep_dummy_mfgp: bool = False) -> None:
    """Remove dummy truth labels from emulation prediction CSVs."""
    best_path = getattr(result, "best_path", None)
    if best_path is not None and Path(best_path).exists():
        best = pd.read_csv(best_path)
        if "true_shell_index" in best.columns:
            best["true_shell_index"] = -1
        best.to_csv(best_path, index=False)

    all_path = getattr(result, "all_path", None)
    if all_path is not None and Path(all_path).exists():
        all_df = pd.read_csv(all_path)
        if "true_shell_index" in all_df.columns:
            all_df["true_shell_index"] = -1
        if "y_raw" in all_df.columns:
            all_df["y_raw"] = np.nan
        all_df.to_csv(all_path, index=False)

    # The aggregate MFGP CSV is based on dummy target_shell labels and is not a
    # valid training/validation target for emulated points. Empty it by default
    # so it cannot accidentally be used downstream.
    mfgp_path = getattr(result, "mfgp_path", None)
    if (not keep_dummy_mfgp) and mfgp_path is not None and Path(mfgp_path).exists():
        header = pd.read_csv(mfgp_path, nrows=0)
        header.iloc[:0].to_csv(mfgp_path, index=False)

def predict_cnp_from_emulated_csv(
    emulated_csv: str | Path,
    runtime: CNPRuntimeConfig,
    model_path: str | Path,
    output_dir: str | Path | None = None,
    shell_cfg: ShellConfig | None = None,
    h5_block_size: int | None = 100_000,
    mc_samples: int = 30,
    chunk_size: int = 20_000,
    device: str | None = None,
    cleanup: bool = True,
    all_shells: bool = False,
    theta_values: Mapping[str, float] | None = None,
    z_center: float | None = None,
    keep_dummy_mfgp: bool = False,
) -> PredictResult:
    """Run a trained CNP on an emulated source-point CSV.

    This creates temporary H5 blocks, points a copied runtime at those blocks,
    runs ``predict_cnp``, and then marks output prediction CSVs as unlabeled.

    Use ``all_shells=False`` for fast best-shell prediction.  Set
    ``all_shells=True`` only when you need the full event-shell diagnostic CSV.
    """
    emulated_csv = Path(emulated_csv).expanduser().resolve()
    output_dir = Path(output_dir if output_dir is not None else CNP_MFGP_ROOT / "outputs" / "emulation").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if shell_cfg is None:
        shell_cfg = ShellConfig(
            R_max=_theta_value_from_runtime_or_none(runtime, "R"),
            Z_max=_theta_value_from_runtime_or_none(runtime, "Z"),
            n_shells=int(runtime.n_shells),
            min_candidate_events=1,
            z_center=z_center,
        )
    shell_cfg.validate()

    h5_dir = emulated_csv.parent / f"{emulated_csv.stem}_cnp_h5"
    if cleanup and h5_dir.exists():
        shutil.rmtree(h5_dir)

    h5_dir = prepare_emulated_csv_for_cnp_predict(
        emulated_csv=emulated_csv,
        runtime=runtime,
        shell_cfg=shell_cfg,
        h5_block_size=h5_block_size,
        output_h5_dir=h5_dir,
        theta_values=theta_values,
        z_center=z_center,
        overwrite=True,
    )

    emulated_runtime = replace(
        runtime,
        predict_dirs=[h5_dir],
        predict_fidelities=[-1],
        predict_iterations=[0],
        out_dir=output_dir,
    )

    predict_kwargs = {}
    if _predict_cnp_accepts_all_shells():
        predict_kwargs["all_shells"] = bool(all_shells)
    elif all_shells is False:
        # Older predict_cnp has no all_shells flag and will always write all shells.
        print("[warn] predict_cnp has no all_shells flag; full all-shell CSV will be written.")

    result = predict_cnp(
        runtime=emulated_runtime,
        model_path=model_path,
        mc_samples=int(mc_samples),
        output_suffix=f"{emulated_csv.stem}_event_shell_distribution",
        chunk_size=int(chunk_size),
        device=device,
        **predict_kwargs,
    )

    _mark_unlabeled_predictions(result, keep_dummy_mfgp=keep_dummy_mfgp)
    return result

def _theta_value_from_runtime_or_none(runtime: CNPRuntimeConfig, axis: str) -> float | None:
    """Best-effort fallback if a caller does not pass ShellConfig.

    In normal use, pass ``shell_cfg`` explicitly from your notebook parameters or
    pipeline config.  This fallback exists only to produce a clearer error later
    if geometry was omitted.
    """
    _ = runtime, axis
    return None

__all__ = [
    "find_repo_root",
    "sample_cylinder_region",
    "emulate_detector_points",
    "prepare_emulated_csv_for_cnp_predict",
    "predict_cnp_from_emulated_csv",
]