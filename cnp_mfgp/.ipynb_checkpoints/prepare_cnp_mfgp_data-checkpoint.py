#!/usr/bin/env python3
"""Prepare cylindrical shell-theta event files for CNP/MF-GP.

Shell construction:
- shells are centered on the TPC center
- outer shell boundaries satisfy:
    R_i = R_max * (i / n_shells)^(1/3)
    Z_i = Z_max * (i / n_shells)^(1/3)
- shell i is the region inside outer boundary i and outside boundary i-1

Target:
- target_shell = zero-indexed shell class label

Dataset split:
- fidelity=0 events are all written to training/lf
- fidelity=1 events are split between training/hf and validation/hf
- split.validation_fraction controls the held-out fraction of HF events only
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import h5py

def find_repo_root(start: Path | None = None) -> Path:
    start = (start or Path.cwd()).resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "PROJECT_EXPERIMENT_GUIDE.md").exists() and (candidate / "README.md").exists():
            return candidate
    raise RuntimeError("Could not find the XLZD repo root from the current working directory.")


REPO_ROOT = find_repo_root()
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from common.config import FileLoadConfig, DEFAULT_FILE_STEMS, SamplingConfig, OutputConfig, ShellConfig, EVENT_ID_COLUMN
from common.dataset import split_pool_into_blocks
from common.io_utils import load_event_file, save_dataframe
from common.theta import add_centered_z_coordinate 
from common.geometry import build_shell_table
from common.blocks import build_shell_event_block
from common.pipeline_utils import log_stage, finish_stage


TARGET_COLUMN = "target_shell"
THETA_HEADERS = ["detector_R", "detector_Z"]
PHI_HEADERS = ["s_r", "s_z_from_center"]
MANIFEST_NAME = "file_manifest.csv"
LOW_FIDELITY = 0
HIGH_FIDELITY = 1
VALID_FIDELITIES = {LOW_FIDELITY, HIGH_FIDELITY}


@dataclass(frozen=True)
class HFValidationSplitConfig:
    """Configuration for holding out a fraction of high-fidelity events."""

    validation_fraction: float = 0.4
    random_seed: int = 42

    def validate(self) -> None:
        if not 0.0 <= self.validation_fraction < 1.0:
            raise ValueError(
                "split.validation_fraction must satisfy 0 <= value < 1. "
                f"Got {self.validation_fraction}."
            )


@dataclass(frozen=True)
class PreparationConfig:
    file_load: FileLoadConfig
    split: HFValidationSplitConfig
    sampling: SamplingConfig
    shell: ShellConfig
    output: OutputConfig

    def validate(self) -> None:
        self.split.validate()
        for section in (self.file_load, self.sampling, self.shell, self.output):
            validate = getattr(section, "validate", None)
            if callable(validate):
                validate()


def _validate_fidelity_series(values: pd.Series, *, context: str) -> pd.Series:
    """Return integer fidelity values and reject anything other than 0 or 1."""
    numeric = pd.to_numeric(values, errors="coerce")
    invalid_numeric = numeric.isna() | ~np.isfinite(numeric.to_numpy(dtype=float))
    non_integer = numeric.notna() & ~np.isclose(
        numeric.to_numpy(dtype=float),
        np.rint(numeric.to_numpy(dtype=float)),
    )
    invalid_binary = numeric.notna() & ~numeric.isin(VALID_FIDELITIES)
    invalid = invalid_numeric | non_integer | invalid_binary

    if invalid.any():
        bad_rows = values.index[invalid].tolist()
        bad_values = values.loc[invalid].tolist()
        raise ValueError(
            f"{context} contains invalid fidelity values at rows {bad_rows}: {bad_values}. "
            "Fidelity must be exactly 0 (low fidelity) or 1 (high fidelity)."
        )

    return numeric.astype(np.int32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("xlzd_shell/config/pipeline_config.json"),
        help="Path to shell JSON config file.")
    parser.add_argument(
        "--manifest",
        type=str,
        default=MANIFEST_NAME,
        help="Manifest filename inside the input data directory.")
    return parser.parse_args()


def load_json_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_file_manifest(input_dir:Path, manifest_name: str = MANIFEST_NAME) -> pd.DataFrame:
    """
    Loads a data manifest to read in the data files, maximum radius and z, and fidelity

    Manifest structure:
        filename: Name of the file, extension included
        R:        Maximum radius of the detector for this data file
        Z:        Maximum Z (half of height) of the detector for this data file
        z_center: Central Z - defines where the central z of the detector is to base z measurements from
        fidelity: 0 for low fidelity or 1 for high fidelity

    Fidelity is required and must be exactly 0 or 1.
    R and Z are allowed to be blank and if so will be inferred from data
    """
    manifest_path = input_dir / manifest_name
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest file: {manifest_path}")
    manifest = pd.read_csv(manifest_path, 
                           na_values=["", "None", "none", "NULL", "null", "NaN", "nan"],
                           keep_default_na=True)

    required = {"filename", "R", "Z", "z_center", "fidelity"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest is missing required columns {sorted(missing)}")
        
    manifest = manifest.copy()
    manifest["filename"] = manifest["filename"].astype(str).str.strip()
    manifest["R"] = pd.to_numeric(manifest["R"], errors="coerce")
    manifest["Z"] = pd.to_numeric(manifest["Z"], errors="coerce")
    manifest["z_center"] = pd.to_numeric(manifest["z_center"], errors="coerce")
    manifest["fidelity"] = _validate_fidelity_series(
        manifest["fidelity"],
        context="file_manifest.csv",
    )
    if manifest["filename"].isna().any() or (manifest["filename"] == "").any():
        raise ValueError("Manifest contains missing filename values.")

    manifest["filename"] = manifest["filename"].astype(str)
    manifest["R"] = manifest["R"].astype(float)
    manifest["Z"] = manifest["Z"].astype(float)
    manifest["z_center"] = manifest["z_center"].astype(float)
    manifest["fidelity"] = manifest["fidelity"].astype(np.int32)

    return manifest


def get_manifest_file_path(input_dir: Path, file_name: str) -> Path:
    path = input_dir / file_name
    if not path.exists():
        raise FileNotFoundError(f"Manifest entry points to missing file: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Manifest entry is not a file: {path}")

    return path


def build_config(args: argparse.Namespace) -> PreparationConfig:
    raw = load_json_config(args.config)
    file_load = raw.get("file_load", {})
    split = raw.get("split", {})
    sampling = raw.get("sampling", {})
    shell = raw.get("shell", {})
    output = raw.get("output", {})

    if "validation_fraction" not in split:
        raise ValueError(
            "Config must define split.validation_fraction. This is the fraction "
            "of fidelity=1 events reserved for validation."
        )

    config = PreparationConfig(
        file_load=FileLoadConfig(
            input_dir=Path(file_load.get("input_dir", "data")),
            file_stems=file_load.get("file_stems", list(DEFAULT_FILE_STEMS)),
            max_rows_per_file=file_load.get("max_rows_per_file"),
        ),
        split=HFValidationSplitConfig(
            validation_fraction=float(split["validation_fraction"]),
            random_seed=int(split.get("random_seed", 42)),
        ),
        sampling=SamplingConfig(
            hf_block_size=int(sampling.get("hf_block_size", 100000)),
            lf_block_size=int(sampling.get("lf_block_size", 10000)),
            validation_block_size=(
                None
                if sampling.get("validation_block_size") is None
                else int(sampling["validation_block_size"])
            ),
            progress=bool(sampling.get("progress", True)),
        ),
        shell=ShellConfig(
            R_max=None if shell.get("R_max") is None else float(shell["R_max"]),
            Z_max=None if shell.get("Z_max") is None else float(shell["Z_max"]),
            n_shells=int(shell.get("n_shells", 100)),
            min_candidate_events=int(shell.get("min_candidate_events", 25)),
            z_center=shell.get("z_center"),
            scale_power=float(shell.get("scale_power", 0.33333333)),
        ),
        output=OutputConfig(
            output_dir=Path(output.get("output_dir", "shell_outputs")),
            output_format=str(output.get("output_format", "csv")),
        ),
    )
    config.validate()
    return config


def build_shell_config_for_manifest_row(
    row: pd.Series,
    base_shell_cfg: ShellConfig,
) -> ShellConfig:
    return ShellConfig(
        R_max=float(row["R"]),
        Z_max=float(row["Z"]),
        n_shells=base_shell_cfg.n_shells,
        min_candidate_events=base_shell_cfg.min_candidate_events,
        z_center=float(row["z_center"]),
        scale_power=base_shell_cfg.scale_power,
    )


def _as_h5_array(value: object) -> np.ndarray:
    arr = np.asarray(value)
    if arr.dtype.kind in {"U", "O"}:
        arr = arr.astype("S")
    return arr


def block_range(blocks: Sequence[pd.DataFrame]) -> tuple[int, int]:
    if not blocks:
        return 0,0
    sizes=[len(block) for block in blocks]
    return int(min(sizes)), int(max(sizes))



def split_hf_training_validation(
    events: pd.DataFrame,
    *,
    validation_fraction: float,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Keep every LF event for training and split only the HF events.

    Fidelity is never assigned by this function. It is read from
    ``file_manifest.csv`` and preserved unchanged:

    - fidelity 0: all events go to LF training
    - fidelity 1: ``validation_fraction`` goes to HF validation and the rest
      goes to HF training
    """
    if "fidelity" not in events.columns:
        raise ValueError("Events are missing the manifest-defined 'fidelity' column.")

    events = events.copy()
    events["fidelity"] = _validate_fidelity_series(
        events["fidelity"],
        context="Prepared events",
    )

    if not 0.0 <= float(validation_fraction) < 1.0:
        raise ValueError(
            "validation_fraction must satisfy 0 <= value < 1. "
            f"Got {validation_fraction}."
        )

    lf_training = events[events["fidelity"] == LOW_FIDELITY].copy()
    hf_events = events[events["fidelity"] == HIGH_FIDELITY].copy().reset_index(drop=True)

    if hf_events.empty:
        raise ValueError(
            "No fidelity=1 events were found. HF data is required for HF training "
            "and validation."
        )

    n_hf = len(hf_events)
    n_validation = int(round(n_hf * float(validation_fraction)))

    if validation_fraction > 0.0:
        if n_hf < 2:
            raise ValueError(
                "At least two fidelity=1 events are required when "
                "split.validation_fraction is greater than zero."
            )
        n_validation = min(max(1, n_validation), n_hf - 1)
    else:
        n_validation = 0

    rng = np.random.default_rng(int(random_seed))
    order = rng.permutation(n_hf)
    validation_idx = order[:n_validation]
    training_idx = order[n_validation:]

    hf_training = hf_events.iloc[training_idx].copy().reset_index(drop=True)
    hf_validation = hf_events.iloc[validation_idx].copy().reset_index(drop=True)
    lf_training = lf_training.reset_index(drop=True)

    print(f"[split] LF training (fidelity=0): {len(lf_training):,}")
    print(f"[split] HF training (fidelity=1): {len(hf_training):,}")
    print(
        f"[split] HF validation (fidelity=1): {len(hf_validation):,} "
        f"({validation_fraction:.2%} requested)"
    )

    return lf_training, hf_training, hf_validation


def assign_shell_labels_for_file(
    *,
    events: pd.DataFrame,
    shell_table_df: pd.DataFrame,
    phi_headers: Sequence[str],
) -> pd.DataFrame:
    work = events.copy()
    if EVENT_ID_COLUMN in work.columns:
        work["original_event_id"] = work[EVENT_ID_COLUMN].to_numpy()
    else:
        work["original_event_id"] = np.arange(len(work), dtype=np.int64)

    # Force a unique temporary event id so build_shell_event_block can be mapped
    # back by position safely.
    work[EVENT_ID_COLUMN] = np.arange(len(work), dtype=np.int64)
    shell_block = build_shell_event_block(
        block_df=work,
        shell_table_df=shell_table_df,
        feature_columns=phi_headers,
        keep_event_data=False,
    )

    if len(shell_block.truth_shell) == 0:
        return work.iloc[:0].copy()
    valid_row_positions = np.asarray(shell_block.event_index, dtype=np.int64)

    labeled = work.iloc[valid_row_positions].copy()
    labeled[TARGET_COLUMN] = np.asarray(shell_block.truth_shell, dtype=np.int64)
    labeled["shell_index"] = np.asarray(shell_block.human_shell, dtype=np.int64)

    return labeled.reset_index(drop=True)

    
def write_h5_class_block(
    *,
    output_path: Path,
    theta: np.ndarray,
    phi: np.ndarray,
    target_shell: np.ndarray,
    theta_headers: Sequence[str],
    phi_headers: Sequence[str],
    meta: dict[str, np.ndarray],
) -> None:
    """
    Write one event-level categorical h5 block

    Datasets:
        theta:         float32, shape (N, theta_dim)
        phi:           float32, shape (N, phi_dim)
        target_shell:  int64, shape (N, ) zero-indexed class labels
        meta:          event_index and one-based shell index
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    theta = np.asarray(theta, dtype=np.float32)
    phi = np.asarray(phi, dtype=np.float32)
    target_shell = np.asarray(target_shell, dtype=np.int64).reshape(-1)

    if theta.ndim != 2:
        raise ValueError(f"theta must have shape (N, theta_dim), got {theta.shape}")
    if phi.ndim != 2:
        raise ValueError(f"phi must have shape (N, phi_dim), got {phi.shape}")
    if target_shell.ndim != 1:
        raise ValueError(f"target_shell must have shape (N,), got {target_shell.shape}")
    if len(theta) != len(phi) or len(phi) != len(target_shell):
        raise ValueError(f"theta/phi/target length mismatch: theta={len(theta)}, phi={len(phi)}, target_shell={len(target_shell)}")
    if theta.shape[1] != len(theta_headers):
        raise ValueError(f"theta dim/header mismatch: theta.shape={theta.shape}, theta_headers={list(theta_headers)}")
    if phi.shape[1] != len(phi_headers):
        raise ValueError(f"phi dim/header mismatch: phi.shape={phi.shape}, phi_headers={list(phi_headers)}")

    with h5py.File(output_path, 'w') as f:
        f.create_dataset("theta", data=theta, compression="gzip", compression_opts=4)
        f.create_dataset("phi", data=phi, compression="gzip", compression_opts=4)
        f.create_dataset("target_shell", data=target_shell, compression="gzip", compression_opts=4)
        f.create_dataset("theta_labels", data=np.asarray(theta_headers, dtype="S"))
        f.create_dataset("phi_labels", data=np.asarray(phi_headers, dtype="S"))
        f.create_dataset("target_headers", data=np.asarray([TARGET_COLUMN], dtype="S"))

        meta_group = f.create_group("meta")
        for key, value in meta.items():
            meta_group.create_dataset(key, data=_as_h5_array(value), compression="gzip", compression_opts=4)

def write_h5_all_class_blocks(
    *,
    blocks: Sequence[pd.DataFrame],
    output_dir: Path,
    split_name: str,
    file_prefix: str,
    theta_headers: Sequence[str],
    phi_headers: Sequence[str],
    n_shells: int,
) -> pd.DataFrame:
    """Save one event-level categorical H5 file per block.

    Fidelity is not inferred from the output folder or block name.  Every H5
    event receives the integer ``fidelity`` that was copied from
    ``file_manifest.csv`` when its source file was loaded.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []

    required_columns = {
        *theta_headers,
        *phi_headers,
        TARGET_COLUMN,
        "shell_index",
        EVENT_ID_COLUMN,
        "fidelity",
    }

    for block_index, block_df in enumerate(blocks):
        if block_df.empty:
            continue

        missing = required_columns - set(block_df.columns)
        if missing:
            raise ValueError(
                f"Block {block_index} is missing required columns: {sorted(missing)}"
            )

        fidelity_values = _validate_fidelity_series(
            block_df["fidelity"],
            context=f"Block {block_index}",
        ).to_numpy(dtype=np.int32)
        unique_fidelities = np.unique(fidelity_values)
        if len(unique_fidelities) != 1:
            raise ValueError(
                f"Block {block_index} mixes fidelities {unique_fidelities.tolist()}. "
                "Blocks must be created within one manifest-defined fidelity."
            )
        block_fidelity = int(unique_fidelities[0])

        theta = block_df[list(theta_headers)].to_numpy(dtype=np.float32)
        phi = block_df[list(phi_headers)].to_numpy(dtype=np.float32)
        target_shell = block_df[TARGET_COLUMN].to_numpy(dtype=np.int64).reshape(-1)
        source_file = (
            block_df["source_file"].astype(str).to_numpy()
            if "source_file" in block_df.columns
            else np.asarray(["unknown"] * len(block_df))
        )

        meta = {
            "event_index": block_df[EVENT_ID_COLUMN].to_numpy(dtype=np.int64),
            "original_event_id": (
                block_df["original_event_id"].to_numpy(dtype=np.int64)
                if "original_event_id" in block_df.columns
                else block_df[EVENT_ID_COLUMN].to_numpy(dtype=np.int64)
            ),
            "shell_index": block_df["shell_index"].to_numpy(dtype=np.int64),
            "source_file": np.asarray(source_file, dtype="S"),
            "fidelity": fidelity_values,
            "split_name": np.asarray([split_name] * len(block_df), dtype="S"),
            "detector_R": block_df["detector_R"].to_numpy(dtype=np.float32),
            "detector_Z": block_df["detector_Z"].to_numpy(dtype=np.float32),
            "detector_z_center": block_df["detector_z_center"].to_numpy(dtype=np.float32),
        }

        output_path = output_dir / f"{file_prefix}_block{block_index:04d}_event_classes.h5"

        write_h5_class_block(
            output_path=output_path,
            theta=theta,
            phi=phi,
            target_shell=target_shell,
            theta_headers=theta_headers,
            phi_headers=phi_headers,
            meta=meta,
        )

        class_counts = np.bincount(target_shell, minlength=n_shells)
        records.append(
            {
                "split_name": split_name,
                "fidelity": block_fidelity,
                "block_index": block_index,
                "file_name": output_path.name,
                "file_path": str(output_path),
                "original_block_rows": int(len(block_df)),
                "saved_event_rows": int(len(target_shell)),
                "dropped_rows": 0,
                "n_shells": int(n_shells),
                "min_class_index": int(target_shell.min()),
                "max_class_index": int(target_shell.max()),
                "nonzero_classes": int(np.count_nonzero(class_counts)),
            }
        )

    return pd.DataFrame.from_records(records)

def print_summary(
    *,
    files_loaded: list[Path],
    total_events_loaded: int,
    lf_training_pool: pd.DataFrame,
    hf_training_pool: pd.DataFrame,
    hf_validation_pool: pd.DataFrame,
    block_summaries: dict[str, dict[str, int | tuple[int, int]]],
    shell_cfg: ShellConfig,
    validation_fraction: float,
) -> None:
    print("\n=== XLZD Cylindrical Shell Summary ===")
    print(f"Total files loaded: {len(files_loaded)}")
    print(f"Total events loaded: {total_events_loaded:,}")
    print(f"Shell classes: {shell_cfg.n_shells:,}")
    print(f"LF training events (fidelity=0): {len(lf_training_pool):,}")
    print(f"HF training events (fidelity=1): {len(hf_training_pool):,}")
    print(f"HF validation events (fidelity=1): {len(hf_validation_pool):,}")
    print(f"Requested HF validation fraction: {validation_fraction:.2%}")

    for label, info in block_summaries.items():
        size_range = info["size_range"]
        print(
            f"{label}: blocks={info['block_count']}, "
            f"block_size_range={size_range[0]}-{size_range[1]}, "
            f"unused_leftover_rows={info['leftover_rows']}"
        )


def main() -> None:
    args = parse_args()
    config = build_config(args)
    validation_fraction = config.split.validation_fraction
    total_start = time.perf_counter()

    input_dir = config.file_load.input_dir
    output_dir = config.output.output_dir
    output_format = config.output.output_format
    manifest_name = getattr(args, "manifest", MANIFEST_NAME)

    if output_dir.exists():
        stage_start = log_stage(f"Clearing existing dataset directory: {output_dir}")
        shutil.rmtree(output_dir)
        finish_stage(stage_start, "Removed previous shell dataset")
    output_dir.mkdir(parents=True, exist_ok=True)

    stage_start = log_stage("Loading file manifest")
    input_manifest = load_file_manifest(input_dir, manifest_name=manifest_name)
    finish_stage(
        stage_start,
        f"Loaded {len(input_manifest)} manifest rows from {input_dir / manifest_name}",
    )

    files_loaded: list[Path] = []
    labeled_event_parts: list[pd.DataFrame] = []
    shell_table_parts: list[pd.DataFrame] = []
    total_events_loaded = 0

    for manifest_index, row in input_manifest.iterrows():
        source_name = str(row["filename"])
        fidelity = int(row["fidelity"])
        data_path = get_manifest_file_path(input_dir, source_name)

        print("\n" + "=" * 80)
        print(f"Processing manifest row {manifest_index + 1}")
        print(f"source_file={source_name}")
        print(f"manifest R={row.get('R')}, Z={row.get('Z')}")
        print(f"fidelity={fidelity}")
        print("=" * 80)

        stage_start = log_stage(f"Loading {data_path.name}")
        events = load_event_file(
            data_path,
            max_rows=config.file_load.max_rows_per_file,
        )
        finish_stage(stage_start, f"Loaded {len(events):,} rows")

        files_loaded.append(data_path)
        total_events_loaded += int(len(events))
        shell_cfg = build_shell_config_for_manifest_row(row, config.shell)

        stage_start = log_stage("Using detector geometry and fidelity from manifest")
        detector_R = float(shell_cfg.R_max)
        detector_Z = float(shell_cfg.Z_max)
        z_center = float(shell_cfg.z_center)
        events = add_centered_z_coordinate(events, z_center)

        events["source_file"] = source_name
        events["fidelity"] = fidelity
        events["detector_R"] = detector_R
        events["detector_Z"] = detector_Z
        events["detector_z_center"] = z_center

        finish_stage(
            stage_start,
            f"Using manifest values: fidelity={fidelity}, "
            f"z_center={z_center:.6g}, R={detector_R:.6g}, Z={detector_Z:.6g}",
        )

        stage_start = log_stage("Building shell table and assigning target shells")
        shell_table_df = build_shell_table(events, shell_cfg)
        shell_table_out = shell_table_df.copy()
        shell_table_out["source_file"] = source_name
        shell_table_out["fidelity"] = fidelity
        shell_table_out["detector_R"] = detector_R
        shell_table_out["detector_Z"] = detector_Z
        shell_table_out["detector_z_center"] = z_center
        shell_table_parts.append(shell_table_out)

        labeled_events = assign_shell_labels_for_file(
            events=events,
            shell_table_df=shell_table_df,
            phi_headers=PHI_HEADERS,
        )
        finish_stage(
            stage_start,
            f"Assigned shells for {len(labeled_events):,}/{len(events):,} events",
        )

        labeled_event_parts.append(labeled_events)
        del events, labeled_events, shell_table_df

    stage_start = log_stage("Concatenating labeled events across all detector geometries")
    if not labeled_event_parts:
        raise RuntimeError("No labeled events were produced from the manifest.")

    all_events = pd.concat(labeled_event_parts, ignore_index=True, sort=False)
    all_events["mixed_event_index"] = np.arange(len(all_events), dtype=np.int64)
    all_events[EVENT_ID_COLUMN] = all_events["mixed_event_index"]
    all_events["fidelity"] = _validate_fidelity_series(
        all_events["fidelity"],
        context="Combined prepared events",
    )
    finish_stage(
        stage_start,
        f"Combined labeled event table has {len(all_events):,} rows",
    )

    stage_start = log_stage("Splitting only high-fidelity events for validation")
    lf_training_pool, hf_training_pool, hf_validation_pool = split_hf_training_validation(
        all_events,
        validation_fraction=validation_fraction,
        random_seed=config.split.random_seed,
    )
    finish_stage(
        stage_start,
        f"HF split complete (validation_fraction={validation_fraction:.4f})",
    )

    validation_block_size = (
        config.sampling.hf_block_size
        if config.sampling.validation_block_size is None
        else config.sampling.validation_block_size
    )

    pool_specs = (
        (
            "LF training",
            lf_training_pool,
            output_dir / "training" / "lf",
            "training",
            "lf",
            config.sampling.lf_block_size,
        ),
        (
            "HF training",
            hf_training_pool,
            output_dir / "training" / "hf",
            "training",
            "hf",
            config.sampling.hf_block_size,
        ),
        (
            "HF validation",
            hf_validation_pool,
            output_dir / "validation" / "hf",
            "validation",
            "hf",
            validation_block_size,
        ),
    )

    manifest_parts: list[pd.DataFrame] = []
    block_summaries: dict[str, dict[str, int | tuple[int, int]]] = {}

    for label, pool_df, pool_output_dir, split_name, file_prefix, block_size in pool_specs:
        stage_start = log_stage(f"Writing {label} event-class H5 blocks")
        block_result = split_pool_into_blocks(pool_df, block_size=block_size)
        blocks = list(block_result.blocks)

        block_manifest = write_h5_all_class_blocks(
            blocks=blocks,
            output_dir=pool_output_dir,
            split_name=split_name,
            file_prefix=file_prefix,
            theta_headers=THETA_HEADERS,
            phi_headers=PHI_HEADERS,
            n_shells=config.shell.n_shells,
        )
        if not block_manifest.empty:
            manifest_parts.append(block_manifest)

        block_summaries[label] = {
            "block_count": int(len(blocks)),
            "size_range": block_range(blocks),
            "leftover_rows": int(block_result.leftover_rows),
        }
        finish_stage(stage_start, f"{len(block_manifest)} {label} blocks written")

    stage_start = log_stage("Writing pool tables and manifests")
    save_dataframe(all_events, output_dir / "processed_all_events", output_format)
    save_dataframe(lf_training_pool, output_dir / "lf_training_pool", output_format)
    save_dataframe(hf_training_pool, output_dir / "hf_training_pool", output_format)
    save_dataframe(hf_validation_pool, output_dir / "hf_validation_pool", output_format)

    if shell_table_parts:
        shell_table_by_theta = pd.concat(shell_table_parts, ignore_index=True, sort=False)
    else:
        shell_table_by_theta = pd.DataFrame()

    shell_table_path = save_dataframe(
        shell_table_by_theta,
        output_dir / "shell_table_by_theta",
        output_format,
    )

    manifest_df = (
        pd.concat(manifest_parts, ignore_index=True, sort=False)
        if manifest_parts
        else pd.DataFrame()
    )
    manifest_path = save_dataframe(
        manifest_df,
        output_dir / "event_class_manifest",
        output_format,
    )
    finish_stage(stage_start, "Output files written")

    print_summary(
        files_loaded=files_loaded,
        total_events_loaded=total_events_loaded,
        lf_training_pool=lf_training_pool,
        hf_training_pool=hf_training_pool,
        hf_validation_pool=hf_validation_pool,
        block_summaries=block_summaries,
        shell_cfg=config.shell,
        validation_fraction=validation_fraction,
    )

    print(
        f"\nArtifacts:"
        f"\n- shell table by theta: {shell_table_path}"
        f"\n- manifest: {manifest_path}"
        f"\n- training H5 root: {output_dir / 'training'}"
        f"\n- validation H5 root: {output_dir / 'validation'}"
    )

    total_elapsed = time.perf_counter() - total_start
    print(f"\nTotal elapsed: {total_elapsed:.2f}s")


if __name__ == "__main__":
    main()
