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
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
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

from common.config import FileLoadConfig, DEFAULT_FILE_STEMS, SplitConfig, SamplingConfig, OutputConfig, ShellConfig, ShellPipelineConfig, EVENT_ID_COLUMN
from common.dataset import split_into_disjoint_pools, split_pool_into_blocks
from common.io_utils import load_event_file, save_dataframe
from common.theta import add_centered_z_coordinate 
from common.geometry import build_shell_table
from common.blocks import build_shell_event_block
from common.pipeline_utils import log_stage, finish_stage


TARGET_COLUMN = "target_shell"
THETA_HEADERS = ["detector_R", "detector_Z"]
PHI_HEADERS = ["s_r", "s_z_from_center"]
MANIFEST_NAME = "file_manifest.csv"

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
        fidelity: Fidelity of this data file, normally based on number of events

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
    manifest["fidelity"] = pd.to_numeric(manifest["fidelity"], errors="coerce")
    if manifest["fidelity"].isna().any():
        bad_rows = manifest.index[manifest["fidelity"].isna()].tolist()
        raise ValueError(f"Manifest contains missing/non-numeric fidelity values at rows: {bad_rows}")
    manifest["fidelity"] = manifest["fidelity"].astype(int)
    if manifest["filename"].isna().any() or (manifest["filename"] == "").any():
        raise ValueError("Manifest contains missing filename values.")

    manifest["filename"] = manifest["filename"].astype(str)
    manifest["R"] = manifest["R"].astype(float)
    manifest["Z"] = manifest["Z"].astype(float)
    manifest["z_center"] = manifest["z_center"].astype(float)
    manifest["fidelity"] = manifest["fidelity"].astype(int)

    return manifest


def get_manifest_file_path(input_dir: Path, file_name: str) -> Path:
    path = input_dir / file_name
    if not path.exists():
        raise FileNotFoundError(f"Manifest entry points to missing file: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Manifest entry is not a file: {path}")

    return path


def build_config(args: argparse.Namespace) -> ShellPipelineConfig:
    raw = load_json_config(args.config)
    file_load = raw.get("file_load", {})
    split = raw.get("split", {})
    sampling = raw.get("sampling", {})
    shell = raw.get("shell", {})
    output = raw.get("output", {})

    config = ShellPipelineConfig(
        file_load=FileLoadConfig(
            input_dir=Path(file_load.get("input_dir", "data")),
            file_stems=file_load.get("file_stems", list(DEFAULT_FILE_STEMS)),
            max_rows_per_file=file_load.get("max_rows_per_file"),
        ),
        split=SplitConfig(
            lf_pool_fraction=float(split.get("lf_pool_fraction", 0.2)),
            hf_pool_fraction=float(split.get("hf_pool_fraction", 0.4)),
            random_seed=int(split.get("random_seed", 42)),
            stratify_by_component=bool(split.get("stratify_by_component", False)),
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
) -> tuple[ShellConfig, str, str]:
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
    output_fidelity: str,
    theta_headers: Sequence[str],
    phi_headers: Sequence[str],
    n_shells: int,
) -> pd.DataFrame:
    """
    Save one event-level categorical H5 file per block.

    Each file contains:
        phi:          [N_valid_events, phi_dim]
        target_shell: [N_valid_events] int64, zero-based classes 0..n_shells-1

    This is the correct format for weighted categorical cross entropy.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []

    required_columns = {
        *theta_headers,
        *phi_headers,
        TARGET_COLUMN,
        "shell_index",
        EVENT_ID_COLUMN,
    }
    
    for block_index, block_df in enumerate(blocks):
        if block_df.empty:
            continue
        missing = required_columns - set(block_df.columns)
        if missing:
            raise ValueError(
                f"Block {block_index} is missing required columns: {sorted(missing)}"
            )

        theta = block_df[list(theta_headers)].to_numpy(dtype=np.float32)
        phi = block_df[list(phi_headers)].to_numpy(dtype=np.float32)
        target_shell = block_df[TARGET_COLUMN].to_numpy(dtype=np.int64).reshape(-1)
        source_file = (
            block_df["source_file"].astype(str).to_numpy()
            if "source_file" in block_df.columns
            else np.asarray(["unknown"] * len(block_df))
        )
        source_fidelity = (
            block_df["source_fidelity"].to_numpy(dtype=np.int32)
            if "source_fidelity" in block_df.columns
            else np.full(len(block_df), -1, dtype=np.int32)
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
            "source_fidelity": source_fidelity,
            "split_name": np.asarray([split_name] * len(block_df), dtype="S"),
            "output_fidelity": np.asarray([output_fidelity] * len(block_df), dtype="S"),
            "detector_R": block_df["detector_R"].to_numpy(dtype=np.float32),
            "detector_Z": block_df["detector_Z"].to_numpy(dtype=np.float32),
            "detector_z_center": block_df["detector_z_center"].to_numpy(dtype=np.float32),
        }

        output_path = output_dir / f"{output_fidelity}_block{block_index:04d}_event_classes.h5"

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
                "fidelity": output_fidelity,
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
    pool_sizes: dict[str, int],
    block_counts: dict[str, int],
    block_size_ranges: dict[str, tuple[int, int]],
    leftover_rows: dict[str, int],
    shell_cfg: ShellConfig,
) -> None:
    print("\n=== XLZD Cylindrical Shell Summary ===")
    print(f"Total files loaded: {len(files_loaded)}")
    print(f"Total events loaded: {total_events_loaded:,}")
    print(f"Shell classes: {shell_cfg.n_shells:,}")

    print(
        f"Pool sizes: LF={pool_sizes['lf']:,}, HF={pool_sizes['hf']:,}, VAL={pool_sizes['validation']:,}")
    print(
        f"Block counts: n(LF)={block_counts['lf']}, m(HF)={block_counts['hf']}, k(VAL)={block_counts['validation']}")
    print(
        f"Block size ranges: "
        f"LF={block_size_ranges['lf'][0]}-{block_size_ranges['lf'][1]}, "
        f"HF={block_size_ranges['hf'][0]}-{block_size_ranges['hf'][1]}, "
        f"VAL={block_size_ranges['validation'][0]}-{block_size_ranges['validation'][1]}")
    print(f"Unused leftover rows: LF={leftover_rows['lf']}, HF={leftover_rows['hf']}, VAL={leftover_rows['validation']}")


def main() -> None:
    args = parse_args()
    config = build_config(args)
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
        source_fidelity = int(row["fidelity"])
        data_path = get_manifest_file_path(input_dir, source_name)

        print("\n" + "=" * 80)
        print(f"Processing manifest row {manifest_index+1}")
        print(f"source_file={source_name}")
        print(f"manifest R={row.get('R')}, Z={row.get('Z')}")
        print(f"source_fidelity={source_fidelity}")
        print("=" * 80)

        stage_start = log_stage(f"Loading {data_path.name}")
        events = load_event_file(
            data_path,
            max_rows=config.file_load.max_rows_per_file)
        finish_stage(stage_start, f"Loaded {len(events):,} rows")

        files_loaded.append(data_path)
        total_events_loaded += int(len(events))
        shell_cfg = build_shell_config_for_manifest_row(row, config.shell)

        stage_start = log_stage("Using detector geometry from manifest")
        detector_R = float(shell_cfg.R_max)
        detector_Z = float(shell_cfg.Z_max)
        z_center = float(shell_cfg.z_center)
        events = add_centered_z_coordinate(events, z_center)

        events["source_file"] = source_name
        events["source_fidelity"] = source_fidelity
        events["detector_R"] = detector_R
        events["detector_Z"] = detector_Z
        events["detector_z_center"] = z_center

        finish_stage(
            stage_start,
            f"Using manifest geometry: "
            f"z_center={z_center:.6g}, R={detector_R:.6g}, Z={detector_Z:.6g}",
        )

        stage_start = log_stage("Building shell table and assigning target shells")
        shell_table_df = build_shell_table(events, shell_cfg)
        shell_table_out = shell_table_df.copy()
        shell_table_out["source_file"] = source_name
        shell_table_out["source_fidelity"] = source_fidelity
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
    
    stage_start = log_stage("Concatenating labeled events across all theta values")
    if not labeled_event_parts:
        raise RuntimeError("No labeled events were produced from the manifest.")
    all_events = pd.concat(labeled_event_parts, ignore_index=True, sort=False)
    all_events["mixed_event_index"] = np.arange(len(all_events), dtype=np.int64)
    all_events[EVENT_ID_COLUMN] = all_events["mixed_event_index"]
    finish_stage(
        stage_start,
        f"Combined labeled event table has {len(all_events):,} rows",
    )

    validation_block_size = (
        config.sampling.hf_block_size
        if config.sampling.validation_block_size is None
        else config.sampling.validation_block_size
    )

    stage_start = log_stage("Splitting combined labeled events into LF/HF/validation pools")
    pools = split_into_disjoint_pools(all_events, config.split)
    finish_stage(stage_start, "Pool split complete")

    stage_start = log_stage("Splitting combined pools into equal-size event blocks")
    lf_blocks = split_pool_into_blocks(
        pools.lf_pool,
        block_size=config.sampling.lf_block_size)
    hf_blocks = split_pool_into_blocks(
        pools.hf_pool,
        block_size=config.sampling.hf_block_size)
    validation_blocks = split_pool_into_blocks(
        pools.validation_pool,
        block_size=validation_block_size)
    lf_blocks_list = list(lf_blocks.blocks)
    hf_blocks_list = list(hf_blocks.blocks)
    validation_blocks_list = list(validation_blocks.blocks)
    finish_stage(stage_start, "Pool block split complete")

    pool_sizes = {
        "lf": int(len(pools.lf_pool)),
        "hf": int(len(pools.hf_pool)),
        "validation": int(len(pools.validation_pool))}
    block_counts = {
        "lf": int(len(lf_blocks_list)),
        "hf": int(len(hf_blocks_list)),
        "validation": int(len(validation_blocks_list))}
    leftover_rows = {
        "lf": int(lf_blocks.leftover_rows),
        "hf": int(hf_blocks.leftover_rows),
        "validation": int(validation_blocks.leftover_rows)}
    block_size_ranges = {
        "lf": block_range(lf_blocks_list),
        "hf": block_range(hf_blocks_list),
        "validation": block_range(validation_blocks_list)}
    
    stage_start = log_stage("Writing LF Training event-class H5 blocks")
    lf_manifest = write_h5_all_class_blocks(
        blocks=lf_blocks_list,
        output_dir=output_dir / "training" / "lf",
        split_name="training",
        output_fidelity="lf",
        theta_headers=THETA_HEADERS,
        phi_headers=PHI_HEADERS,
        n_shells=config.shell.n_shells)
    finish_stage(stage_start, f"{len(lf_manifest)} LF Training blocks written")

    stage_start = log_stage("Writing HF Training event-class H5 blocks")
    hf_manifest = write_h5_all_class_blocks(
        blocks=hf_blocks_list,
        output_dir=output_dir / "training" / "hf",
        split_name="training",
        output_fidelity="hf",
        theta_headers=THETA_HEADERS,
        phi_headers=PHI_HEADERS,
        n_shells=config.shell.n_shells)
    finish_stage(stage_start, f"{len(hf_manifest)} HF Training blocks written")

    stage_start = log_stage("Writing HF Validation event-class H5 blocks")
    val_manifest = write_h5_all_class_blocks(
        blocks=validation_blocks_list,
        output_dir=output_dir / "validation" / "hf",
        split_name="validation",
        output_fidelity="hf",
        theta_headers=THETA_HEADERS,
        phi_headers=PHI_HEADERS,
        n_shells=config.shell.n_shells)
    finish_stage(stage_start, f"{len(val_manifest)} HF Validation blocks written")

    stage_start = log_stage("Writing pool tables and manifests")

    save_dataframe(all_events, output_dir / "processed_all_events", output_format)
    save_dataframe(pools.lf_pool, output_dir / "lf_training_pool", output_format)
    save_dataframe(pools.hf_pool, output_dir / "hf_training_pool", output_format)
    save_dataframe(pools.validation_pool, output_dir / "hf_validation_pool", output_format)

    if shell_table_parts:
        shell_table_by_theta = pd.concat(shell_table_parts, ignore_index=True, sort=False)
    else:
        shell_table_by_theta = pd.DataFrame()

    shell_table_path = save_dataframe(
        shell_table_by_theta,
        output_dir / "shell_table_by_theta",
        output_format,
    )

    manifest_df = pd.concat(
        [lf_manifest, hf_manifest, val_manifest],
        ignore_index=True,
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
        pool_sizes=pool_sizes,
        block_counts=block_counts,
        block_size_ranges=block_size_ranges,
        leftover_rows=leftover_rows,
        shell_cfg=config.shell,
    )

    print(
        f"\nArtifacts:"
        f"\n- shell table by theta: {shell_table_path}"
        f"\n- manifest: {manifest_path}"
    )

    total_elapsed = time.perf_counter() - total_start
    print(f"\nTotal elapsed: {total_elapsed:.2f}s")
      

if __name__ == "__main__":
    main()
