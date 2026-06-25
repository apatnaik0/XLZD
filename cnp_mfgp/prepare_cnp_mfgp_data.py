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

from common.config import FileLoadConfig, DEFAULT_FILE_STEMS, SplitConfig, SamplingConfig, OutputConfig, ShellConfig, ShellPipelineConfig
from common.dataset import split_into_disjoint_pools, split_pool_into_blocks
from common.io_utils import load_event_collection, save_dataframe
from common.theta import Z_FROM_CENTER_COLUMN, add_centered_z_coordinate 
from common.geometry import build_shell_table, update_detector_maximums, infer_centered_z_coordinate
from common.blocks import build_shell_event_block
from common.h5_utils import write_shell_table_group
from common.pipeline_utils import log_stage, finish_stage


TARGET_COLUMN = "target_shell"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("xlzd_shell/config/pipeline_config.json"),
        help="Path to shell JSON config file.",
    )
    return parser.parse_args()


def load_json_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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


def write_h5_class_block(
    *,
    output_path: Path,
    phi: np.ndarray,
    target_shell: np.ndarray,
    phi_headers: Sequence[str],
    shell_table_df: pd.DataFrame,
    meta: dict[str, np.ndarray],
) -> None:
    """
    Write one event-level categorical h5 block

    Datasets:
        phi:           float32, shape (N, phi_dim)
        target_shell:  int64, shape (N, ) zero-indexed class labels
        shell_table:   metadata table describing shell geometry
        meta:          event_index and one-based shell index
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    target_shell = np.asarray(target_shell, dtype=np.int64).reshape(-1)

    if phi.ndim != 2:
        raise ValueError(f"phi must have shape (N, phi_dim), got {phi.shape}")

    if len(phi) != len(target_shell):
        raise ValueError(
            f"phi/target length mismatch: phi={len(phi)}, target_shell={len(target_shell)}"
        )

    with h5py.File(output_path, 'w') as f:
        f.create_dataset("phi", data=phi.astype(np.float32), compression="gzip", compression_opts=4)
        f.create_dataset("target_shell", data=target_shell.astype(np.int64), compression="gzip", compression_opts=4)
        f.create_dataset("phi_labels", data=np.asarray(phi_headers, dtype="S"))
        f.create_dataset("target_headers", data=np.asarray([TARGET_COLUMN], dtype="S"))

        write_shell_table_group(f, shell_table_df)

        meta_group = f.create_group("meta")
        for key, value in meta.items():
            meta_group.create_dataset(key, data=value, compression="gzip", compression_opts=4)

def write_h5_all_class_blocks(
    *,
    blocks: Sequence[pd.DataFrame],
    shell_table_df: pd.DataFrame,
    output_dir: Path,
    split_name: str,
    fidelity: str,
    phi_headers: Sequence[str],
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
    n_shells = int(shell_table_df["shell_index"].max())

    for block_index, block_df in enumerate(blocks):
        try:
            shell_block = build_shell_event_block(
                block_df=block_df,
                shell_table_df=shell_table_df,
                feature_columns=phi_headers,
                keep_event_data=False,
            )
        except RuntimeError as e:
            print(f"[warn] skipping block {block_index}: {e}")
            continue

        phi = shell_block.features
        target_shell = shell_block.truth_shell
        meta = {
            "event_index": shell_block.event_index,
            "shell_index": shell_block.human_shell
        }
        
        output_path = output_dir / f"{fidelity}_block{block_index:04d}_event_classes.h5"

        write_h5_class_block(
            output_path=output_path,
            phi=phi,
            target_shell=target_shell,
            phi_headers=phi_headers,
            shell_table_df=shell_table_df,
            meta=meta,
        )

        class_counts = np.bincount(target_shell, minlength=n_shells)

        records.append(
            {
                "split_name": split_name,
                "fidelity": fidelity,
                "block_index": block_index,
                "file_name": output_path.name,
                "file_path": str(output_path),
                "original_block_rows": int(len(block_df)),
                "saved_event_rows": int(len(target_shell)),
                "dropped_rows": int(len(block_df) - len(target_shell)),
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
    all_events: pd.DataFrame,
    z_center: float,
    pool_sizes: dict[str, int],
    block_counts: dict[str, int],
    block_size_ranges: dict[str, tuple[int, int]],
    leftover_rows: dict[str, int],
    shell_table_df: pd.DataFrame,
    manifest_df: pd.DataFrame,
    shell_cfg: ShellConfig,
) -> None:
    print("\n=== XLZD Cylindrical Shell Summary ===")
    print(f"Files loaded: {len(files_loaded)}")
    print(f"Total raw events: {len(all_events):,}")
    print(f"z_center used: {z_center:.6g}")
    print(f"Final-position r range: [{all_events['r'].min():.6g}, {all_events['r'].max():.6g}]")
    print(
        f"Final-position z_from_center range: "
        f"[{all_events[Z_FROM_CENTER_COLUMN].min():.6g}, "
        f"{all_events[Z_FROM_CENTER_COLUMN].max():.6g}]"
    )
    print(f"R_max={shell_cfg.R_max:.6g}, Z_max={shell_cfg.Z_max:.6g}, n_shells={shell_cfg.n_shells}")
    print(f"Shell classes: {len(shell_table_df):,}")
    print(f"Pool sizes: LF={pool_sizes['lf']:,}, HF={pool_sizes['hf']:,}, VAL={pool_sizes['validation']:,}")
    print(f"Block counts: n(LF)={block_counts['lf']}, m(HF)={block_counts['hf']}, k(VAL)={block_counts['validation']}")

    print(
        f"Block size ranges: LF={block_size_ranges['lf'][0]}-{block_size_ranges['lf'][1]}, "
        f"HF={block_size_ranges['hf'][0]}-{block_size_ranges['hf'][1]}, "
        f"VAL={block_size_ranges['validation'][0]}-{block_size_ranges['validation'][1]}"
    )

    print(
        f"Unused leftover rows: LF={leftover_rows['lf']}, "
        f"HF={leftover_rows['hf']}, VAL={leftover_rows['validation']}"
    )

    low_support = shell_table_df[shell_table_df["candidate_events"] < shell_cfg.min_candidate_events]
    if not low_support.empty:
        print("\nLow-support shell classes kept:")
        print(
            low_support[
                ["shell_index", "class_index", "candidate_events"]
            ].to_string(index=False)
        )

    print("\nGenerated file statistics:")
    if manifest_df.empty:
        print("No H5 block files were generated")
        return

    total_original = int(manifest_df["original_block_rows"].sum())
    total_saved = int(manifest_df["saved_event_rows"].sum())
    total_dropped = int(manifest_df["dropped_rows"].sum())

    print(f"H5 files written: {len(manifest_df):,}")
    print(f"Original block event rows: {total_original:,}")
    print(f"Saved event rows: {total_saved:,}")
    print(f"Dropped rows outside shell bounds: {total_dropped:,}")

    if total_original > 0:
        print(f"Events assigned shell: {100.0 * total_saved / total_original:.2f}%")

    print("\nPer split/fidelity summary:")
    summary = (
        manifest_df.groupby(["split_name", "fidelity"], dropna=False)[
            [
                "original_block_rows",
                "saved_event_rows",
                "dropped_rows",
                "nonzero_classes",
            ]
        ]
        .sum()
        .reset_index()
    )
    print(summary.to_string(index=False))

    print("\nPer-file event row statistics:")
    print(
        manifest_df[
            [
                "original_block_rows",
                "saved_event_rows",
                "dropped_rows",
                "nonzero_classes",
            ]
        ]
        .describe()
        .to_string()
    )

def main() -> None:
    args = parse_args()
    config = build_config(args)
    total_start = time.perf_counter()

    stage_start = log_stage("Loading and normalizing raw event files")
    loaded = load_event_collection(config.file_load)
    finish_stage(stage_start, f"Loaded {len(loaded.files_loaded)} files")

    stage_start = log_stage("Computing detector coordinates")
    all_events = loaded.concatenated
    z_center = infer_centered_z_coordinate(all_events, config.shell)
    update_detector_maximums(all_events, config.shell, z_center)
    all_events = add_centered_z_coordinate(all_events, z_center)
    finish_stage(stage_start, 
                 f"Using z_center={config.shell.z_center:.6g}, "
                 f"R_max={config.shell.R_max:.6g}, "
                 f"Z_max: {config.shell.Z_max:.6g}")

    stage_start = log_stage("Splitting raw events into disjoint LF/HF/validation pools")
    pools = split_into_disjoint_pools(all_events, config.split)
    finish_stage(stage_start, "Pool split complete")

    validation_block_size = (
        config.sampling.hf_block_size
        if config.sampling.validation_block_size is None
        else config.sampling.validation_block_size
    )

    stage_start = log_stage("Splitting pools into equal-size event blocks")
    lf_blocks = split_pool_into_blocks(pools.lf_pool, block_size=config.sampling.lf_block_size)
    hf_blocks = split_pool_into_blocks(pools.hf_pool, block_size=config.sampling.hf_block_size)
    validation_blocks = split_pool_into_blocks(pools.validation_pool, block_size=validation_block_size)
    finish_stage(stage_start, "Pool block split complete")

    stage_start = log_stage("Building full shell table")
    shell_table_df = build_shell_table(pools.lf_pool, config.shell)
    finish_stage(stage_start, f"Built full shell table with {len(shell_table_df)} shell classes")
    
    lf_blocks_list = list(lf_blocks.blocks)
    hf_blocks_list = list(hf_blocks.blocks)
    validation_blocks_list = list(validation_blocks.blocks)

    output_dir = config.output.output_dir
    output_format = config.output.output_format

    if output_dir.exists():
        stage_start = log_stage(f"Clearing existing dataset directory: {output_dir}")
        shutil.rmtree(output_dir)
        finish_stage(stage_start, "Removed previous shell dataset")

    phi_headers = ["s_r", "s_z_from_center"]
    
    stage_start = log_stage("Writing LF Training event-class H5 blocks")
    lf_manifest = write_h5_all_class_blocks(
        blocks=lf_blocks_list,
        shell_table_df=shell_table_df,
        output_dir=output_dir / "training" / "lf",
        split_name="training",
        fidelity="lf",
        phi_headers=phi_headers,
    )
    finish_stage(stage_start, f"{len(lf_manifest)} LF Training blocks written")

    stage_start = log_stage("Writing HF Training event-class H5 blocks")
    hf_manifest = write_h5_all_class_blocks(
        blocks=hf_blocks_list,
        shell_table_df=shell_table_df,
        output_dir=output_dir / "training" / "hf",
        split_name="training",
        fidelity="hf",
        phi_headers=phi_headers,
    )
    finish_stage(stage_start, f"{len(hf_manifest)} HF Training blocks written")

    stage_start = log_stage("Writing HF Validation event-class H5 blocks")
    val_manifest = write_h5_all_class_blocks(
        blocks=validation_blocks_list,
        shell_table_df=shell_table_df,
        output_dir=output_dir / "validation" / "hf",
        split_name="validation",
        fidelity="hf",
        phi_headers=phi_headers,
    )
    finish_stage(stage_start, f"{len(val_manifest)} HF Validation blocks written")
    
    stage_start = log_stage("Writing pool tables and manifests")
    save_dataframe(all_events, output_dir / "processed_all_events", output_format)
    save_dataframe(pools.lf_pool, output_dir / "lf_training_pool", output_format)
    save_dataframe(pools.hf_pool, output_dir / "hf_training_pool", output_format)
    save_dataframe(pools.validation_pool, output_dir / "hf_validation_pool", output_format)
    shell_table_path = save_dataframe(shell_table_df, output_dir / "shell_table", output_format)
    manifest_df = pd.concat([lf_manifest, hf_manifest, val_manifest], ignore_index=True)
    manifest_path = save_dataframe(manifest_df, output_dir / "event_class_manifest", output_format)
    finish_stage(stage_start, "Output files written")

    pool_sizes = {"lf": len(pools.lf_pool), "hf": len(pools.hf_pool), "validation": len(pools.validation_pool)}
    block_counts = {"lf": len(lf_blocks_list), "hf": len(hf_blocks_list), "validation": len(validation_blocks_list)}
    block_size_ranges = {
        "lf": (min(len(block) for block in lf_blocks_list), max(len(block) for block in lf_blocks_list)),
        "hf": (min(len(block) for block in hf_blocks_list), max(len(block) for block in hf_blocks_list)),
        "validation": (
            min(len(block) for block in validation_blocks_list),
            max(len(block) for block in validation_blocks_list),
        ),
    }
    leftover_rows = {
        "lf": lf_blocks.leftover_rows + sum(len(block) for block in lf_blocks.blocks[len(lf_blocks_list) :]),
        "hf": hf_blocks.leftover_rows + sum(len(block) for block in hf_blocks.blocks[len(hf_blocks_list) :]),
        "validation": validation_blocks.leftover_rows
        + sum(len(block) for block in validation_blocks.blocks[len(validation_blocks_list) :]),
    }

    print_summary(
        files_loaded=loaded.files_loaded,
        all_events=all_events,
        z_center=z_center,
        pool_sizes=pool_sizes,
        block_counts=block_counts,
        block_size_ranges=block_size_ranges,
        leftover_rows=leftover_rows,
        shell_table_df=shell_table_df,
        manifest_df=manifest_df,
        shell_cfg=config.shell,
    )
    print(f"\nArtifacts:\n- shell table: {shell_table_path}\n- manifest: {manifest_path}")
    total_elapsed = time.perf_counter() - total_start
    print(f"\nTotal elapsed: {total_elapsed:.2f}s")


if __name__ == "__main__":
    main()
