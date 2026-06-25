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


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from xlzd_resum.config import (  # noqa: E402
    DEFAULT_FILE_STEMS,
    FileLoadConfig,
    OutputConfig,
    SamplingConfig,
    SplitConfig,
)
from xlzd_resum.dataset import (  # noqa: E402
    split_into_disjoint_pools,
    split_pool_into_blocks,
)
from xlzd_resum.io_utils import load_event_collection, save_dataframe  # noqa: E402
from xlzd_resum.theta import Z_FROM_CENTER_COLUMN, add_centered_z_coordinate  # noqa: E402


TARGET_COLUMN = "target_shell"


@dataclass(slots=True)
class ShellConfig:
    R_max: float | None = None
    Z_max: float | None = None
    n_shells: int = 100
    min_candidate_events: int = 25
    z_center: float | None = None
    scale_power: float = 1.0/3.0

    def validate(self) -> None:
        if self.R_max is not None and self.R_max <=0:
            raise ValueError("R_max must be positive")
        if self.Z_max is not None and self.Z_max <=0:
            raise ValueError("Z_max must be positive")
        if self.n_shells <= 0:
            raise ValueError("n_shells must be positive.")
        if self.min_candidate_events <= 0:
            raise ValueError("min_candidate_events must be positive.")


@dataclass(slots=True)
class ShellPipelineConfig:
    file_load: FileLoadConfig
    split: SplitConfig
    sampling: SamplingConfig
    shell: ShellConfig
    output: OutputConfig

    def validate(self) -> None:
        self.file_load.validate()
        self.split.validate()
        self.sampling.validate()
        self.shell.validate()
        self.output.validate()


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


def log_stage(message: str) -> float:
    print(f"\n[{time.strftime('%H:%M:%S')}] {message}", flush=True)
    return time.perf_counter()


def finish_stage(stage_start: float, message: str) -> None:
    elapsed = time.perf_counter() - stage_start
    print(f"[done in {elapsed:.2f}s] {message}", flush=True)

def update_detector_maximums(
    df: pd.DataFrame,
    shell_cfg: ShellConfig,
    z_center: float,
) -> None:
    # Find Z and R Maximum
    if "r" not in df.columns or "z" not in df.columns:
        raise ValueError("Dataframe must contain 'z' and 'r' to infer detector maximums")
    if shell_cfg.R_max is None:
        shell_cfg.R_max = df['r'].max()
    if shell_cfg.Z_max is None:
        shell_cfg.Z_max = np.abs(df['z'] - z_center).max()
    
def infer_centered_z_coordinate(df: pd.DataFrame, shell_cfg: ShellConfig) -> float:
    if "z" not in df.columns:
        raise ValueError("Dataframe must contain 'z' to infer centered coordinates.")
    if shell_cfg.z_center is not None:
        return float(shell_cfg.z_center)
    else:
        z_center = 0.5 * (df["z"].min() + df["z"].max())
        shell_cfg.z_center = z_center
        return z_center
    

def shell_boundaries(shell_cfg: ShellConfig) -> pd.DataFrame:
    idx = np.arange(0, shell_cfg.n_shells + 1, dtype=float)
    frac = idx / float(shell_cfg.n_shells)
    scale = frac ** shell_cfg.scale_power
    r = shell_cfg.R_max * scale
    z = shell_cfg.Z_max * scale
    return pd.DataFrame(
        {
            "shell_level": idx.astype(int),
            "R_boundary": r.astype(float),
            "Z_boundary": z.astype(float),
        }
    )

def inside_shell(
    df: pd.DataFrame,
    *,
    R_inner: float,
    Z_inner: float,
    R_outer: float,
    Z_outer: float,
) -> np.ndarray:
    required = {"r", Z_FROM_CENTER_COLUMN}
    if not required.issubset(df.columns):
        raise ValueError(f"Block dataframe must contain columns {required}.")

    r = df["r"].to_numpy(dtype=float)
    z = df[Z_FROM_CENTER_COLUMN].to_numpy(dtype=float)

    inside_outer = (r <= R_outer) & (z <= Z_outer)
    inside_inner = (r <= R_inner) & (z <= Z_inner)
    return inside_outer & ~inside_inner


def build_shell_table(df_for_support: pd.DataFrame, shell_cfg: ShellConfig) -> pd.DataFrame:
    boundaries = shell_boundaries(shell_cfg)
    support_rows: list[dict[str, float | int]] = []
    shell_volume = 2.0 * np.pi * shell_cfg.Z_max * (shell_cfg.R_max**2) / float(shell_cfg.n_shells)

    for i in range(1, shell_cfg.n_shells + 1):
        prev = boundaries.iloc[i - 1]
        curr = boundaries.iloc[i]
        mask = inside_shell(
            df_for_support,
            R_inner=float(prev["R_boundary"]),
            Z_inner=float(prev["Z_boundary"]),
            R_outer=float(curr["R_boundary"]),
            Z_outer=float(curr["Z_boundary"]),
        )
        support_rows.append(
            {
                "shell_index": int(i),
                "class_index": int(i-1),
                "R_inner": float(prev["R_boundary"]),
                "Z_inner": float(prev["Z_boundary"]),
                "R_shell": float(curr["R_boundary"]),
                "Z_shell": float(curr["Z_boundary"]),
                "candidate_events": int(mask.sum()),
                "shell_volume": float(shell_volume),
            }
        )

    out = pd.DataFrame(support_rows)
    low_support = out[out["candidate_events"] < shell_cfg.min_candidate_events]
    if not low_support.empty:
        print("[warn] Some shell classes have low support, but they are kept for categorical CE because class IDs must remain fixed")

    return out.sort_values(["shell_index"]).reset_index(drop=True)

def positive_shells_for_block(
    block_df: pd.DataFrame,
    shell_table_df: pd.DataFrame,
) -> pd.Series:
    """
    Return one positive shell index per event
    
    The returned shell is one-indexed
    Events outside the detector bounds are labelled NaN
    """
    positive_shell = pd.Series(np.nan, index=block_df.index, dtype="float")
    
    for row in shell_table_df.itertuples(index=False):
        mask = inside_shell(
            block_df,
            R_inner=float(row.R_inner),
            Z_inner=float(row.Z_inner),
            R_outer=float(row.R_shell),
            Z_outer=float(row.Z_shell),
        )
        positive_shell.loc[mask] = int(row.shell_index)

    return positive_shell

def build_event_class_block(
    *,
    block_df: pd.DataFrame,
    shell_table_df: pd.DataFrame,
    phi_headers: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """
    Build one categorical training row per event

    Returns:
        phi:           [N_valid_events, phi_dim]
        target_shell:  [N_valid_events] zero-indexed class labels
        meta:          event_index and readable one-indexed shell index
    """
    positive_shell_one_based = positive_shells_for_block(
        block_df=block_df,
        shell_table_df=shell_table_df
    )

    valid_mask=positive_shell_one_based.notna()
    if not valid_mask.any():
        raise RuntimeError("This block has no events with a valid shell")
    valid_events = block_df.loc[valid_mask].copy()

    shell_one_based = (
        positive_shell_one_based.loc[valid_mask].astype(np.int64).to_numpy()
    )

    # Loss expects classes 0...n_classes-1
    target_shell_zero_based = shell_one_based - 1
    phi = valid_events[list(phi_headers)].to_numpy(dtype=np.float32)

    meta = {
        "event_index": valid_events.index.to_numpy(dtype=np.int64),
        "shell_index": shell_one_based.astype(np.int32),
    }

    return phi, target_shell_zero_based.astype(np.int64), meta

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

        shell_group = f.create_group("shell_table")
        for col in ["shell_index", "class_index", "R_inner", "Z_inner", "R_shell", "Z_shell", "candidate_events", "shell_volume"]:
            shell_group.create_dataset(col, data=shell_table_df[col].to_numpy(), compression="gzip", compression_opts=4)

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
            phi, target_shell, meta = build_event_class_block(
                block_df=block_df,
                shell_table_df=shell_table_df,
                phi_headers=phi_headers,
            )
        except RuntimeError as e:
            print(f"[warn] skipping block {block_index}: {e}")
            continue

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
