#!/usr/bin/env python3
"""Prepare equal-volume cylindrical shell-theta event files for CNP/MF-GP.

Shell construction:
- shells are centered on the TPC center
- outer shell boundaries satisfy:
    R_i = R_max * (i / n_shells)^(1/3)
    Z_i = Z_max * (i / n_shells)^(1/3)
- shell i is the region inside outer boundary i and outside boundary i-1

Target:
- inside_shell = 1 if an event lies within that equal-volume shell band
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


THETA_HEADERS = ["R_shell", "Z_shell"]
TARGET_COLUMN = "inside_shell"


@dataclass(slots=True)
class EqualVolumeShellConfig:
    R_max: float = 1500.0
    Z_max: float = 2000.0
    n_shells: int = 160
    min_candidate_events: int = 25
    z_center: float | None = 1982.48
    rounding_decimals: int = 6
    negative_shells: int = 2
    scale_power: float = 1.0/3.0

    def validate(self) -> None:
        if self.R_max <= 0 or self.Z_max <= 0:
            raise ValueError("R_max and Z_max must be positive.")
        if self.n_shells <= 0:
            raise ValueError("n_shells must be positive.")
        if self.min_candidate_events <= 0:
            raise ValueError("min_candidate_events must be positive.")
        if self.rounding_decimals < 0:
            raise ValueError("rounding_decimals must be non-negative.")
        if self.negative_shells < 0:
            raise ValueError("The number of negative shells must be non-negative")


@dataclass(slots=True)
class EqualVolumeShellPipelineConfig:
    file_load: FileLoadConfig
    split: SplitConfig
    sampling: SamplingConfig
    shell: EqualVolumeShellConfig
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
        default=Path("xlzd_equal_volume_shell_theta/config/pipeline_config.json"),
        help="Path to equal-volume shell JSON config file.",
    )
    return parser.parse_args()


def load_json_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_config(args: argparse.Namespace) -> EqualVolumeShellPipelineConfig:
    raw = load_json_config(args.config)
    file_load = raw.get("file_load", {})
    split = raw.get("split", {})
    sampling = raw.get("sampling", {})
    shell = raw.get("shell", {})
    output = raw.get("output", {})

    config = EqualVolumeShellPipelineConfig(
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
        shell=EqualVolumeShellConfig(
            R_max=float(shell.get("R_max", 1500.0)),
            Z_max=float(shell.get("Z_max", 2000.0)),
            n_shells=int(shell.get("n_shells", 160)),
            min_candidate_events=int(shell.get("min_candidate_events", 25)),
            z_center=shell.get("z_center"),
            rounding_decimals=int(shell.get("rounding_decimals", 6)),
            negative_shells=int(shell.get("negative_shells", 4)),
            scale_power=float(shell.get("scale_power", 0.33333333)),
        ),
        output=OutputConfig(
            output_dir=Path(output.get("output_dir", "outputs_equal_volume_shell_theta")),
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


def infer_z_center(df: pd.DataFrame, shell_cfg: EqualVolumeShellConfig) -> float:
    if "z" not in df.columns:
        raise ValueError("Dataframe must contain 'z' to infer centered coordinates.")
    if shell_cfg.z_center is not None:
        return float(shell_cfg.z_center)
    return float(0.5 * (df["z"].min() + df["z"].max()))

def shell_boundaries(shell_cfg: EqualVolumeShellConfig) -> pd.DataFrame:
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


def build_shell_candidate_table(df_for_support: pd.DataFrame, shell_cfg: EqualVolumeShellConfig) -> pd.DataFrame:
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
        count = int(mask.sum())
        if count < shell_cfg.min_candidate_events:
            continue
        support_rows.append(
            {
                "shell_index": int(i),
                "R_inner": float(prev["R_boundary"]),
                "Z_inner": float(prev["Z_boundary"]),
                "R_shell": float(curr["R_boundary"]),
                "Z_shell": float(curr["Z_boundary"]),
                "candidate_events": count,
                "shell_volume": float(shell_volume),
            }
        )

    out = pd.DataFrame(support_rows)
    if out.empty:
        raise RuntimeError(
            "No valid equal-volume shell candidates were found. Lower min_candidate_events or reduce n_shells."
        )
    return out.sort_values(["shell_index"]).reset_index(drop=True)

def positive_shells_for_block(
    block_df: pd.DataFrame,
    candidates_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return one positive shell index per event
    Returns NaN if no positive shells for that event
    Assumes no event is in 2 shells - picks the highest number if so
    """
    positive_shell = pd.Series(np.nan, index=block_df.index, dtype="float")
    
    for row in candidates_df.itertuples(index=False):
        mask = inside_shell(
            block_df,
            R_inner=float(row.R_inner),
            Z_inner=float(row.Z_inner),
            R_outer=float(row.R_shell),
            Z_outer=float(row.Z_shell),
        )
        positive_shell.loc[mask] = int(row.shell_index)

    return positive_shell

def build_event_pairs_per_block(
    *,
    block_df: pd.DataFrame,
    candidates_df: pd.DataFrame,
    phi_headers: Sequence[str],
    num_neg_shells: int,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """
    For each event in a block:
        Creates 1 positive (event, shell) pair
        Creates k negative (event, shell) pair
    Returns:
        Theta:  [N * (1+k), theta_dim]
        Phi:    [N * (1+k), phi_dim]
        target: [N * (1+k), 1]
        meta:   useful debugging arrays
    """
    rng = np.random.default_rng(random_seed)

    # First grab the positive shells for that block and mask out the NaN values
    # This means that any event that doesnt land in a shell (for example if min_candidates > 1), it doesnt get trained on
    positive_shell = positive_shells_for_block(block_df, candidates_df)
    valid_mask = positive_shell.notna()
    if not valid_mask.any():
        raise RuntimeError("This block has no events with a valid positive shell.")

    # Find what events, positives, and phis match with that mask
    valid_events = block_df.loc[valid_mask].copy()
    positive_shell = positive_shell.loc[valid_mask].astype(int).to_numpy()
    phi_values = valid_events[list(phi_headers)].to_numpy(dtype=np.float32)
    
    # Build a lookup table for available shells that matches shell index
    shell_table = candidates_df.set_index("shell_index", drop=False)
    all_shell_indices = shell_table.index.to_numpy(dtype=int)

    # Initialize
    theta_rows: list[list[float]] = []
    phi_rows: list[np.ndarray] = []
    target_rows: list[list[float]] = []

    event_indices: list[int] = []
    shell_indices: list[int] = []
    pair_types: list[str] = []

    for i, (event_index, pos_shell_index) in enumerate(zip(valid_events.index.to_numpy(), positive_shell)):
        event_phi = phi_values[i]
        # Create positive pair - target=1
        pos_shell = shell_table.loc[pos_shell_index]
        theta_rows.append([
            float(pos_shell["R_shell"]),
            float(pos_shell["Z_shell"]),
        ])
        phi_rows.append(event_phi)
        target_rows.append([1.0])

        event_indices.append(int(event_index))
        shell_indices.append(int(pos_shell_index))
        pair_types.append("positive")

        # Create negative pairs
        negative_choices = all_shell_indices[all_shell_indices != pos_shell_index]
        if num_neg_shells > len(negative_choices):
            sampled_neg_shells = negative_choices
        else:
            sampled_neg_shells = rng.choice(
                negative_choices,
                size=num_neg_shells,
                replace=False,
            )

        for neg_shell_index in sampled_neg_shells:
            neg_shell = shell_table.loc[int(neg_shell_index)]

            theta_rows.append([
                float(neg_shell["R_shell"]),
                float(neg_shell["Z_shell"]),
            ])
            phi_rows.append(event_phi)
            target_rows.append([0.0])

            event_indices.append(int(event_index))
            shell_indices.append(int(neg_shell_index))
            pair_types.append("negative")
            
    theta = np.asarray(theta_rows, dtype=np.float32)
    phi = np.asarray(phi_rows, dtype=np.float32)
    target = np.asarray(target_rows, dtype=np.float32)

    meta = {
        "event_index": np.asarray(event_indices, dtype=np.int64),
        "shell_index": np.asarray(shell_indices, dtype=np.int32),
        "pair_type": np.asarray(pair_types, dtype="S16"),
    }

    return theta, phi, target, meta

def write_h5_single_block(
    *,
    output_path: Path,
    theta: np.ndarray,
    phi: np.ndarray,
    target: np.ndarray,
    theta_headers: Sequence[str],
    phi_headers: Sequence[str],
    target_headers: Sequence[str],
    meta: dict[str, np.ndarray],
) -> None:
    # Writes an h5 entry for a single event
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as f:
        f.create_dataset("theta", data=theta, compression="gzip", compression_opts=4)
        f.create_dataset("phi", data=phi, compression="gzip", compression_opts=4)
        f.create_dataset("target", data=target, compression="gzip", compression_opts=4)
        f.create_dataset("theta_headers", data=np.asarray(theta_headers, dtype="S"))
        f.create_dataset("phi_labels", data=np.asarray(phi_headers, dtype="S"))
        f.create_dataset("target_headers", data=np.asarray(target_headers, dtype="S"))

        meta_group = f.create_group("meta")
        for key, value in meta.items():
            meta_group.create_dataset(key, data=value, compression="gzip", compression_opts=4)

def write_h5_all_blocks(
    *,
    blocks: Sequence[pd.DataFrame],
    candidates_df: pd.DataFrame,
    output_dir: Path,
    split_name: str,
    fidelity: str,
    phi_headers: Sequence[str],
    num_neg_shells: int,
    random_seed: int,
) -> pd.DataFrame:
    """
    Save one h5 file per block

    Each file contains all event-shell pairs for that block:
        rows = events_with_positive_shell * (1 + k_negative_shells)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    records=[]
    for block_index, block_df in enumerate(blocks):
        # Gets all event-pairs for a single block (positive and negative)
        try:
            theta, phi, target, meta = build_event_pairs_per_block(
                block_df=block_df,
                candidates_df=candidates_df,
                phi_headers=phi_headers,
                num_neg_shells=num_neg_shells,
                random_seed=random_seed + block_index,
            )
        except RuntimeError as e:
            print(f"[warn] skipping block {block_index}: {e}")
            continue
        
        output_path = output_dir / f"{fidelity}_block{block_index:04d}_event_shell_pairs.h5"
        write_h5_single_block(
            output_path=output_path,
            theta=theta,
            phi=phi,
            target=target,
            theta_headers=["R_shell", "Z_shell"],
            phi_headers=phi_headers,
            target_headers=[TARGET_COLUMN],
            meta=meta,
        )

        records.append(
            {
                "split_name": split_name,
                "fidelity": fidelity,
                "block_index": block_index,
                "file_name": output_path.name,
                "file_path": str(output_path),
                "original_block_rows": int(len(block_df)),
                "saved_pair_rows": int(len(target)),
                "positive_rows": int(target.sum()),
                "negative_rows": int(len(target) - target.sum()),
                "negative_shells_per_event": int(num_neg_shells),
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
    candidates_df: pd.DataFrame,
    manifest_df: pd.DataFrame,
    shell_cfg: EqualVolumeShellConfig,
) -> None:
    print("\n=== XLZD Equal-Volume Cylindrical Shell Summary ===")
    print(f"Files loaded: {len(files_loaded)}")
    print(f"Total raw events: {len(all_events):,}")
    print(f"z_center used: {z_center:.6g}")
    print(f"Final-position r range: [{all_events['r'].min():.6g}, {all_events['r'].max():.6g}]")
    print(f"Final-position z_from_center range: [{all_events[Z_FROM_CENTER_COLUMN].min():.6g}, {all_events[Z_FROM_CENTER_COLUMN].max():.6g}]")
    print(f"R_max={shell_cfg.R_max:.6g}, Z_max={shell_cfg.Z_max:.6g}, n_shells={shell_cfg.n_shells}")
    print(f"Valid shells: {len(candidates_df):,}")
    print(f"Pool sizes: LF={pool_sizes['lf']:,}, HF={pool_sizes['hf']:,}, VAL={pool_sizes['validation']:,}")
    print(f"Block counts: n(LF)={block_counts['lf']}, m(HF)={block_counts['hf']}, k(VAL)={block_counts['validation']}")
    print(
        f"Block size ranges: LF={block_size_ranges['lf'][0]}-{block_size_ranges['lf'][1]}, "
        f"HF={block_size_ranges['hf'][0]}-{block_size_ranges['hf'][1]}, "
        f"VAL={block_size_ranges['validation'][0]}-{block_size_ranges['validation'][1]}"
    )
    print(
        f"Unused leftover rows: LF={leftover_rows['lf']}, HF={leftover_rows['hf']}, VAL={leftover_rows['validation']}"
    )
    print("\nGenerated file statistics:")
    if manifest_df.empty:
        print("No H5 block files were generated")
        return

    total_original = int(manifest_df["original_block_rows"].sum())
    total_pairs = int(manifest_df["saved_pair_rows"].sum())
    total_positive = int(manifest_df["positive_rows"].sum())
    total_negative = int(manifest_df["negative_rows"].sum())

    print(f"H5 files written: {len(manifest_df):,}")
    print(f"Original block event rows: {total_original:,}")
    print(f"Saved event-shell pair rows: {total_pairs:,}")
    print(f"Positive rows: {total_positive:,}")
    print(f"Negative rows: {total_negative:,}")

    if total_positive > 0:
        print(f"Negative/positive ratio: {total_negative / total_positive:.3f}:1")
        print(f"Events assigned positive shell: {100.0 * total_positive / total_original:.2f}%")

    expected_pairs = total_positive * (1 + shell_cfg.negative_shells)
    print(f"Expected pair rows from positives: {expected_pairs:,}")
    print(f"Actual pair rows:                  {total_pairs:,}")

    if expected_pairs != total_pairs:
        print(
            "[warn] Actual pair rows do not equal positive_rows * (1 + negative_shells). "
            "This can happen if some events have fewer available negative shells than requested."
        )

    print("\nPer split/fidelity summary:")
    summary = (
        manifest_df.groupby(["split_name", "fidelity"], dropna=False)[
            [
                "original_block_rows",
                "saved_pair_rows",
                "positive_rows",
                "negative_rows",
            ]
        ]
        .sum()
        .reset_index()
    )
    print(summary.to_string(index=False))

    print("\nPer-file pair row statistics:")
    print(
        manifest_df[
            [
                "original_block_rows",
                "saved_pair_rows",
                "positive_rows",
                "negative_rows",
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

    stage_start = log_stage("Computing centered coordinates")
    all_events = loaded.concatenated
    z_center = infer_z_center(all_events, config.shell)
    all_events = add_centered_z_coordinate(all_events, z_center)
    finish_stage(stage_start, f"Computed z_from_center using z_center={z_center:.6g}")

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

    stage_start = log_stage("Building equal-volume shell candidates from LF support")
    candidates_df = build_shell_candidate_table(pools.lf_pool, config.shell)
    finish_stage(stage_start, f"Built {len(candidates_df)} valid shell candidates")
    
    lf_blocks_list = list(lf_blocks.blocks)
    hf_blocks_list = list(hf_blocks.blocks)
    validation_blocks_list = list(validation_blocks.blocks)

    output_dir = config.output.output_dir
    output_format = config.output.output_format

    if output_dir.exists():
        stage_start = log_stage(f"Clearing existing dataset directory: {output_dir}")
        shutil.rmtree(output_dir)
        finish_stage(stage_start, "Removed previous equal-volume shell dataset")

    phi_headers = ["s_r", "s_z_from_center"]
    
    stage_start = log_stage("Writing LF Training event-shell pair H5 blocks")
    lf_manifest = write_h5_all_blocks(
        blocks=lf_blocks_list,
        candidates_df=candidates_df,
        output_dir=output_dir / "training" / "lf",
        split_name="training",
        fidelity="lf",
        phi_headers=phi_headers,
        num_neg_shells=config.shell.negative_shells,
        random_seed=config.split.random_seed,
    )
    finish_stage(stage_start, f"{len(lf_manifest)} LF Training block written")

    stage_start = log_stage("Writing HF Training event-shell pair H5 blocks")
    hf_manifest = write_h5_all_blocks(
        blocks=hf_blocks_list,
        candidates_df=candidates_df,
        output_dir=output_dir / "training" / "hf",
        split_name="training",
        fidelity="hf",
        phi_headers=phi_headers,
        num_neg_shells=config.shell.negative_shells,
        random_seed=config.split.random_seed,
    )
    finish_stage(stage_start, f"{len(hf_manifest)} HF Training block written")

    stage_start = log_stage("Writing HF Validation event-shell pair H5 blocks")
    val_manifest = write_h5_all_blocks(
        blocks=validation_blocks_list,
        candidates_df=candidates_df,
        output_dir=output_dir / "validation" / "hf",
        split_name="validation",
        fidelity="hf",
        phi_headers=phi_headers,
        num_neg_shells=config.shell.negative_shells,
        random_seed=config.split.random_seed,
    )
    finish_stage(stage_start, f"{len(val_manifest)} HF Validation block written")
    
    stage_start = log_stage("Writing pool tables and manifests")
    save_dataframe(all_events, output_dir / "processed_all_events", output_format)
    save_dataframe(pools.lf_pool, output_dir / "lf_training_pool", output_format)
    save_dataframe(pools.hf_pool, output_dir / "hf_training_pool", output_format)
    save_dataframe(pools.validation_pool, output_dir / "hf_validation_pool", output_format)
    candidates_path = save_dataframe(candidates_df, output_dir / "equal_volume_shell_candidates", output_format)
    manifest_df = pd.concat([lf_manifest, hf_manifest, val_manifest], ignore_index=True)
    manifest_path = save_dataframe(manifest_df, output_dir / "equal_volume_shell_file_manifest", output_format)
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
        candidates_df=candidates_df,
        manifest_df=manifest_df,
        shell_cfg=config.shell,
    )
    print(f"\nArtifacts:\n- candidates: {candidates_path}\n- manifest: {manifest_path}")
    total_elapsed = time.perf_counter() - total_start
    print(f"\nTotal elapsed: {total_elapsed:.2f}s")


if __name__ == "__main__":
    main()
