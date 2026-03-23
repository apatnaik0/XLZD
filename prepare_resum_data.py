#!/usr/bin/env python3
"""Prepare XLZD 0nuBB background events using pool/block-based centered-theta sampling.

Pipeline summary:
1. Load the raw component CSV/text files and normalize them to the expected event schema.
2. Compute final-position radius r and centered axial distance z_from_center.
3. Split raw events into three disjoint pools: LF training, HF training, HF validation.
4. Split each pool into equal-size event blocks.
5. Sample centered theta parameters (R_max, Z_max) from the LF pool support.
6. Use LF theta values for LF blocks, choose an HF subset for HF blocks, and assign HF-based
   theta values to validation blocks.
7. Write one file per block with an inside_theta column.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import pandas as pd

from xlzd_resum.config import (
    DEFAULT_FILE_STEMS,
    FileLoadConfig,
    OutputConfig,
    PipelineConfig,
    SamplingConfig,
    SplitConfig,
    ThetaSamplingConfig,
)
from xlzd_resum.dataset import (
    build_theta_sets_from_blocks,
    describe_target_columns,
    split_into_disjoint_pools,
    split_pool_into_blocks,
    write_theta_block_files,
)
from xlzd_resum.io_utils import load_event_collection, save_dataframe
from xlzd_resum.theta import add_centered_z_coordinate, infer_z_center


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/pipeline_config.json"),
        help="Path to JSON config file.",
    )
    return parser.parse_args()


def load_json_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_config(args: argparse.Namespace) -> PipelineConfig:
    raw = load_json_config(args.config)

    file_load = raw.get("file_load", {})
    split = raw.get("split", {})
    theta = raw.get("theta", {})
    sampling = raw.get("sampling", {})
    output = raw.get("output", {})

    config = PipelineConfig(
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
        theta=ThetaSamplingConfig(
            z_lower=theta.get("z_lower"),
            z_upper=theta.get("z_upper"),
            r_lower=theta.get("r_lower"),
            r_upper=theta.get("r_upper"),
            min_z_width=float(theta.get("min_z_width", 1.0)),
            min_r_width=float(theta.get("min_r_width", 1.0)),
            z_center=theta.get("z_center"),
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
        output=OutputConfig(
            output_dir=Path(output.get("output_dir", "outputs")),
            output_format=str(output.get("output_format", "csv")),
        ),
    )
    config.validate()
    return config


def log_stage(message: str) -> float:
    print(f"\n[{time.strftime('%H:%M:%S')}] {message}")
    return time.perf_counter()


def finish_stage(stage_start: float, message: str) -> None:
    elapsed = time.perf_counter() - stage_start
    print(f"[done in {elapsed:.2f}s] {message}")


def print_summary(
    *,
    files_loaded: list[Path],
    all_events: pd.DataFrame,
    z_center: float,
    pool_sizes: dict[str, int],
    block_counts: dict[str, int],
    block_size_ranges: dict[str, tuple[int, int]],
    leftover_rows: dict[str, int],
    manifest_df: pd.DataFrame,
) -> None:
    print("\n=== XLZD Pool/Block Theta Summary ===")
    print(f"Files loaded: {len(files_loaded)}")
    print(f"Total events: {len(all_events):,}")
    print(f"Inferred z_center: {z_center:.6g}")
    print(f"Final-position z range: [{all_events['z'].min():.6g}, {all_events['z'].max():.6g}]")
    print(f"Final-position r range: [{all_events['r'].min():.6g}, {all_events['r'].max():.6g}]")
    print(
        f"Pool sizes: LF={pool_sizes['lf']:,}, HF={pool_sizes['hf']:,}, "
        f"VAL={pool_sizes['validation']:,}"
    )
    print(
        f"Block counts: n(LF)={block_counts['lf']}, m(HF)={block_counts['hf']}, "
        f"k(VAL)={block_counts['validation']}"
    )
    print(
        f"Block size ranges: LF={block_size_ranges['lf'][0]}-{block_size_ranges['lf'][1]}, "
        f"HF={block_size_ranges['hf'][0]}-{block_size_ranges['hf'][1]}, "
        f"VAL={block_size_ranges['validation'][0]}-{block_size_ranges['validation'][1]}"
    )
    print(
        f"Unused leftover rows: LF={leftover_rows['lf']}, HF={leftover_rows['hf']}, "
        f"VAL={leftover_rows['validation']}"
    )

    print("\nGenerated file statistics:")
    print(
        describe_target_columns(
            manifest_df,
            ["sample_size", "inside_theta_count", "inside_theta_fraction"],
        ).to_string()
    )


def main() -> None:
    args = parse_args()
    config = build_config(args)
    total_start = time.perf_counter()

    stage_start = log_stage("Loading and normalizing raw event files")
    loaded = load_event_collection(config.file_load)
    finish_stage(stage_start, f"Loaded {len(loaded.files_loaded)} files")

    stage_start = log_stage("Computing centered z coordinate")
    all_events = loaded.concatenated
    z_center = infer_z_center(all_events, config.theta)
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
    validation_blocks = split_pool_into_blocks(
        pools.validation_pool,
        block_size=validation_block_size,
    )
    finish_stage(stage_start, "Pool block split complete")

    stage_start = log_stage("Sampling centered theta values from LF support")
    lf_theta_df, hf_theta_df, validation_theta_df = build_theta_sets_from_blocks(
        lf_blocks=lf_blocks,
        hf_blocks=hf_blocks,
        validation_blocks=validation_blocks,
        theta_config=config.theta,
        df_for_bounds=pools.lf_pool,
        random_seed=config.split.random_seed,
    )
    finish_stage(
        stage_start,
        (
            f"Generated n={len(lf_theta_df)} LF thetas, m={len(hf_theta_df)} HF thetas, "
            f"k={len(validation_theta_df)} validation theta assignments"
        ),
    )

    output_dir = config.output.output_dir
    output_format = config.output.output_format

    stage_start = log_stage("Writing LF block files")
    lf_manifest = write_theta_block_files(
        blocks=lf_blocks,
        theta_df=lf_theta_df,
        split_name="training",
        fidelity="lf",
        output_dir=output_dir / "training" / "lf",
        output_format=output_format,
        progress=config.sampling.progress,
    )
    finish_stage(stage_start, "LF block files complete")

    stage_start = log_stage("Writing HF training block files")
    hf_manifest = write_theta_block_files(
        blocks=hf_blocks,
        theta_df=hf_theta_df,
        split_name="training",
        fidelity="hf",
        output_dir=output_dir / "training" / "hf",
        output_format=output_format,
        progress=config.sampling.progress,
    )
    finish_stage(stage_start, "HF training block files complete")

    stage_start = log_stage("Writing HF validation block files")
    validation_manifest = write_theta_block_files(
        blocks=validation_blocks,
        theta_df=validation_theta_df,
        split_name="validation",
        fidelity="hf",
        output_dir=output_dir / "validation" / "hf",
        output_format=output_format,
        progress=config.sampling.progress,
    )
    finish_stage(stage_start, "HF validation block files complete")

    stage_start = log_stage("Writing pool tables and manifests")
    all_events_path = save_dataframe(all_events, output_dir / "processed_all_events", output_format)
    lf_pool_path = save_dataframe(pools.lf_pool, output_dir / "lf_training_pool", output_format)
    hf_pool_path = save_dataframe(pools.hf_pool, output_dir / "hf_training_pool", output_format)
    val_pool_path = save_dataframe(
        pools.validation_pool,
        output_dir / "hf_validation_pool",
        output_format,
    )
    manifest_df = pd.concat([lf_manifest, hf_manifest, validation_manifest], ignore_index=True)
    manifest_path = save_dataframe(manifest_df, output_dir / "theta_file_manifest", output_format)
    finish_stage(stage_start, "Output files written")

    pool_sizes = {
        "lf": len(pools.lf_pool),
        "hf": len(pools.hf_pool),
        "validation": len(pools.validation_pool),
    }
    block_counts = {
        "lf": len(lf_blocks.blocks),
        "hf": len(hf_blocks.blocks),
        "validation": len(validation_blocks.blocks),
    }
    block_size_ranges = {
        "lf": (lf_blocks.min_block_size, lf_blocks.max_block_size),
        "hf": (hf_blocks.min_block_size, hf_blocks.max_block_size),
        "validation": (validation_blocks.min_block_size, validation_blocks.max_block_size),
    }
    leftover_rows = {
        "lf": lf_blocks.leftover_rows,
        "hf": hf_blocks.leftover_rows,
        "validation": validation_blocks.leftover_rows,
    }

    print_summary(
        files_loaded=loaded.files_loaded,
        all_events=all_events,
        z_center=z_center,
        pool_sizes=pool_sizes,
        block_counts=block_counts,
        block_size_ranges=block_size_ranges,
        leftover_rows=leftover_rows,
        manifest_df=manifest_df,
    )

    print("\nWritten outputs:")
    print(all_events_path)
    print(lf_pool_path)
    print(hf_pool_path)
    print(val_pool_path)
    print(manifest_path)
    print(output_dir / "training" / "lf")
    print(output_dir / "training" / "hf")
    print(output_dir / "validation" / "hf")
    print(f"\nTotal runtime: {time.perf_counter() - total_start:.2f}s")


if __name__ == "__main__":
    main()
