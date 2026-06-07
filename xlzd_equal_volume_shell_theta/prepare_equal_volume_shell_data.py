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
    describe_target_columns,
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
    negative_shells: int = 4

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
            negative_shells=int(shell.get("negative_shells", 4))
        ),
        output=OutputConfig(
            output_dir=Path(output.get("output_dir", "outputs_equal_volume_shell_theta")),
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


def infer_z_center(df: pd.DataFrame, shell_cfg: EqualVolumeShellConfig) -> float:
    if "z" not in df.columns:
        raise ValueError("Dataframe must contain 'z' to infer centered coordinates.")
    if shell_cfg.z_center is not None:
        return float(shell_cfg.z_center)
    return float(0.5 * (df["z"].min() + df["z"].max()))


def shell_boundaries(shell_cfg: EqualVolumeShellConfig) -> pd.DataFrame:
    idx = np.arange(0, shell_cfg.n_shells + 1, dtype=float)
    frac = idx / float(shell_cfg.n_shells)
    scale = np.cbrt(frac)
    r = shell_cfg.R_max * scale
    z = shell_cfg.Z_max * scale
    return pd.DataFrame(
        {
            "shell_level": idx.astype(int),
            "R_boundary": r.astype(float),
            "Z_boundary": z.astype(float),
        }
    )


def inside_equal_volume_shell(
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
    return (inside_outer & ~inside_inner).astype(np.int8)


def build_shell_candidate_table(df_for_support: pd.DataFrame, shell_cfg: EqualVolumeShellConfig) -> pd.DataFrame:
    boundaries = shell_boundaries(shell_cfg)
    support_rows: list[dict[str, float | int]] = []
    shell_volume = 2.0 * np.pi * shell_cfg.Z_max * (shell_cfg.R_max**2) / float(shell_cfg.n_shells)

    for i in range(1, shell_cfg.n_shells + 1):
        prev = boundaries.iloc[i - 1]
        curr = boundaries.iloc[i]
        mask = inside_equal_volume_shell(
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


def choose_shell_sets(
    candidates: pd.DataFrame,
    *,
    n_lf: int,
    n_hf: int,
    n_validation: int,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if n_hf >= n_lf:
        raise ValueError(f"HF block count must be smaller than LF block count, but got m={n_hf}, n={n_lf}.")

    rng = np.random.default_rng(random_seed)
    requested_lf = n_lf
    requested_hf = n_hf
    available = len(candidates)
    requested_unique = n_lf + n_validation
    if available < 2:
        raise ValueError("Need at least 2 valid shell candidates to assign LF and validation shells.")

    if available < requested_unique:
        print("This shouldnt print")
        scale = available / float(requested_unique)
        shrunk_lf = max(2, min(n_lf, int(np.floor(n_lf * scale))))
        shrunk_validation = max(1, min(n_validation, available - shrunk_lf))
        while shrunk_lf + shrunk_validation > available:
            if shrunk_lf >= shrunk_validation and shrunk_lf > 2:
                shrunk_lf -= 1
            elif shrunk_validation > 1:
                shrunk_validation -= 1
            else:
                shrunk_lf -= 1
        print(
            f"[warn] Only {available} shell candidates available for requested LF+validation count "
            f"{requested_unique}; shrinking to LF={shrunk_lf}, validation={shrunk_validation}."
        )
        n_lf = shrunk_lf
        n_validation = shrunk_validation

    # Preserve the original HF/LF ratio after LF shrinkage.
    n_hf = min(
        requested_hf,
        n_lf - 1,
        max(1, int(np.floor(requested_hf * (n_lf / float(max(requested_lf, 1)))))),
    )
    if n_hf >= n_lf:
        n_hf = n_lf - 1

    perm = rng.permutation(available)
    lf_idx = np.sort(perm[:n_lf])
    remaining = perm[n_lf:]
    val_idx = np.sort(remaining[:n_validation])

    lf_df = candidates.iloc[lf_idx].copy().reset_index(drop=True)
    hf_pick = np.sort(rng.choice(np.arange(n_lf), size=n_hf, replace=False))
    hf_df = lf_df.iloc[hf_pick].copy().reset_index(drop=True)
    val_df = candidates.iloc[val_idx].copy().reset_index(drop=True)

    lf_df["random_seed_used"] = random_seed
    lf_df["shell_assignment_instance"] = np.arange(len(lf_df))
    hf_df["random_seed_used"] = random_seed
    hf_df["shell_assignment_instance"] = np.arange(len(hf_df))
    val_df["random_seed_used"] = random_seed
    val_df["validation_shell_instance"] = np.arange(len(val_df))
    return lf_df, hf_df, val_df

def positive_shells_for_block(
    block_df: pd.DataFrame,
    candidates_df: pd.DataFrame,
) -> pd.DataFrame:
    # Find all positive shells for a singular block
    positive_rows = []
    for row in candidates_df.itertuples(index=False):
        mask = inside_equal_volume_shell(
            block_df,
            R_inner=float(row.R_inner),
            Z_inner=float(row.Z_inner),
            R_outer=float(row.R_shell),
            Z_outer=float(row.Z_shell),
        )
        if int(mask.sum()) > 0:
            positive_rows.append(row._asdict())
    return pd.DataFrame(positive_rows)

def build_pos_neg_shell_pairs(
    *,
    blocks: Sequence[pd.DataFrame],
    candidates_df: pd.DataFrame,
    num_neg_shells: int,
    random_seed: int,
) -> pd.DataFrame:
    # Make dataframe that matches up positive shell and given number of negative shells
    rng = np.random.default_rng(random_seed)
    records: list[dict[str, float | int | str]] = []
    all_shell_idxs = set(candidates_df["shell_index"].astype(int))

    # Run through each block
    for block_index, block_df in enumerate(blocks):
        positive_df = positive_shells_for_block(block_df, candidates_df)
        if positive_df.empty:
            continue

        # Make a record of the positive set
        for positive_row in positive_df.itertuples(index=False):
            positive_shell_idx = int(positive_row.shell_index)
            record = positive_row._asdict()
            record["block_index"] = block_index
            record["pair_type"] = "positive"
            record["random_seed_used"] = random_seed
            records.append(record)

            # Choose a certain number of negative sets
            negative_shell_idxs = list(all_shell_idxs - {positive_shell_idx})
            if num_neg_shells > len(negative_shell_idxs):
                sampled_negative_indices = negative_shell_idxs
            else:
                sampled_negative_indices = rng.choice(
                    negative_shell_idxs,
                    size=num_neg_shells,
                    replace=False,
                )
            negative_df = candidates_df[candidates_df["shell_index"].isin(sampled_negative_indices)]

            # Make a record of each negative set
            for negative_row in negative_df.itertuples(index=False):
                record = negative_row._asdict()
                record["block_index"] = block_index
                record["pair_type"] = "negative"
                record["random_seed_used"] = random_seed
                records.append(record)
                
    if not records:
        raise RuntimeError("No positive/negative shell pairs were created")
    return pd.DataFrame.from_records(records).reset_index(drop=True)


def _format_float_for_filename(value: float, decimals: int = 3) -> str:
    text = f"{value:.{decimals}f}"
    return text.replace("-", "m").replace(".", "p")


def _build_shell_filename(
    fidelity: str,
    pair_type: str,
    block_index: int,
    shell_index: int,
    R_shell: float,
    Z_shell: float,
    *,
    extension: str,
    existing_names: set[str],
) -> str:
    base = (
        f"{fidelity}_"
        f"{pair_type}_"
        f"block{block_index:04d}_"
        f"shell{shell_index:03d}_"
        f"R{_format_float_for_filename(R_shell)}_"
        f"Z{_format_float_for_filename(Z_shell)}"
    )
    candidate = f"{base}.{extension}"
    duplicate_index = 2
    while candidate in existing_names:
        candidate = f"{base}__dup{duplicate_index}.{extension}"
        duplicate_index += 1
    existing_names.add(candidate)
    return candidate


def write_shell_block_files(
    *,
    blocks: Sequence[pd.DataFrame],
    pair_df: pd.DataFrame,
    split_name: str,
    fidelity: str,
    output_dir: Path,
    output_format: str,
) -> pd.DataFrame:
    used_blocks = set(pair_df["block_index"].astype(int))
    expected_blocks = set(range(len(blocks)))
    missing_blocks = expected_blocks - used_blocks
    if missing_blocks:
        print(
            f"[warn] {len(missing_blocks)} blocks were never assigned a shell pair. "
            f"Examples: {sorted(list(missing_blocks))[:10]}"
        )
    if len(used_blocks) == 0:
        raise RuntimeError(
            "No valid block-shell pairs were generated."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    existing_names: set[str] = set()
    records: list[dict[str, float | int | str]] = []
    extension = output_format.lower()

    for pair_index, row in enumerate(pair_df.itertuples(index=False)):
        block_index = int(row.block_index)
        block_df = blocks[block_index]
        out_df = block_df.copy()

        R_inner = float(row.R_inner)
        Z_inner = float(row.Z_inner)
        R_outer = float(row.R_shell)
        Z_outer = float(row.Z_shell)
        shell_index = int(row.shell_index)

        out_df["split_name"] = split_name
        out_df["fidelity"] = fidelity
        out_df["block_index"] = block_index
        out_df["pair_index"] = pair_index
        out_df["pair_type"] = row.pair_type

        out_df["shell_index"] = shell_index
        out_df["R_inner"] = R_inner
        out_df["Z_inner"] = Z_inner
        out_df["R_shell"] = R_outer
        out_df["Z_shell"] = Z_outer

        out_df[TARGET_COLUMN] = inside_equal_volume_shell(
            out_df,
            R_inner=R_inner,
            Z_inner=Z_inner,
            R_outer=R_outer,
            Z_outer=Z_outer,
        ).astype(np.int8)
        
        filename = _build_shell_filename(
            fidelity=fidelity,
            pair_type=row.pair_type,
            block_index=block_index,
            shell_index=shell_index,
            R_shell=R_outer,
            Z_shell=Z_outer,
            extension=extension,
            existing_names=existing_names,
        )
        written_path = save_dataframe(out_df, output_dir / Path(filename).stem, output_format)

        inside_count = int(out_df[TARGET_COLUMN].sum())
        records.append(
            {
                "pair_index": int(pair_index),
                "pair_type": str(row.pair_type),
                "block_index": int(block_index),
                "random_seed_used": int(row.random_seed_used),
                "split_name": split_name,
                "fidelity": fidelity,
                "file_name": written_path.name,
                "file_path": str(written_path),
                "sample_size": int(len(out_df)),
                "shell_index": shell_index,
                "R_inner": R_inner,
                "Z_inner": Z_inner,
                "R_shell": R_outer,
                "Z_shell": Z_outer,
                "inside_shell_count": inside_count,
                "inside_shell_fraction": float(inside_count / len(out_df)),
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
    print(f"Total events: {len(all_events):,}")
    print(f"z_center used: {z_center:.6g}")
    print(f"Final-position r range: [{all_events['r'].min():.6g}, {all_events['r'].max():.6g}]")
    print(f"Final-position z_from_center range: [{all_events[Z_FROM_CENTER_COLUMN].min():.6g}, {all_events[Z_FROM_CENTER_COLUMN].max():.6g}]")
    print(f"R_max={shell_cfg.R_max:.6g}, Z_max={shell_cfg.Z_max:.6g}, n_shells={shell_cfg.n_shells}")
    print(f"Valid equal-volume shells: {len(candidates_df):,}")
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
    print(
        describe_target_columns(
            manifest_df,
            ["sample_size", "inside_shell_count", "inside_shell_fraction"],
        ).to_string()
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

    stage_start = log_stage("Assigning positive/negative shell pairs to LF/HF/Validation Blocks")
    lf_pair_df = build_pos_neg_shell_pairs(
        blocks=lf_blocks.blocks,
        candidates_df=candidates_df,
        num_neg_shells=config.shell.negative_shells,
        random_seed=config.split.random_seed,
    )
    hf_pair_df = build_pos_neg_shell_pairs(
        blocks=hf_blocks.blocks,
        candidates_df=candidates_df,
        num_neg_shells=config.shell.negative_shells,
        random_seed=config.split.random_seed + 1,
    )
    validation_pair_df = build_pos_neg_shell_pairs(
        blocks=validation_blocks.blocks,
        candidates_df=candidates_df,
        num_neg_shells=config.shell.negative_shells,
        random_seed=config.split.random_seed + 2,
    )

    finish_stage(
        stage_start,
        f"Assigned {len(lf_pair_df)} LF pairs, "
        f"{len(hf_pair_df)} HF pairs, "
        f"{len(validation_pair_df)} validation pairs"
    )

    lf_blocks_list = list(lf_blocks.blocks)
    hf_blocks_list = list(hf_blocks.blocks)
    validation_blocks_list = list(validation_blocks.blocks)

    output_dir = config.output.output_dir
    output_format = config.output.output_format

    if output_dir.exists():
        stage_start = log_stage(f"Clearing existing dataset directory: {output_dir}")
        shutil.rmtree(output_dir)
        finish_stage(stage_start, "Removed previous equal-volume shell dataset")

    stage_start = log_stage("Writing LF equal-volume shell block files")
    lf_manifest = write_shell_block_files(
        blocks=lf_blocks_list,
        pair_df=lf_pair_df,
        split_name="training",
        fidelity="lf",
        output_dir=output_dir / "training" / "lf",
        output_format=output_format,
    )
    finish_stage(stage_start, "LF shell block files complete")

    stage_start = log_stage("Writing HF training equal-volume shell block files")
    hf_manifest = write_shell_block_files(
        blocks=hf_blocks_list,
        pair_df=hf_pair_df,
        split_name="training",
        fidelity="hf",
        output_dir=output_dir / "training" / "hf",
        output_format=output_format,
    )
    finish_stage(stage_start, "HF training shell block files complete")

    stage_start = log_stage("Writing HF validation equal-volume shell block files")
    validation_manifest = write_shell_block_files(
        blocks=validation_blocks_list,
        pair_df=validation_pair_df,
        split_name="validation",
        fidelity="hf",
        output_dir=output_dir / "validation" / "hf",
        output_format=output_format,
    )
    finish_stage(stage_start, "HF validation shell block files complete")

    stage_start = log_stage("Writing pool tables and manifests")
    save_dataframe(all_events, output_dir / "processed_all_events", output_format)
    save_dataframe(pools.lf_pool, output_dir / "lf_training_pool", output_format)
    save_dataframe(pools.hf_pool, output_dir / "hf_training_pool", output_format)
    save_dataframe(pools.validation_pool, output_dir / "hf_validation_pool", output_format)
    candidates_path = save_dataframe(candidates_df, output_dir / "equal_volume_shell_candidates", output_format)
    manifest_df = pd.concat([lf_manifest, hf_manifest, validation_manifest], ignore_index=True)
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
