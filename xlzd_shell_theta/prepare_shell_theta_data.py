#!/usr/bin/env python3
"""Prepare XLZD shell-theta event files for CNP/MF-GP.

This is a separate pipeline from the cumulative `(R_max, Z_max)` volume-theta workflow.

Shell-theta definition:
- theta = (r_shell, z_shell)
- target = near_shell
- near_shell = 1 if:
    |r - r_shell| <= delta_r
    |z_from_center - z_shell| <= delta_z
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from xlzd_resum.config import (
    DEFAULT_FILE_STEMS,
    FileLoadConfig,
    OutputConfig,
    SamplingConfig,
    SplitConfig,
)
from xlzd_resum.dataset import (
    describe_target_columns,
    split_into_disjoint_pools,
    split_pool_into_blocks,
)
from xlzd_resum.io_utils import load_event_collection, save_dataframe
from xlzd_resum.theta import Z_FROM_CENTER_COLUMN, add_centered_z_coordinate


THETA_HEADERS = ["r_shell", "z_shell"]
TARGET_COLUMN = "near_shell"


@dataclass(slots=True)
class ShellGridConfig:
    r_shell_min: float = 0.0
    r_shell_max: float = 1500.0
    z_shell_min: float = 0.0
    z_shell_max: float = 2000.0
    r_shell_step: float = 100.0
    z_shell_step: float = 100.0
    delta_r: float = 50.0
    delta_z: float = 50.0
    min_candidate_events: int = 25
    z_center: float | None = 1982.48
    rounding_decimals: int = 6

    def validate(self) -> None:
        if self.r_shell_step <= 0 or self.z_shell_step <= 0:
            raise ValueError("Shell grid steps must be positive.")
        if self.delta_r <= 0 or self.delta_z <= 0:
            raise ValueError("Shell widths delta_r and delta_z must be positive.")
        if self.r_shell_min >= self.r_shell_max:
            raise ValueError("r_shell_min must be smaller than r_shell_max.")
        if self.z_shell_min >= self.z_shell_max:
            raise ValueError("z_shell_min must be smaller than z_shell_max.")
        if self.min_candidate_events <= 0:
            raise ValueError("min_candidate_events must be positive.")
        if self.rounding_decimals < 0:
            raise ValueError("rounding_decimals must be non-negative.")


@dataclass(slots=True)
class ShellPipelineConfig:
    file_load: FileLoadConfig
    split: SplitConfig
    sampling: SamplingConfig
    shell: ShellGridConfig
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
        default=Path("xlzd_shell_theta/config/pipeline_config.json"),
        help="Path to shell-theta JSON config file.",
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
        shell=ShellGridConfig(
            r_shell_min=float(shell.get("r_shell_min", 0.0)),
            r_shell_max=float(shell.get("r_shell_max", 1500.0)),
            z_shell_min=float(shell.get("z_shell_min", 0.0)),
            z_shell_max=float(shell.get("z_shell_max", 2000.0)),
            r_shell_step=float(shell.get("r_shell_step", 100.0)),
            z_shell_step=float(shell.get("z_shell_step", 100.0)),
            delta_r=float(shell.get("delta_r", 50.0)),
            delta_z=float(shell.get("delta_z", 50.0)),
            min_candidate_events=int(shell.get("min_candidate_events", 25)),
            z_center=shell.get("z_center"),
            rounding_decimals=int(shell.get("rounding_decimals", 6)),
        ),
        output=OutputConfig(
            output_dir=Path(output.get("output_dir", "outputs_shell_theta")),
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


def _shell_key(r_shell: float, z_shell: float, decimals: int) -> tuple[float, float]:
    return (round(float(r_shell), decimals), round(float(z_shell), decimals))


def infer_z_center_shell(df: pd.DataFrame, shell_cfg: ShellGridConfig) -> float:
    if "z" not in df.columns:
        raise ValueError("Dataframe must contain 'z' to infer centered coordinates.")
    if shell_cfg.z_center is not None:
        return float(shell_cfg.z_center)
    return float(0.5 * (df["z"].min() + df["z"].max()))


def build_shell_candidate_table(df_for_support: pd.DataFrame, shell_cfg: ShellGridConfig) -> pd.DataFrame:
    required = {"r", Z_FROM_CENTER_COLUMN}
    if not required.issubset(df_for_support.columns):
        raise ValueError(f"Support dataframe must contain columns {required}.")

    r_values = np.arange(shell_cfg.r_shell_min, shell_cfg.r_shell_max + 1e-9, shell_cfg.r_shell_step, dtype=float)
    z_values = np.arange(shell_cfg.z_shell_min, shell_cfg.z_shell_max + 1e-9, shell_cfg.z_shell_step, dtype=float)

    support_r = df_for_support["r"].to_numpy(dtype=float)
    support_z = df_for_support[Z_FROM_CENTER_COLUMN].to_numpy(dtype=float)

    rows: list[dict[str, float | int | str]] = []
    for r_shell in r_values:
        for z_shell in z_values:
            mask = (np.abs(support_r - r_shell) <= shell_cfg.delta_r) & (np.abs(support_z - z_shell) <= shell_cfg.delta_z)
            count = int(mask.sum())
            if count < shell_cfg.min_candidate_events:
                continue
            rows.append(
                {
                    "r_shell": float(r_shell),
                    "z_shell": float(z_shell),
                    "candidate_events": count,
                    "shell_key": f"{r_shell:.{shell_cfg.rounding_decimals}f}|{z_shell:.{shell_cfg.rounding_decimals}f}",
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("No valid shell-theta candidates were found. Loosen the shell grid or lower min_candidate_events.")
    return out.sort_values(["r_shell", "z_shell"]).reset_index(drop=True)


def choose_shell_theta_sets(
    candidates: pd.DataFrame,
    *,
    n_lf: int,
    n_hf: int,
    n_validation: int,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if n_hf >= n_lf:
        raise ValueError(f"HF block count must be smaller than LF block count, but got m={n_hf}, n={n_lf}.")
    if len(candidates) < (n_lf + n_validation):
        raise ValueError(
            f"Not enough shell-theta candidates. Need at least {n_lf + n_validation}, found {len(candidates)}."
        )

    rng = np.random.default_rng(random_seed)
    perm = rng.permutation(len(candidates))
    lf_idx = np.sort(perm[:n_lf])
    remaining = perm[n_lf:]
    val_idx = np.sort(remaining[:n_validation])

    lf_theta_df = candidates.iloc[lf_idx][["r_shell", "z_shell", "shell_key"]].copy().reset_index(drop=True)
    hf_pick = np.sort(rng.choice(np.arange(n_lf), size=n_hf, replace=False))
    hf_theta_df = lf_theta_df.iloc[hf_pick].copy().reset_index(drop=True)
    validation_theta_df = candidates.iloc[val_idx][["r_shell", "z_shell", "shell_key"]].copy().reset_index(drop=True)

    lf_theta_df["random_seed_used"] = random_seed
    hf_theta_df["random_seed_used"] = random_seed
    validation_theta_df["random_seed_used"] = random_seed
    validation_theta_df["validation_theta_instance"] = np.arange(len(validation_theta_df))
    return lf_theta_df, hf_theta_df, validation_theta_df


def _format_float_for_filename(value: float, decimals: int = 3) -> str:
    text = f"{value:.{decimals}f}"
    return text.replace("-", "m").replace(".", "p")


def _build_shell_filename(
    fidelity: str,
    r_shell: float,
    z_shell: float,
    *,
    extension: str,
    existing_names: set[str],
) -> str:
    base = (
        f"{fidelity}_"
        f"rS{_format_float_for_filename(r_shell)}_"
        f"zS{_format_float_for_filename(z_shell)}"
    )
    candidate = f"{base}.{extension}"
    duplicate_index = 2
    while candidate in existing_names:
        candidate = f"{base}__dup{duplicate_index}.{extension}"
        duplicate_index += 1
    existing_names.add(candidate)
    return candidate


def near_shell_mask(df: pd.DataFrame, *, r_shell: float, z_shell: float, delta_r: float, delta_z: float) -> pd.Series:
    required = {"r", Z_FROM_CENTER_COLUMN}
    if not required.issubset(df.columns):
        raise ValueError(f"Block dataframe must contain columns {required}.")
    return (np.abs(df["r"].to_numpy(dtype=float) - float(r_shell)) <= float(delta_r)) & (
        np.abs(df[Z_FROM_CENTER_COLUMN].to_numpy(dtype=float) - float(z_shell)) <= float(delta_z)
    )


def write_shell_theta_block_files(
    *,
    blocks: Sequence[pd.DataFrame],
    theta_df: pd.DataFrame,
    split_name: str,
    fidelity: str,
    output_dir: Path,
    output_format: str,
    shell_cfg: ShellGridConfig,
) -> pd.DataFrame:
    if len(blocks) != len(theta_df):
        raise ValueError(f"Number of blocks and theta rows must match for {split_name}: {len(blocks)} != {len(theta_df)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    existing_names: set[str] = set()
    records: list[dict[str, float | int | str]] = []
    extension = output_format.lower()

    for block_index, (block_df, row) in enumerate(zip(blocks, theta_df.itertuples(index=False), strict=True)):
        r_shell = float(row.r_shell)
        z_shell = float(row.z_shell)
        out_df = block_df.copy()
        out_df["split_name"] = split_name
        out_df["fidelity"] = fidelity
        out_df["r_shell"] = r_shell
        out_df["z_shell"] = z_shell
        out_df[TARGET_COLUMN] = near_shell_mask(
            out_df,
            r_shell=r_shell,
            z_shell=z_shell,
            delta_r=shell_cfg.delta_r,
            delta_z=shell_cfg.delta_z,
        ).astype(np.int8)

        filename = _build_shell_filename(
            fidelity=fidelity,
            r_shell=r_shell,
            z_shell=z_shell,
            extension=extension,
            existing_names=existing_names,
        )
        written_path = save_dataframe(out_df, output_dir / Path(filename).stem, output_format)

        near_count = int(out_df[TARGET_COLUMN].sum())
        records.append(
            {
                "block_index": int(block_index),
                "random_seed_used": int(row.random_seed_used),
                "split_name": split_name,
                "fidelity": fidelity,
                "file_name": written_path.name,
                "file_path": str(written_path),
                "sample_size": int(len(out_df)),
                "r_shell": r_shell,
                "z_shell": z_shell,
                "near_shell_count": near_count,
                "near_shell_fraction": float(near_count / len(out_df)),
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
    shell_cfg: ShellGridConfig,
) -> None:
    print("\n=== XLZD Shell Theta Summary ===")
    print(f"Files loaded: {len(files_loaded)}")
    print(f"Total events: {len(all_events):,}")
    print(f"z_center used: {z_center:.6g}")
    print(f"Final-position r range: [{all_events['r'].min():.6g}, {all_events['r'].max():.6g}]")
    print(f"Final-position z_from_center range: [{all_events[Z_FROM_CENTER_COLUMN].min():.6g}, {all_events[Z_FROM_CENTER_COLUMN].max():.6g}]")
    print(f"Shell widths: delta_r={shell_cfg.delta_r:.6g}, delta_z={shell_cfg.delta_z:.6g}")
    print(f"Valid shell-theta candidates: {len(candidates_df):,}")
    print(
        f"Pool sizes: LF={pool_sizes['lf']:,}, HF={pool_sizes['hf']:,}, VAL={pool_sizes['validation']:,}"
    )
    print(
        f"Block counts: n(LF)={block_counts['lf']}, m(HF)={block_counts['hf']}, k(VAL)={block_counts['validation']}"
    )
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
            ["sample_size", "near_shell_count", "near_shell_fraction"],
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
    z_center = infer_z_center_shell(all_events, config.shell)
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

    stage_start = log_stage("Building shell-theta candidate grid from LF pool support")
    candidates_df = build_shell_candidate_table(pools.lf_pool, config.shell)
    finish_stage(stage_start, f"Built {len(candidates_df)} valid shell-theta candidates")

    stage_start = log_stage("Assigning shell-theta values to LF/HF/validation blocks")
    lf_theta_df, hf_theta_df, validation_theta_df = choose_shell_theta_sets(
        candidates_df,
        n_lf=len(lf_blocks.blocks),
        n_hf=len(hf_blocks.blocks),
        n_validation=len(validation_blocks.blocks),
        random_seed=config.split.random_seed,
    )
    finish_stage(
        stage_start,
        f"Assigned n={len(lf_theta_df)} LF shell thetas, m={len(hf_theta_df)} HF shell thetas, k={len(validation_theta_df)} validation shell thetas",
    )

    output_dir = config.output.output_dir
    output_format = config.output.output_format

    stage_start = log_stage("Writing LF shell-theta block files")
    lf_manifest = write_shell_theta_block_files(
        blocks=lf_blocks.blocks,
        theta_df=lf_theta_df,
        split_name="training",
        fidelity="lf",
        output_dir=output_dir / "training" / "lf",
        output_format=output_format,
        shell_cfg=config.shell,
    )
    finish_stage(stage_start, "LF shell-theta block files complete")

    stage_start = log_stage("Writing HF training shell-theta block files")
    hf_manifest = write_shell_theta_block_files(
        blocks=hf_blocks.blocks,
        theta_df=hf_theta_df,
        split_name="training",
        fidelity="hf",
        output_dir=output_dir / "training" / "hf",
        output_format=output_format,
        shell_cfg=config.shell,
    )
    finish_stage(stage_start, "HF training shell-theta block files complete")

    stage_start = log_stage("Writing HF validation shell-theta block files")
    validation_manifest = write_shell_theta_block_files(
        blocks=validation_blocks.blocks,
        theta_df=validation_theta_df,
        split_name="validation",
        fidelity="hf",
        output_dir=output_dir / "validation" / "hf",
        output_format=output_format,
        shell_cfg=config.shell,
    )
    finish_stage(stage_start, "HF validation shell-theta block files complete")

    stage_start = log_stage("Writing pool tables and manifests")
    save_dataframe(all_events, output_dir / "processed_all_events", output_format)
    save_dataframe(pools.lf_pool, output_dir / "lf_training_pool", output_format)
    save_dataframe(pools.hf_pool, output_dir / "hf_training_pool", output_format)
    save_dataframe(pools.validation_pool, output_dir / "hf_validation_pool", output_format)
    candidates_path = save_dataframe(candidates_df, output_dir / "shell_theta_candidates", output_format)
    manifest_df = pd.concat([lf_manifest, hf_manifest, validation_manifest], ignore_index=True)
    manifest_path = save_dataframe(manifest_df, output_dir / "shell_theta_file_manifest", output_format)
    finish_stage(stage_start, "Output files written")

    pool_sizes = {"lf": len(pools.lf_pool), "hf": len(pools.hf_pool), "validation": len(pools.validation_pool)}
    block_counts = {"lf": len(lf_blocks.blocks), "hf": len(hf_blocks.blocks), "validation": len(validation_blocks.blocks)}
    block_size_ranges = {
        "lf": (lf_blocks.min_block_size, lf_blocks.max_block_size),
        "hf": (hf_blocks.min_block_size, hf_blocks.max_block_size),
        "validation": (validation_blocks.min_block_size, validation_blocks.max_block_size),
    }
    leftover_rows = {"lf": lf_blocks.leftover_rows, "hf": hf_blocks.leftover_rows, "validation": validation_blocks.leftover_rows}

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
