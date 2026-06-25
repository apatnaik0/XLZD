"""Dataset-building utilities for pool/block-based centered-theta sampling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - fallback when tqdm is unavailable
    def tqdm(iterable, **_: object):  # type: ignore[misc]
        return iterable

from .config import SamplingConfig, SplitConfig, ThetaSamplingConfig
from .io_utils import save_dataframe
from .theta import ThetaRegion, infer_theta_bounds, sample_theta_region, theta_mask


@dataclass(slots=True)
class PoolSplitResult:
    """Disjoint raw event pools for LF/HF training and HF validation."""

    lf_pool: pd.DataFrame
    hf_pool: pd.DataFrame
    validation_pool: pd.DataFrame


@dataclass(slots=True)
class BlockSet:
    """Near-equal-size blocks created from one raw pool."""

    blocks: list[pd.DataFrame]
    block_size: int
    min_block_size: int
    max_block_size: int
    used_rows: int
    leftover_rows: int


def split_into_disjoint_pools(df: pd.DataFrame, config: SplitConfig) -> PoolSplitResult:
    """Shuffle once and split raw events into three disjoint pools by proportion."""

    config.validate()
    if df.empty:
        raise ValueError("Cannot split an empty dataframe.")

    rng = np.random.default_rng(config.random_seed)
    perm = rng.permutation(len(df))
    shuffled = df.iloc[perm].reset_index(drop=True)

    n_total = len(shuffled)
    n_lf = int(np.floor(n_total * config.lf_pool_fraction))
    n_hf = int(np.floor(n_total * config.hf_pool_fraction))
    n_val = n_total - n_lf - n_hf

    if min(n_lf, n_hf, n_val) <= 0:
        raise ValueError(
            "Pool split produced an empty LF, HF, or validation pool. "
            "Adjust the pool fractions."
        )

    lf_pool = shuffled.iloc[:n_lf].reset_index(drop=True)
    hf_pool = shuffled.iloc[n_lf : n_lf + n_hf].reset_index(drop=True)
    validation_pool = shuffled.iloc[n_lf + n_hf :].reset_index(drop=True)
    return PoolSplitResult(lf_pool=lf_pool, hf_pool=hf_pool, validation_pool=validation_pool)


def split_pool_into_blocks(pool_df: pd.DataFrame, *, block_size: int) -> BlockSet:
    """Split one pool into near-equal-size blocks while using as many rows as possible.

    The requested ``block_size`` is treated as the nominal target size. The function creates
    ``floor(len(pool_df) / block_size)`` blocks and distributes the remainder rows across those
    blocks so that block sizes differ by at most one row.
    """

    if block_size <= 0:
        raise ValueError("block_size must be positive.")
    if pool_df.empty:
        return BlockSet(
            blocks=[],
            block_size=block_size,
            min_block_size=0,
            max_block_size=0,
            used_rows=0,
            leftover_rows=0,
        )

    num_blocks = len(pool_df) // block_size
    if num_blocks == 0:
        return BlockSet(
            blocks=[],
            block_size=block_size,
            min_block_size=0,
            max_block_size=0,
            used_rows=0,
            leftover_rows=len(pool_df),
        )

    base_size = len(pool_df) // num_blocks
    remainder = len(pool_df) % num_blocks

    blocks: list[pd.DataFrame] = []
    start = 0
    for block_index in range(num_blocks):
        current_size = base_size + (1 if block_index < remainder else 0)
        end = start + current_size
        blocks.append(pool_df.iloc[start:end].reset_index(drop=True))
        start = end

    sizes = [len(block) for block in blocks]
    return BlockSet(
        blocks=blocks,
        block_size=block_size,
        min_block_size=min(sizes),
        max_block_size=max(sizes),
        used_rows=sum(sizes),
        leftover_rows=len(pool_df) - sum(sizes),
    )


def _build_theta_table(
    *,
    theta_config: ThetaSamplingConfig,
    df_for_bounds: pd.DataFrame,
    rng: np.random.Generator,
    random_seed: int,
    num_thetas: int,
    excluded_theta_keys: set[tuple[float, float]] | None = None,
    rounding_decimals: int = 6,
) -> pd.DataFrame:
    """Generate a dataframe of centered theta regions."""

    bounds = infer_theta_bounds(df_for_bounds, theta_config)
    excluded_theta_keys = set() if excluded_theta_keys is None else set(excluded_theta_keys)
    records: list[dict[str, float | int]] = []
    while len(records) < num_thetas:
        theta = sample_theta_region(rng, bounds)
        theta_key = (
            round(float(theta.R_max), rounding_decimals),
            round(float(theta.Z_max), rounding_decimals),
        )
        if theta_key in excluded_theta_keys:
            continue
        excluded_theta_keys.add(theta_key)
        records.append(
            {
                "random_seed_used": random_seed,
                **theta.to_dict(),
            }
        )
    return pd.DataFrame.from_records(records)


def build_theta_sets_from_blocks(
    *,
    lf_blocks: BlockSet,
    hf_blocks: BlockSet,
    validation_blocks: BlockSet,
    theta_config: ThetaSamplingConfig,
    df_for_bounds: pd.DataFrame,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build LF/HF training theta sets and a separate held-out validation theta set."""

    if not lf_blocks.blocks:
        raise ValueError("LF pool produced zero blocks.")
    if not hf_blocks.blocks:
        raise ValueError("HF pool produced zero blocks.")
    if not validation_blocks.blocks:
        raise ValueError("Validation pool produced zero blocks.")

    n = len(lf_blocks.blocks)
    m = len(hf_blocks.blocks)
    k = len(validation_blocks.blocks)
    if m >= n:
        raise ValueError(
            f"HF block count must be smaller than LF block count, but got m={m}, n={n}."
        )

    rng = np.random.default_rng(random_seed)
    lf_theta_df = _build_theta_table(
        theta_config=theta_config,
        df_for_bounds=df_for_bounds,
        rng=rng,
        random_seed=random_seed,
        num_thetas=n,
    )

    hf_indices = np.sort(rng.choice(n, size=m, replace=False))
    hf_theta_df = lf_theta_df.iloc[hf_indices].reset_index(drop=True)

    training_theta_keys = {
        (round(float(row.R_max), 6), round(float(row.Z_max), 6))
        for row in lf_theta_df.itertuples(index=False)
    }
    validation_theta_df = _build_theta_table(
        theta_config=theta_config,
        df_for_bounds=df_for_bounds,
        rng=rng,
        random_seed=random_seed,
        num_thetas=k,
        excluded_theta_keys=training_theta_keys,
    )
    validation_theta_df["validation_theta_instance"] = np.arange(len(validation_theta_df))

    return lf_theta_df.reset_index(drop=True), hf_theta_df, validation_theta_df


def _format_float_for_filename(value: float, decimals: int = 3) -> str:
    text = f"{value:.{decimals}f}"
    return text.replace("-", "m").replace(".", "p")


def _build_theta_filename(
    fidelity: str,
    theta: ThetaRegion,
    *,
    extension: str,
    existing_names: set[str],
) -> str:
    base = (
        f"{fidelity}_"
        f"R{_format_float_for_filename(theta.R_max)}_"
        f"Z{_format_float_for_filename(theta.Z_max)}"
    )
    candidate = f"{base}.{extension}"
    duplicate_index = 2
    while candidate in existing_names:
        candidate = f"{base}__dup{duplicate_index}.{extension}"
        duplicate_index += 1
    existing_names.add(candidate)
    return candidate


def write_theta_block_files(
    *,
    blocks: BlockSet,
    theta_df: pd.DataFrame,
    split_name: str,
    fidelity: str,
    output_dir: Path,
    output_format: str,
    progress: bool,
) -> pd.DataFrame:
    """Write one file per block with attached centered theta metadata and inside_theta."""

    if len(blocks.blocks) != len(theta_df):
        raise ValueError(
            f"Number of blocks and theta rows must match for {split_name}: "
            f"{len(blocks.blocks)} != {len(theta_df)}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    existing_names: set[str] = set()
    records: list[dict[str, float | int | bool | str]] = []

    iterator = zip(blocks.blocks, theta_df.itertuples(index=False), strict=True)
    if progress:
        iterator = tqdm(iterator, total=len(blocks.blocks), desc=f"Writing {split_name} block files")

    extension = output_format.lower()
    for block_index, (block_df, row) in enumerate(iterator):
        theta = ThetaRegion(Z_max=float(row.Z_max), R_max=float(row.R_max))
        out_df = block_df.copy()
        out_df["split_name"] = split_name
        out_df["fidelity"] = fidelity
        out_df["R_max"] = theta.R_max
        out_df["Z_max"] = theta.Z_max
        out_df["inside_theta"] = theta_mask(out_df, theta).astype(np.int8)

        filename = _build_theta_filename(
            fidelity=fidelity,
            theta=theta,
            extension=extension,
            existing_names=existing_names,
        )
        written_path = save_dataframe(out_df, output_dir / Path(filename).stem, output_format)

        inside_count = int(out_df["inside_theta"].sum())
        records.append(
            {
                "block_index": int(block_index),
                "random_seed_used": int(row.random_seed_used),
                "split_name": split_name,
                "fidelity": fidelity,
                "file_name": written_path.name,
                "file_path": str(written_path),
                "sample_size": int(len(out_df)),
                "R_max": theta.R_max,
                "Z_max": theta.Z_max,
                "inside_theta_count": inside_count,
                "inside_theta_fraction": float(inside_count / len(out_df)),
            }
        )

    return pd.DataFrame.from_records(records)


def describe_target_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Return descriptive statistics for the selected columns."""

    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Target columns not found in dataframe: {missing}")
    return df[columns].describe()
