#!/usr/bin/env python3
"""Convert XLZD per-theta CSV/parquet files into HDF5 files for the CNP pipeline.

Expected input layout:
- outputs/training/lf
- outputs/training/hf
- outputs/validation/hf

Each source file must contain:
- theta columns: R_max, Z_max
- phi columns: s_r, s_z_from_center
- target column: inside_theta

The converter writes one `.h5` file next to each source table.
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, List, Tuple

import h5py
import numpy as np
import pandas as pd

THETA_COLS = ["R_max", "Z_max"]
PHI_COLS = ["s_r", "s_z_from_center"]
TARGET_COLS = ["inside_theta"]
WEIGHTS_COL = "weights"
INPUT_SUFFIXES = (".csv", ".parquet")


def to_bytes_array(items: Iterable[str]) -> np.ndarray:
    return np.array(list(items), dtype="S")


def read_table_checked(table_path: Path) -> pd.DataFrame:
    suffix = table_path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(table_path)
    elif suffix == ".parquet":
        df = pd.read_parquet(table_path)
    else:
        raise ValueError(f"Unsupported input format: {table_path.suffix}")

    required = THETA_COLS + PHI_COLS + TARGET_COLS
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{table_path.name}: missing required columns {missing}")
    if df.empty:
        raise ValueError(f"{table_path.name}: table is empty")
    return df


def build_arrays(df: pd.DataFrame, fidelity_value: float) -> Tuple[np.ndarray, ...]:
    theta_values = df[THETA_COLS].iloc[0].to_numpy(dtype=np.float64)
    phi_values = df[PHI_COLS].to_numpy(dtype=np.float32)
    target_values = df[TARGET_COLS].to_numpy(dtype=np.int8)

    if WEIGHTS_COL in df.columns:
        weights_values = df[[WEIGHTS_COL]].to_numpy(dtype=np.float32)
    else:
        weights_values = np.ones((len(df), 1), dtype=np.float32)

    fidelity_values = np.full((len(df), 1), float(fidelity_value), dtype=np.float32)
    return theta_values, phi_values, target_values, weights_values, fidelity_values


def write_h5(
    h5_path: Path,
    theta: np.ndarray,
    phi: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    fidelity: np.ndarray,
) -> None:
    with h5py.File(h5_path, "w") as handle:
        handle.create_dataset("theta", data=theta, compression="gzip")
        handle.create_dataset("theta_headers", data=to_bytes_array(THETA_COLS), compression="gzip")

        handle.create_dataset("phi", data=phi, compression="gzip")
        handle.create_dataset("phi_labels", data=to_bytes_array(PHI_COLS), compression="gzip")

        handle.create_dataset("target", data=target, compression="gzip")
        handle.create_dataset("target_headers", data=to_bytes_array(TARGET_COLS), compression="gzip")

        handle.create_dataset("weights", data=weights, compression="gzip")
        handle.create_dataset("weights_labels", data=to_bytes_array([WEIGHTS_COL]), compression="gzip")

        handle.create_dataset("fidelity", data=fidelity, compression="gzip")


def convert_one(table_path: Path, h5_path: Path, fidelity_value: float, force: bool) -> str:
    if h5_path.exists() and not force and h5_path.stat().st_mtime >= table_path.stat().st_mtime:
        return f"skip {table_path.name} (up-to-date)"

    df = read_table_checked(table_path)
    theta, phi, target, weights, fidelity = build_arrays(df, fidelity_value)
    write_h5(h5_path, theta, phi, target, weights, fidelity)
    return f"ok   {table_path.name} -> {h5_path.name}"


def discover_tables(directory: Path) -> List[Path]:
    tables: List[Path] = []
    for suffix in INPUT_SUFFIXES:
        tables.extend(sorted(p for p in directory.glob(f"*{suffix}") if p.is_file()))
    deduped = sorted(set(tables))
    return deduped


def convert_directory(directory: Path, fidelity_value: float, workers: int, force: bool) -> Tuple[int, int]:
    if not directory.exists():
        print(f"[warn] missing directory: {directory}")
        return 0, 0

    tables = discover_tables(directory)
    if not tables:
        print(f"[warn] no CSV/parquet files found: {directory}")
        return 0, 0

    print(f"[{directory}] converting {len(tables)} files (fidelity={fidelity_value})")

    success = 0
    failed = 0

    if workers <= 1:
        for idx, table_path in enumerate(tables, start=1):
            try:
                message = convert_one(table_path, table_path.with_suffix(".h5"), fidelity_value, force)
                success += 1
                if idx <= 5 or idx % 100 == 0:
                    print(f"  {message}")
            except Exception as exc:  # pragma: no cover - surfaced to CLI
                failed += 1
                print(f"  fail {table_path.name}: {exc}")
        return success, failed

    try:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(convert_one, table_path, table_path.with_suffix(".h5"), fidelity_value, force): table_path
                for table_path in tables
            }
            for idx, future in enumerate(as_completed(futures), start=1):
                table_path = futures[future]
                try:
                    message = future.result()
                    success += 1
                    if idx <= 5 or idx % 100 == 0:
                        print(f"  {message}")
                except Exception as exc:  # pragma: no cover - surfaced to CLI
                    failed += 1
                    print(f"  fail {table_path.name}: {exc}")
    except PermissionError:
        print("  [warn] multiprocessing unavailable on this system; retrying serial conversion")
        return convert_directory(directory, fidelity_value=fidelity_value, workers=1, force=force)

    return success, failed


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Convert XLZD per-theta CSV/parquet files to HDF5")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=repo_root / "outputs",
        help="Root containing training/ and validation/ folders",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, os.cpu_count() or 1),
        help="Number of worker processes",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild H5 even when output appears up-to-date",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.dataset_root.resolve()

    layout: List[Tuple[Path, float]] = [
        (root / "training" / "lf", 0.0),
        (root / "training" / "hf", 1.0),
        (root / "validation" / "hf", 1.0),
    ]

    print("=" * 72)
    print("CSV/PARQUET -> HDF5 conversion (XLZD)")
    print(f"dataset_root: {root}")
    print(f"workers     : {args.workers}")
    print(f"force       : {args.force}")
    print("=" * 72)

    total_success = 0
    total_failed = 0
    for folder, fidelity in layout:
        ok, fail = convert_directory(folder, fidelity, workers=args.workers, force=args.force)
        total_success += ok
        total_failed += fail

    print("=" * 72)
    print(f"finished: success={total_success}, failed={total_failed}")
    if total_failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
