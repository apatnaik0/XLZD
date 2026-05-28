#!/usr/bin/env python3
"""Run a trained equal-volume shell CNP model on a raw/source-point CSV.

This is a lightweight inference utility. It does not rebuild the shell dataset
and it does not retrain the model. It uses empirical H5 training files as the
CNP context set, then evaluates new CSV rows as target/query points.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
RUN_CNP_ROOT = SRC_ROOT / "run_cnp"
for path in [REPO_ROOT, SRC_ROOT, RUN_CNP_ROOT]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from cnp_clean_pipeline import H5EventPool, load_model_checkpoint, load_runtime_config, set_seed  # noqa: E402


DEFAULT_CONFIG = Path("xlzd_equal_volume_shell_theta/settings_equal_volume_shell_minibatch.yaml")
DEFAULT_Z_CENTER = 1982.48


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True, help="CSV containing source/event-wise points.")
    parser.add_argument("--output-csv", type=Path, required=True, help="Output CSV with event-wise CNP predictions.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="CNP settings YAML used for training.")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Trained CNP checkpoint. Defaults to path_out_cnp/cnp_<version>_model_<epochs>epochs.pth.",
    )
    parser.add_argument(
        "--theta",
        type=float,
        nargs="+",
        default=None,
        help="One theta query, in the same order as config theta_headers, e.g. --theta 750 1000.",
    )
    parser.add_argument(
        "--theta-csv",
        type=Path,
        default=None,
        help="Optional CSV of theta queries. Must contain the config theta_headers columns.",
    )
    parser.add_argument(
        "--context-dir",
        type=Path,
        default=None,
        help="H5 directory used as CNP context. Defaults to path_to_files_train from the config.",
    )
    parser.add_argument("--context-size", type=int, default=20000, help="Number of empirical context rows to sample.")
    parser.add_argument("--context-files", type=int, default=None, help="Number of H5 files to sample context from.")
    parser.add_argument("--chunk-size", type=int, default=50000, help="CSV target rows predicted per model chunk.")
    parser.add_argument("--mc-samples", type=int, default=30, help="MC dropout samples for prediction uncertainty.")
    parser.add_argument("--z-center", type=float, default=DEFAULT_Z_CENTER, help="TPC center used to derive s_z_from_center.")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional limit for quick test runs.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for context sampling.")
    parser.add_argument("--device", type=str, default=None, help="Force device, e.g. cpu or cuda.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output CSV if it exists.")
    return parser.parse_args()


def resolve_model_path(runtime, model_path: Path | None) -> Path:
    if model_path is not None:
        return model_path.expanduser().resolve()
    return runtime.out_dir / f"cnp_{runtime.version}_model_{runtime.epochs}epochs.pth"


def read_theta_queries(args: argparse.Namespace, theta_headers: list[str]) -> pd.DataFrame:
    if args.theta is not None and args.theta_csv is not None:
        raise ValueError("Use either --theta or --theta-csv, not both.")
    if args.theta is None and args.theta_csv is None:
        raise ValueError(
            "Provide a theta query with --theta or --theta-csv. "
            f"Expected theta headers: {theta_headers}"
        )

    if args.theta is not None:
        if len(args.theta) != len(theta_headers):
            raise ValueError(f"--theta needs {len(theta_headers)} values: {theta_headers}")
        return pd.DataFrame([dict(zip(theta_headers, args.theta, strict=True))])

    theta_df = pd.read_csv(args.theta_csv)
    missing = [col for col in theta_headers if col not in theta_df.columns]
    if missing:
        raise ValueError(f"{args.theta_csv} is missing theta columns: {missing}")
    if theta_df.empty:
        raise ValueError(f"{args.theta_csv} has no theta rows.")
    return theta_df[theta_headers].copy()


def first_existing_column(df: pd.DataFrame, names: Iterable[str]) -> str | None:
    for name in names:
        if name in df.columns:
            return name
    return None


def derive_phi_table(df: pd.DataFrame, phi_headers: list[str], z_center: float) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    for name in phi_headers:
        if name in df.columns:
            out[name] = df[name].astype(float)
            continue

        if name == "s_r":
            direct = first_existing_column(df, ["r_start", "source_r", "r"])
            if direct is not None:
                out[name] = df[direct].astype(float)
                continue
            sx = first_existing_column(df, ["sx", "s_x", "source_x", "x"])
            sy = first_existing_column(df, ["sy", "s_y", "source_y", "y"])
            if sx is not None and sy is not None:
                out[name] = np.sqrt(df[sx].astype(float) ** 2 + df[sy].astype(float) ** 2)
                continue

        if name == "s_z_from_center":
            direct = first_existing_column(df, ["z_start_from_center", "source_z_from_center", "z_from_center"])
            if direct is not None:
                out[name] = df[direct].astype(float).abs()
                continue
            sz = first_existing_column(df, ["sz", "s_z", "source_z", "z_start", "z"])
            if sz is not None:
                out[name] = (df[sz].astype(float) - float(z_center)).abs()
                continue

        raise ValueError(
            f"Could not derive phi column {name!r}. "
            "Provide it directly or include compatible source coordinate columns."
        )

    return out.astype(np.float32)


def sample_context(runtime, context_dir: Path, context_size: int, context_files: int | None, seed: int, device: torch.device):
    pool = H5EventPool(
        context_dir,
        theta_headers=runtime.theta_headers,
        phi_headers=runtime.phi_headers,
        target_headers=runtime.target_headers,
        seed=seed,
        cache_files=True,
    )
    files_per_batch = context_files if context_files is not None else runtime.files_per_batch_train
    batch = pool.sample_batch(batch_size=context_size, files_per_batch=files_per_batch)
    return batch.x.to(device), batch.y.to(device), len(pool.files)


def write_prediction_chunks(
    *,
    model,
    device: torch.device,
    source_df: pd.DataFrame,
    phi_df: pd.DataFrame,
    theta_df: pd.DataFrame,
    theta_headers: list[str],
    phi_headers: list[str],
    target_headers: list[str],
    context_x: torch.Tensor,
    context_y: torch.Tensor,
    output_csv: Path,
    chunk_size: int,
    mc_samples: int,
) -> int:
    header_written = False
    total_rows = 0
    y_raw_col = target_headers[0] if len(target_headers) == 1 and target_headers[0] in source_df.columns else None

    with torch.no_grad():
        for theta_row_idx, theta_row in theta_df.reset_index(drop=True).iterrows():
            theta_values = theta_row[theta_headers].to_numpy(dtype=np.float32)
            for start in range(0, len(phi_df), chunk_size):
                end = min(start + chunk_size, len(phi_df))
                phi_chunk = phi_df.iloc[start:end].to_numpy(dtype=np.float32)
                theta_chunk = np.repeat(theta_values.reshape(1, -1), repeats=len(phi_chunk), axis=0)
                target_x_np = np.hstack([theta_chunk, phi_chunk]).astype(np.float32)
                target_x = torch.from_numpy(target_x_np).to(device)

                mu, std = model.predict_proba_mc(context_x, context_y, target_x, mc_samples=mc_samples)
                mu_np = mu.cpu().numpy()
                std_np = std.cpu().numpy()

                out = pd.DataFrame(
                    {
                        "theta_query_index": np.full(len(phi_chunk), theta_row_idx, dtype=np.int32),
                        "event_index": np.arange(start, end, dtype=np.int64),
                    }
                )
                for i, name in enumerate(theta_headers):
                    out[name] = theta_values[i]
                for name in phi_headers:
                    out[name] = phi_df.iloc[start:end][name].to_numpy()

                if y_raw_col is not None:
                    out["y_raw"] = source_df.iloc[start:end][y_raw_col].to_numpy()

                if len(target_headers) == 1:
                    out["y_cnp"] = mu_np[:, 0]
                    out["y_cnp_err"] = std_np[:, 0]
                else:
                    for i, name in enumerate(target_headers):
                        out[f"y_cnp_{name}"] = mu_np[:, i]
                        out[f"y_cnp_err_{name}"] = std_np[:, i]

                out.to_csv(output_csv, mode="a", index=False, header=not header_written)
                header_written = True
                total_rows += len(out)
                print(
                    f"theta {theta_row_idx + 1}/{len(theta_df)} | "
                    f"rows {start:,}-{end:,} | total_written={total_rows:,}",
                    flush=True,
                )

    return total_rows


def main() -> None:
    args = parse_args()
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive.")
    if args.context_size <= 1:
        raise ValueError("--context-size must be greater than 1.")

    config_path = args.config.expanduser().resolve()
    runtime = load_runtime_config(config_path, seed=args.seed)
    model_path = resolve_model_path(runtime, args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    output_csv = args.output_csv.expanduser().resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if output_csv.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {output_csv}. Use --overwrite to replace it.")
    if output_csv.exists():
        output_csv.unlink()

    source_df = pd.read_csv(args.input_csv)
    if args.max_rows is not None:
        source_df = source_df.head(int(args.max_rows)).copy()
    if source_df.empty:
        raise ValueError(f"No rows found in input CSV: {args.input_csv}")

    theta_df = read_theta_queries(args, runtime.theta_headers)
    phi_df = derive_phi_table(source_df, runtime.phi_headers, z_center=args.z_center)

    set_seed(args.seed)
    model = load_model_checkpoint(model_path, device=args.device)
    device = next(model.parameters()).device

    expected_x_dim = len(runtime.theta_headers) + len(runtime.phi_headers)
    if int(model.x_dim) != expected_x_dim:
        raise ValueError(
            f"Model x_dim={model.x_dim}, but config gives {expected_x_dim} "
            f"({runtime.theta_headers} + {runtime.phi_headers})."
        )

    context_dir = args.context_dir.expanduser().resolve() if args.context_dir else runtime.train_dir
    context_x, context_y, n_context_files_available = sample_context(
        runtime,
        context_dir=context_dir,
        context_size=int(args.context_size),
        context_files=args.context_files,
        seed=int(args.seed),
        device=device,
    )

    print("=== CNP CSV inference ===")
    print(f"input_csv: {args.input_csv}")
    print(f"output_csv: {output_csv}")
    print(f"config: {config_path}")
    print(f"model: {model_path}")
    print(f"context_dir: {context_dir}")
    print(f"context_rows: {len(context_x):,}")
    print(f"context_files_available: {n_context_files_available}")
    print(f"source_rows: {len(source_df):,}")
    print(f"theta_queries: {len(theta_df):,}")
    print(f"theta_headers: {runtime.theta_headers}")
    print(f"phi_headers: {runtime.phi_headers}")

    total = write_prediction_chunks(
        model=model,
        device=device,
        source_df=source_df,
        phi_df=phi_df,
        theta_df=theta_df,
        theta_headers=runtime.theta_headers,
        phi_headers=runtime.phi_headers,
        target_headers=runtime.target_headers,
        context_x=context_x,
        context_y=context_y,
        output_csv=output_csv,
        chunk_size=int(args.chunk_size),
        mc_samples=int(args.mc_samples),
    )
    print(f"Done. prediction_rows_written={total:,}")


if __name__ == "__main__":
    main()
