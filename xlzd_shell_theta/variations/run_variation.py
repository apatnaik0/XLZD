#!/usr/bin/env python3
"""Run shell-theta variation data-preparation stages.

This script is intentionally limited to the preprocessing side:
- build shell-theta CSV/parquet files
- convert them to H5

The variation notebooks then handle:
- CNP training/prediction
- MF-GP fitting
- plots
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = Path(__file__).resolve().parent / "config"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variation",
        required=True,
        help="Variation slug, for example: method1_larger_delta",
    )
    parser.add_argument(
        "--stage",
        default="prepare_convert",
        choices=["prepare", "convert", "prepare_convert"],
        help="Which preprocessing stage(s) to run.",
    )
    parser.add_argument(
        "--force-h5",
        action="store_true",
        help="Pass --force to the H5 converter.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Optional worker count for H5 conversion.",
    )
    return parser.parse_args()


def variation_config_path(variation: str) -> Path:
    path = CONFIG_DIR / f"{variation}.json"
    if not path.exists():
        raise FileNotFoundError(f"Variation config does not exist: {path}")
    return path


def load_output_dir(config_path: Path) -> Path:
    payload = json.loads(config_path.read_text())
    output_dir = payload.get("output", {}).get("output_dir")
    if not output_dir:
        raise ValueError(f"Missing output.output_dir in {config_path}")
    return REPO_ROOT / output_dir


def run_command(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def main() -> None:
    args = parse_args()
    config_path = variation_config_path(args.variation)
    dataset_root = load_output_dir(config_path)

    print(f"Repo root   : {REPO_ROOT}")
    print(f"Variation   : {args.variation}")
    print(f"Config path : {config_path}")
    print(f"Dataset root: {dataset_root}")
    print(f"Stage       : {args.stage}")

    if args.stage in {"prepare", "prepare_convert"}:
        run_command(
            [
                sys.executable,
                "xlzd_shell_theta/prepare_shell_theta_data.py",
                "--config",
                str(config_path.relative_to(REPO_ROOT)),
            ]
        )

    if args.stage in {"convert", "prepare_convert"}:
        cmd = [
            sys.executable,
            "xlzd_shell_theta/convert_shell_theta_to_h5.py",
            "--dataset-root",
            str(dataset_root.relative_to(REPO_ROOT)),
        ]
        if args.workers is not None:
            cmd.extend(["--workers", str(args.workers)])
        if args.force_h5:
            cmd.append("--force")
        run_command(cmd)


if __name__ == "__main__":
    main()
