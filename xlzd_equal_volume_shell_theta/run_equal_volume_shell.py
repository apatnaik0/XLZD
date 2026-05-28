#!/usr/bin/env python3
"""Run equal-volume shell-theta preprocessing.

By default this runs both:
- dataset preparation
- H5 conversion
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "xlzd_equal_volume_shell_theta" / "config" / "pipeline_config.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to equal-volume shell JSON config file.",
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


def load_dataset_root(config_path: Path) -> Path:
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
    config_path = args.config.resolve()
    dataset_root = load_dataset_root(config_path)

    print(f"Repo root   : {REPO_ROOT}")
    print(f"Config path : {config_path}")
    print(f"Dataset root: {dataset_root}")
    print(f"Stage       : {args.stage}")

    if args.stage in {"prepare", "prepare_convert"}:
        run_command(
            [
                sys.executable,
                "xlzd_equal_volume_shell_theta/prepare_equal_volume_shell_data.py",
                "--config",
                str(config_path.relative_to(REPO_ROOT)),
            ]
        )

    if args.stage in {"convert", "prepare_convert"}:
        cmd = [
            sys.executable,
            "xlzd_equal_volume_shell_theta/convert_equal_volume_shell_to_h5.py",
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
