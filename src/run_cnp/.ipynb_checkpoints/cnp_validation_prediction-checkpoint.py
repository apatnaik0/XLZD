#!/usr/bin/env python3
"""Run held-out HF validation prediction for the XLZD CNP with default paths.

This script assumes:
- the training model already exists under `data/out/cnp`
- validation H5 files already exist under `outputs/validation/hf`
"""

from __future__ import annotations

import json
from pathlib import Path

from cnp_clean_pipeline import predict_cnp, load_runtime_config


def main() -> None:
    config_path = (Path(__file__).resolve().parents[1] / "xlzd" / "settings_validation.yaml").resolve()
    runtime = load_runtime_config(config_path, seed=42)
    model_path = runtime.out_dir / f"cnp_{runtime.version}_model_{runtime.epochs}epochs.pth"

    result = predict_cnp(
        runtime,
        model_path=model_path,
        mc_samples=30,
        chunk_size=20000,
    )
    print(
        json.dumps(
            {
                "csv_path": str(result.csv_path),
                "heatmap_path": str(result.heatmap_path),
                "error_heatmap_path": str(result.error_heatmap_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
