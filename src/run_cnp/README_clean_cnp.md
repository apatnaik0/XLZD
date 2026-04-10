# Clean CNP Pipeline (XLZD)

Files added:
- `src/run_cnp/cnp_clean_pipeline.py`
- `src/run_cnp/cnp_predict_per_signal.py`
- `src/run_cnp/preprocess_mixup_xlzd.py`
- `src/xlzd/settings.yaml`

## What it does
- Reads `src/xlzd/settings.yaml` paths and headers
- Trains a self-contained deterministic CNP on H5 data
- Saves model + training history CSV + training plots
- Runs prediction on configured folders
- Exports CSV compatible with MFGP usage (`y_cnp`, `y_cnp_err`, `y_raw`, `fidelity`, `iteration`)
- Saves prediction heatmaps

## CLI usage
Run mixup preprocessing first:
```bash
python src/run_cnp/preprocess_mixup_xlzd.py --config src/xlzd/settings.yaml
```

Then train and predict:
```bash
python src/run_cnp/cnp_clean_pipeline.py --config src/xlzd/settings.yaml train --steps-per-epoch 5000 --monitor-every 1000
python src/run_cnp/cnp_clean_pipeline.py --config src/xlzd/settings.yaml predict --model-path data/out/cnp/cnp_xlzd_v1_model_15epochs.pth
```

or end-to-end:
```bash
python src/run_cnp/cnp_clean_pipeline.py --config src/xlzd/settings.yaml full --steps-per-epoch 5000 --monitor-every 1000
```

## Notebook usage
Open `src/run_cnp/cnp_clean_workflow.ipynb` and run cells in order.

## Notes
- This pipeline deliberately avoids importing from `resum`.
- If `use_data_augmentation: mixup` is set in `settings.yaml`, training reads `phi_mixedup` / `target_mixedup` (with fallback to base datasets per-file if mixup arrays are empty).
