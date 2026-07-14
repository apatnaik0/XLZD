# XLZD

Code and experiment workflows for CNP and MF-GP modeling on XLZD event data.

This repository contains:

- the standard RESuM-style LF/HF preprocessing pipeline
- CNP training and prediction workflows
- MF-GP fitting and visualization workflows
- alternative theta experiments
- depth-penetration modeling experiments and interactive visualizations

For a detailed explanation of the code structure, experiment logic, and why each branch exists, see:

- [`PROJECT_EXPERIMENT_GUIDE.md`](PROJECT_EXPERIMENT_GUIDE.md)

## Repository Structure

### Core pipeline

- [`prepare_resum_data.py`](prepare_resum_data.py)
  - builds the standard LF/HF/validation block files for the cumulative-theta workflow
- [`convert_csv_to_h5_xlzd.py`](convert_csv_to_h5_xlzd.py)
  - converts generated CSV/parquet blocks to H5 for the CNP
- [`xlzd_resum/`](xlzd_resum)
  - shared preprocessing, theta, config, and dataset utilities
- [`src/xlzd/`](src/xlzd)
  - standard settings files for CNP and MF-GP experiments
- [`src/run_cnp/`](src/run_cnp)
  - CNP training, prediction, and notebook workflows
- [`src/run_mfgp/`](src/run_mfgp)
  - MF-GP fitting, prediction, and visualization workflows

### Experiment folders

- [`depth_penetration_experiments/`](depth_penetration_experiments)
  - event penetration modeling, MLP/VBLL baselines, and Plotly explorers
- [`lf_augmentations/`](lf_augmentations)
  - experiments that add synthetic LF trials at existing theta values
- [`theta_augmentations/`](theta_augmentations)
  - experiments that add LF support at new synthetic theta values
- [`xlzd_shell_theta/`](xlzd_shell_theta)
  - a separate shell-based theta experiment where theta is local rather than cumulative

## Main Workflows

### 1. Standard cumulative-theta workflow

Use this for the original theta definition:

- `theta = (R_max, Z_max)`
- target = whether an event falls inside that cumulative centered region

Run order:

```bash
python3 prepare_resum_data.py
python3 convert_csv_to_h5_xlzd.py
```

Then run:

- [`src/run_cnp/cnp_xlzd_workflow.ipynb`](src/run_cnp/cnp_xlzd_workflow.ipynb)
- [`src/run_mfgp/mfgp_xlzd_workflow.ipynb`](src/run_mfgp/mfgp_xlzd_workflow.ipynb)

Inside the main CNP notebook, set `EXPERIMENT` to one of:

- `default`
- `minibatch`
- `fixedcontext`
- `fullpass`

### 2. Shell-theta workflow

Use this for the alternative shell-based theta definition:

- `theta = (r_shell, z_shell)`
- target = whether an event falls near a local shell-centered region

Run order:

```bash
python3 xlzd_shell_theta/prepare_shell_theta_data.py
python3 xlzd_shell_theta/convert_shell_theta_to_h5.py
```

Then run:

- [`xlzd_shell_theta/00_shell_theta_cnp_workflow.ipynb`](xlzd_shell_theta/00_shell_theta_cnp_workflow.ipynb)
- [`xlzd_shell_theta/01_shell_theta_mfgp.ipynb`](xlzd_shell_theta/01_shell_theta_mfgp.ipynb)

This shell-theta workflow is intentionally separate from the standard cumulative-theta workflow.

## CNP Workflows

Main notebook:

- [`src/run_cnp/cnp_xlzd_workflow.ipynb`](src/run_cnp/cnp_xlzd_workflow.ipynb)

This is the primary notebook for standard CNP experiments.

Archived fixed-path notebooks are kept under:

- [`src/run_cnp/additional_experiments/`](src/run_cnp/additional_experiments)

These are retained for reference only. The main switchable notebook should be the default entry point.

Important CNP scripts:

- [`src/run_cnp/cnp_clean_pipeline.py`](src/run_cnp/cnp_clean_pipeline.py)
  - train/predict pipeline
- [`src/run_cnp/cnp_predict_per_signal.py`](src/run_cnp/cnp_predict_per_signal.py)
  - per-event/per-signal prediction export
- [`src/run_cnp/cnp_validation_prediction.py`](src/run_cnp/cnp_validation_prediction.py)
  - validation prediction helper
- [`src/run_cnp/preprocess_mixup_xlzd.py`](src/run_cnp/preprocess_mixup_xlzd.py)
  - optional mixup preprocessing

## MF-GP Workflows

Main notebook:

- [`src/run_mfgp/mfgp_xlzd_workflow.ipynb`](src/run_mfgp/mfgp_xlzd_workflow.ipynb)

Main script:

- [`src/run_mfgp/mfgp_clean_pipeline.py`](src/run_mfgp/mfgp_clean_pipeline.py)

This workflow consumes aggregated CNP CSV outputs and produces:

- MF-GP fits
- mean/std surfaces
- parity plots
- validation plots
- 3D HTML views where enabled

## Depth Penetration Experiments

Folder:

- [`depth_penetration_experiments/`](depth_penetration_experiments)

This contains a separate modeling track for event penetration into the detector, including:

- global `d_center` prediction
- grouped axial/radial prediction
- deterministic MLP and VBLL-head models
- interactive 2D and 3D Plotly explorers

Start with:

- [`depth_penetration_experiments/README.md`](depth_penetration_experiments/README.md)

## LF and Theta Augmentation Experiments

These folders contain exploratory work and are not part of the main pipeline:

- [`lf_augmentations/`](lf_augmentations)
- [`theta_augmentations/`](theta_augmentations)

They are useful when testing whether MF-GP behavior improves with:

- more LF support at existing theta values
- synthetic LF support at nearby or midpoint theta values

## Outputs

Generated artifacts are typically written under:

- `outputs/`
- `outputs_shell_theta/`
- `data/out/cnp/`
- `data/out/mfgp/`
- experiment-specific `artifacts/` folders

These outputs are generally not intended to be committed unless explicitly needed.

## Environment

Install dependencies with:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Suggested Entry Points

If you are new to the repo, start here:

1. [`prepare_resum_data.py`](prepare_resum_data.py)
2. [`convert_csv_to_h5_xlzd.py`](convert_csv_to_h5_xlzd.py)
3. [`src/run_cnp/cnp_xlzd_workflow.ipynb`](src/run_cnp/cnp_xlzd_workflow.ipynb)
4. [`src/run_mfgp/mfgp_xlzd_workflow.ipynb`](src/run_mfgp/mfgp_xlzd_workflow.ipynb)

If you are working on shell theta, start here:

1. [`xlzd_shell_theta/prepare_shell_theta_data.py`](xlzd_shell_theta/prepare_shell_theta_data.py)
2. [`xlzd_shell_theta/convert_shell_theta_to_h5.py`](xlzd_shell_theta/convert_shell_theta_to_h5.py)
3. [`xlzd_shell_theta/00_shell_theta_cnp_workflow.ipynb`](xlzd_shell_theta/00_shell_theta_cnp_workflow.ipynb)
4. [`xlzd_shell_theta/01_shell_theta_mfgp.ipynb`](xlzd_shell_theta/01_shell_theta_mfgp.ipynb)

## Notes

- The standard cumulative-theta workflow and the shell-theta workflow are different experiments and should not be mixed.
- The notebooks under `src/run_cnp/additional_experiments/` are archived variants, not the main entry points.
- The LF augmentation and theta augmentation folders are exploratory branches rather than the default pipeline.
