# LF Augmentations

This folder contains experiments for expanding the LF side of the RESuM pipeline without changing the HF rows.

Contents:
- `lf_augmentation.py`
  - helper functions to:
    - load per-signal CNP predictions
    - build original LF file-level trials
    - synthesize LF trials by bootstrap or merged-block aggregation
    - write augmented CNP-style CSVs for MF-GP
    - write MF-GP config variants that keep outputs separate
- `00_lf_augmentation_workflow.ipynb`
  - visual notebook for inspecting how the LF support changes under augmentation
- `01_lf_augmentation_mfgp.ipynb`
  - notebook for rerunning MF-GP on baseline vs augmented LF training tables

Expected remote inputs:
- `data/out/cnp/cnp_<version>_output_per_signal_<epochs>epochs.csv`
- `data/out/cnp/cnp_<version>_output_<epochs>epochs.csv`
- `data/out/cnp/cnp_<version>_output_validation_<epochs>epochs.csv`

Generated outputs go under:
- `lf_augmentations/artifacts/`
