# XLZD Shell Theta Experiment

This folder is for a separate experiment where `theta` is no longer a cumulative detector volume `(R_max, Z_max)`.

Instead:

- `theta = (r_shell, z_shell)`
- each theta defines a **local shell-centered neighborhood**
- the target is the **fraction of events whose final position lies near that shell**

This is meant to remove the current inclusivity problem where larger volume thetas always contain smaller ones.

## Goal

Replace the old cumulative target:

- `inside_theta = 1` if `r <= R_max` and `z_from_center <= Z_max`

with a local shell target:

- `near_shell = 1` if the final position is close to `(r_shell, z_shell)`

Then run:

1. shell-theta data construction
2. CNP on shell-theta labels
3. MF-GP on aggregated shell-theta `y_cnp` / `y_raw`

## First recommended shell definition

Use a fixed rectangular proximity region around the shell center:

- `|r - r_shell| <= delta_r`
- `|z_from_center - z_shell| <= delta_z`

This is the cleanest first version because it is:

- easy to implement
- easy to visualize
- comparable across theta points
- compatible with the existing 2D theta framework

## Recommended first target

Use a **fraction**, not a raw count:

- `y_raw = mean(near_shell)`
- `y_cnp = mean(predicted probability of near_shell)`

This keeps the target comparable across files and trials with different sample sizes.

## Folder contents

- `EXPERIMENT_PLAN.md`
  - full experiment plan
- `PARAMETERS.md`
  - concrete recommended first-pass parameters
- `prepare_shell_theta_data.py`
  - builds shell-theta CSV/parquet blocks
- `convert_shell_theta_to_h5.py`
  - converts shell-theta blocks to H5 for the generic CNP pipeline
- `settings_shell_minibatch.yaml`
  - shell-theta CNP/MF-GP training config
- `settings_shell_validation_minibatch.yaml`
  - shell-theta validation-prediction config
- `00_shell_theta_cnp_workflow.ipynb`
  - CNP training and prediction notebook
- `01_shell_theta_mfgp.ipynb`
  - MF-GP notebook with the same transform/plot style as the earlier workflows

## Run order

From the repo root:

1. Build the shell-theta block files:

```bash
python3 xlzd_shell_theta/prepare_shell_theta_data.py
```

2. Convert those files to H5:

```bash
python3 xlzd_shell_theta/convert_shell_theta_to_h5.py
```

3. Run the CNP notebook:

- `xlzd_shell_theta/00_shell_theta_cnp_workflow.ipynb`

This will train the shell-theta minibatch CNP and write:

- `data/out/cnp/cnp_xlzd_shell_v1_minibatch_model_15epochs.pth`
- `data/out/cnp/cnp_xlzd_shell_v1_minibatch_output_15epochs.csv`
- `data/out/cnp/cnp_xlzd_shell_v1_minibatch_output_validation_15epochs.csv`

4. If event-level shell-theta CNP output is needed later, use the existing generic exporter with the shell config:

```bash
python3 src/run_cnp/cnp_predict_per_signal.py \
  --config xlzd_shell_theta/settings_shell_minibatch.yaml \
  --model-path data/out/cnp/cnp_xlzd_shell_v1_minibatch_model_15epochs.pth \
  --out-csv data/out/cnp/cnp_xlzd_shell_v1_minibatch_output_per_signal_15epochs.csv \
  --overwrite
```

5. Run the MF-GP notebook:

- `xlzd_shell_theta/01_shell_theta_mfgp.ipynb`

This uses the shell-theta aggregated CNP CSVs and produces the same `linear`, `log_hf`, `log_lf`, and `log_both` views as the older MF-GP workflow.

## Important design principle

Keep this experiment separate from the current cumulative-volume theta pipeline.

Do not reuse the old `inside_theta` interpretation here. This should be treated as a different target construction entirely.
