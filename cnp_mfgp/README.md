# CNP–MF-GP Pipeline - Updated July 14th, 2026

This folder contains the active XLZD event-shell modeling workflow.

The pipeline combines:

1. **Geometry-aware data preparation**
2. **Event-level Conditional Neural Process classification**
3. **Aggregation of predicted shell probabilities**
4. **Two-level autoregressive multi-fidelity Gaussian-process modeling**
5. **Optional emulation and visualization**

The intended use is to learn event distributions across multiple cylindrical detector geometries while combining inexpensive low-fidelity simulations with more expensive high-fidelity simulations.

---

## Folder layout

```text
cnp_mfgp/
├── additional_experiments/
├── config/
├── examples/
├── notebooks/
├── outputs/
├── __init__.py
├── cnp_clean_pipeline.py
├── emulation.py
├── mfgp_clean_pipeline.py
├── prepare_cnp_mfgp_data.py
├── README.md
└── visualize.py
```

### `additional_experiments/`

Experimental variations that are still relevant to the current codebase but are not part of the default run path.

Examples may include:

- alternate training settings;
- loss or weighting tests;
- alternate feature definitions;
- target-transform comparisons;
- augmentation studies;
- one-off validation analyses.

Stable functionality should eventually move into the main scripts or `common/`. Retired experiments should move to the repository-level `archive/`.

### `config/`

Active JSON and YAML configuration files used by the pipeline.

Typical files include:

```text
pipeline_config.json
settings_shell_minibatch.yaml
settings_shell_validation.yaml
```

The preprocessing script reads JSON. The CNP and MF-GP scripts read YAML.

### `examples/`

Small runnable examples and reference scripts.

Use this folder for minimal demonstrations of:

- preparing a dataset;
- loading a trained model;
- running prediction;
- generating emulated source points;
- calling visualization utilities.

### `notebooks/`

Interactive workflows for configuring, running, inspecting, and comparing experiments.

Notebooks should call the pipeline functions rather than duplicating their implementations.

### `outputs/`

Generated model artifacts and diagnostics.

A useful organization is:

```text
outputs/
├── cnp/
├── mfgp/
├── emulation/
└── figures/
```

Large generated files should normally not be committed.

### `__init__.py`

Marks `cnp_mfgp` as an importable Python package.

### `prepare_cnp_mfgp_data.py`

Converts raw event files into event-level categorical HDF5 blocks for CNP training and prediction.

### `cnp_clean_pipeline.py`

Defines the HDF5 event loader, deterministic CNP, training loop, checkpoint format, MC-dropout prediction, and CSV exports.

### `mfgp_clean_pipeline.py`

Fits the two-level autoregressive MF-GP to aggregated CNP outputs and creates geometry-level predictions and diagnostics.

### `emulation.py`

Generates synthetic source positions, converts them to the CNP HDF5 format, and runs a trained CNP without treating dummy labels as physical truth.

### `visualize.py`

Reusable plotting functions for raw event positions, shell occupancy, and CNP prediction distributions.

---

# Pipeline overview

```text
Raw event CSVs + file_manifest.csv
                 │
                 ▼
     prepare_cnp_mfgp_data.py
                 │
                 ├── training/lf/*.h5
                 ├── training/hf/*.h5
                 └── validation/hf/*.h5
                 │
                 ▼
        cnp_clean_pipeline.py
                 │
                 ├── model checkpoint
                 ├── training diagnostics
                 ├── best shell per event
                 └── aggregated shell probabilities
                              │
                              ▼
                 mfgp_clean_pipeline.py
                              │
                              ├── fitted model metadata
                              ├── geometry-grid predictions
                              ├── metrics
                              └── diagnostic plots
```

`emulation.py` and `visualize.py` support the workflow but are not required for standard training.

---

# Model definition

## Detector geometry: theta

The default global detector features are:

```text
theta = [detector_R, detector_Z]
```

`detector_R` is the detector radial extent.

`detector_Z` is the maximum absolute distance from the detector center in z, normally the detector half-height.

The values are read from `file_manifest.csv` for each input file.

## Event/source features: phi

The default event-level source features are:

```text
phi = [s_r, s_z_from_center]
```

where:

```text
s_r = sqrt(sx^2 + sy^2)
s_z_from_center = abs(sz - z_center)
```

## Target

The target is:

```text
target_shell
```

This is a zero-based categorical class:

```text
0, 1, ..., n_shells - 1
```

Exported human-readable shell indices are normally one-based:

```text
1, 2, ..., n_shells
```

---

# Shell geometry

The detector is divided into nested cylindrical shells.

For shell level `i`:

```text
fraction_i = i / n_shells

R_i = R_max * fraction_i^scale_power
Z_i = Z_max * fraction_i^scale_power
```

Shell `i` is inside boundary `i` and outside boundary `i - 1`.

With:

```text
scale_power = 1/3
```

the radius and axial extent scale together so that the enclosed cylindrical volume increases linearly with shell number. The resulting shells therefore have equal nominal volume.

The shell assignment is based on the event endpoint:

```text
r = sqrt(x^2 + y^2)
z_from_center = abs(z - z_center)
```

The CNP inputs use source-position features, while the target shell is determined from the endpoint position.

---

# 1. Prepare the data

## Raw event columns

Input event files are expected to contain:

| Column | Meaning |
|---|---|
| `E0` | Initial event energy |
| `sx`, `sy`, `sz` | Source coordinates |
| `ETPC` | Energy deposited in the TPC |
| `x`, `y`, `z` | Event endpoint coordinates |

The shared loader normalizes several common naming variations. A global event ID is preserved or generated.

## File manifest

Place a manifest inside the configured raw-data directory.

Default name:

```text
file_manifest.csv
```

Required columns:

| Column | Meaning |
|---|---|
| `filename` | Raw event filename, including extension |
| `R` | Detector radius |
| `Z` | Detector centered-z extent, normally half-height |
| `z_center` | Center of the detector in the raw z-coordinate system |
| `fidelity` | Fidelity metadata associated with the source file |

Example:

```csv
filename,R,Z,z_center,fidelity
geometry_A_lf.csv,1490,1983,1983,0
geometry_A_hf.csv,1490,1983,1983,1
geometry_B_lf.csv,1600,2100,2100,0
geometry_B_hf.csv,1600,2100,2100,1
```

`R`, `Z`, and `z_center` should be supplied explicitly. They should not be inferred from event coverage because an individual source component may not span the complete detector.

The manifest fidelity is retained as source metadata. The later LF/HF pool split is performed after events from all manifest files are combined.

## Preprocessing configuration

Example `config/pipeline_config.json`:

```json
{
  "file_load": {
    "input_dir": "data/raw",
    "max_rows_per_file": null
  },
  "split": {
    "lf_pool_fraction": 0.2,
    "hf_pool_fraction": 0.4,
    "random_seed": 42,
    "stratify_by_component": false
  },
  "sampling": {
    "lf_block_size": 20000,
    "hf_block_size": 100000,
    "validation_block_size": null,
    "progress": true
  },
  "shell": {
    "n_shells": 100,
    "min_candidate_events": 1,
    "scale_power": 0.3333333333333333
  },
  "output": {
    "output_dir": "data/processed/cnp_mfgp",
    "output_format": "csv"
  }
}
```

`validation_block_size: null` uses the HF block size.

`max_rows_per_file: null` loads the complete file.

The geometry in each manifest row replaces any global `R_max`, `Z_max`, or `z_center` values in the shell configuration.

## Run preprocessing

From the repository root:

```bash
python cnp_mfgp/prepare_cnp_mfgp_data.py \
  --config cnp_mfgp/config/pipeline_config.json \
  --manifest file_manifest.csv
```

Always pass the configuration path explicitly.

> **Important:** the preprocessing script deletes the complete configured `output_dir` before rebuilding it. Use a dedicated or versioned dataset directory.

## What preprocessing does

The script:

1. Loads every file in the manifest.
2. Adds centered source and endpoint coordinates.
3. Stores each file's detector geometry on every event.
4. Builds the shell table for that geometry.
5. Assigns one zero-based shell target to every valid event.
6. Concatenates labeled events across geometries.
7. Shuffles and creates disjoint LF, HF, and validation pools.
8. Divides each pool into near-equal blocks.
9. Writes categorical HDF5 blocks.
10. Writes pool tables, shell tables, and an HDF5-file manifest.

## Prepared directory structure

```text
<configured output_dir>/
├── training/
│   ├── lf/
│   │   └── lf_block####_event_classes.h5
│   └── hf/
│       └── hf_block####_event_classes.h5
├── validation/
│   └── hf/
│       └── hf_block####_event_classes.h5
├── processed_all_events.csv
├── lf_training_pool.csv
├── hf_training_pool.csv
├── hf_validation_pool.csv
├── shell_table_by_theta.csv
└── event_class_manifest.csv
```

Pool tables use Parquet instead of CSV when `output_format` is set to `parquet`.

## HDF5 contract

Each event block contains:

```text
theta           float32 [N, theta_dim]
phi             float32 [N, phi_dim]
target_shell    int64   [N]
theta_labels
phi_labels
target_headers
meta/
```

Typical metadata includes:

```text
event_index
original_event_id
shell_index
source_file
source_fidelity
split_name
output_fidelity
detector_R
detector_Z
detector_z_center
```

The HDF5 label arrays must match the CNP YAML settings exactly.

---

# 2. Configure the CNP and MF-GP

The CNP and MF-GP read the same YAML file.

Example `config/settings_shell_minibatch.yaml`:

```yaml
simulation_settings:
  simulation_type: shell

  n_shells: 100

  theta_headers:
    - detector_R
    - detector_Z

  phi_labels:
    - s_r
    - s_z_from_center

  target_headers:
    - target_shell

  theta_min:
    - 1000.0
    - 1400.0

  theta_max:
    - 1800.0
    - 2400.0

cnp_settings:
  training_mode: minibatch
  context_mode: random

  training_epochs: 15
  steps_per_epoch: 5000

  context_ratio: 0.3333333333333333
  batch_size_train: 12000
  files_per_batch_train: 32
  ratio_testing_vs_training: 0.1

  plot_after: 1000

path_settings:
  version: shell_v1

  path_to_files_train: ../../data/processed/cnp_mfgp/training/lf

  path_to_files_predict:
    - ../../data/processed/cnp_mfgp/training/lf
    - ../../data/processed/cnp_mfgp/training/hf

  iteration:
    - 0
    - 0

  fidelity:
    - 0
    - 1

  path_out_cnp: ../outputs/cnp
  path_out_mfgp: ../outputs/mfgp
```

Relative paths in the YAML are resolved relative to the YAML file.

The entries in these three lists refer to the same prediction datasets:

```yaml
path_to_files_predict:
iteration:
fidelity:
```

Keep their ordering aligned.

Typical MF-GP fidelity IDs are:

```text
0 = LF
1 = HF
```

---

# 3. Train the CNP

## CNP input and output

For each event:

```text
x = concatenate(theta, phi)
y = target_shell
```

The model returns a vector of shell logits, which becomes a shell-probability distribution after softmax.

The current implementation uses:

- a deterministic CNP encoder and decoder;
- context labels represented as one-hot shell vectors;
- weighted categorical cross entropy;
- class weights derived from observed shell counts;
- gradient clipping;
- random or fixed context size;
- minibatch or full-pass training;
- MC dropout during prediction.

## Training modes

### `minibatch`

Each epoch performs a configured number of random sampled steps:

```yaml
training_mode: minibatch
steps_per_epoch: 5000
```

An epoch does not necessarily visit every event.

### `full_pass`

Each epoch iterates over all events in the configured training directory:

```yaml
training_mode: full_pass
```

This is more directly comparable to a conventional epoch, but it can take substantially longer.

## Context modes

### `random`

The number of context events varies between the minimum allowed value and the configured maximum.

### `fixed`

Every batch uses the configured context count.

## Train from the command line

```bash
python cnp_mfgp/cnp_clean_pipeline.py \
  --config cnp_mfgp/config/settings_shell_minibatch.yaml \
  --device cuda \
  train
```

CPU example:

```bash
python cnp_mfgp/cnp_clean_pipeline.py \
  --config cnp_mfgp/config/settings_shell_minibatch.yaml \
  --device cpu \
  train
```

Common overrides:

```bash
python cnp_mfgp/cnp_clean_pipeline.py \
  --config cnp_mfgp/config/settings_shell_minibatch.yaml \
  --device cuda \
  train \
  --steps-per-epoch 5000 \
  --lr 1e-4 \
  --weight-decay 0 \
  --repr-dim 32 \
  --hidden 128 \
  --dropout 0.1 \
  --monitor-every 1000
```

## Training outputs

The configured CNP output directory receives files such as:

```text
cnp_<version>_model_<epochs>epochs.pth
cnp_<version>_history_<epochs>epochs.csv
cnp_<version>_training_curve_<epochs>epochs.png
cnp_<version>_sample_predictions_<epochs>epochs.png
cnp_<version>_class_monitor_latest.png
```

The history CSV records:

```text
train_loss
val_loss
train_acc
val_acc
train_mae_shell
val_mae_shell
```

The validation values generated during CNP training come from independently sampled batches from the configured training pool. Held-out HF validation prediction is a separate workflow.

---

# 4. Run CNP prediction

Prediction loads a checkpoint and evaluates every configured prediction directory.

MC dropout repeats each forward pass and estimates:

```text
mean shell probability
standard deviation of shell probability
```

## Predict from an existing checkpoint

```bash
python cnp_mfgp/cnp_clean_pipeline.py \
  --config cnp_mfgp/config/settings_shell_minibatch.yaml \
  --device cuda \
  predict \
  --model-path cnp_mfgp/outputs/cnp/cnp_shell_v1_model_15epochs.pth \
  --mc-samples 30 \
  --chunk-size 20000 \
  --output-suffix output
```

## Train and predict in one run

```bash
python cnp_mfgp/cnp_clean_pipeline.py \
  --config cnp_mfgp/config/settings_shell_minibatch.yaml \
  --device cuda \
  full \
  --mc-samples 30 \
  --chunk-size 20000 \
  --output-suffix output
```

## Predict on held-out validation data

Use a validation YAML whose `path_to_files_predict` points to:

```text
<prepared data>/validation/hf
```

Then run:

```bash
python cnp_mfgp/cnp_clean_pipeline.py \
  --config cnp_mfgp/config/settings_shell_validation.yaml \
  --device cuda \
  predict \
  --model-path cnp_mfgp/outputs/cnp/cnp_shell_v1_model_15epochs.pth \
  --output-suffix output_validation
```

Use a distinct suffix or version so the validation files do not overwrite the training prediction files.

## Prediction outputs

### Aggregated MF-GP input

```text
cnp_<version>_<suffix>_<epochs>epochs.csv
```

This table is grouped by:

```text
iteration
fidelity
theta columns
shell_index
```

and contains:

```text
n_samples
y_cnp
y_cnp_err
y_raw
log_prop
bce
source_file
```

Definitions:

- `y_cnp`: mean predicted probability for that shell;
- `y_cnp_err`: root-mean-square MC-dropout standard deviation;
- `y_raw`: observed fraction of events in that shell;
- `shell_index`: one-based shell index.

### Best shell per event

```text
cnp_<version>_<suffix>_<epochs>epochs_best_shell.csv
```

This contains the most probable shell for each event, the corresponding probability and uncertainty, and available event metadata.

### Optional full event-shell table

```text
cnp_<version>_<suffix>_<epochs>epochs_all_shells.csv
```

This contains one row per:

```text
event × shell
```

The default is:

```python
all_shells=False
```

This should remain disabled for routine prediction. With millions of events and 100 shells, the full table can require tens of gigabytes and greatly increase runtime.

The command-line parser does not expose an `--all-shells` flag. Enable it through Python only when the full diagnostic table is needed:

```python
from cnp_mfgp.cnp_clean_pipeline import (
    load_runtime_config,
    predict_cnp,
)

runtime = load_runtime_config(
    "cnp_mfgp/config/settings_shell_minibatch.yaml"
)

result = predict_cnp(
    runtime=runtime,
    model_path="cnp_mfgp/outputs/cnp/cnp_shell_v1_model_15epochs.pth",
    mc_samples=30,
    chunk_size=20000,
    all_shells=True,
)
```

---

# 5. Fit the MF-GP

The MF-GP uses a two-level autoregressive model:

```text
y_HF(theta) = rho * y_LF(theta) + delta(theta)
```

The stages are:

1. Fit an LF Gaussian process.
2. Evaluate the LF model at HF geometries.
3. Estimate `rho`.
4. Fit a discrepancy GP to:

```text
y_HF - rho * y_LF
```

5. Combine LF and discrepancy predictions and uncertainties.

The implementation uses scikit-learn with:

- standardized inputs;
- standardized targets;
- constant × Matérn-3/2 kernels;
- white-noise kernels;
- separate LF and discrepancy GPs.

## MF-GP input columns

The input CSV must contain:

```text
theta headers
iteration
fidelity
y_cnp
y_cnp_err
y_raw
```

The LF model is trained on:

```text
theta -> y_cnp
```

The HF discrepancy model is trained using:

```text
theta -> y_raw
```

## Run the MF-GP

```bash
python cnp_mfgp/mfgp_clean_pipeline.py \
  --config cnp_mfgp/config/settings_shell_minibatch.yaml \
  --cnp-csv cnp_mfgp/outputs/cnp/cnp_shell_v1_output_15epochs.csv \
  --validation-csv cnp_mfgp/outputs/cnp/cnp_shell_v1_output_validation_15epochs.csv \
  --iteration 0 \
  --lf-fidelity 0 \
  --hf-fidelity 1 \
  --target-transform linear
```

Available target transforms:

```text
linear
log_hf
log_lf
log_both
```

Run all four:

```bash
python cnp_mfgp/mfgp_clean_pipeline.py \
  --config cnp_mfgp/config/settings_shell_minibatch.yaml \
  --cnp-csv cnp_mfgp/outputs/cnp/cnp_shell_v1_output_15epochs.csv \
  --validation-csv cnp_mfgp/outputs/cnp/cnp_shell_v1_output_validation_15epochs.csv \
  --all-transforms
```

Other useful options:

```text
--grid-points 120
--predict-chunk-size 20000
--random-state 42
--log-epsilon <value>
--prefer-validation-csv
--quiet
```

## MF-GP outputs

The configured MF-GP directory receives:

```text
mfgp_<version-and-transform>_model_iter<iteration>.json
mfgp_<version-and-transform>_metrics_iter<iteration>.json
mfgp_<version-and-transform>_predictions_iter<iteration>.csv
mfgp_<version-and-transform>_grid_iter<iteration>.csv
```

It also creates available diagnostics such as:

```text
HF observation maps
mean and standard-deviation surfaces
interactive 3D HTML surfaces
parity plots
residual plots
across-theta comparisons
coverage plots
validation uncertainty bands
```

The metrics JSON includes:

```text
RMSE
MAE
R²
rho
sample counts
target-transform metadata
```

## Current shell aggregation behavior

The CNP aggregate CSV preserves `shell_index`.

The current MF-GP loader then groups rows by:

```text
theta headers + fidelity + iteration
```

and does not include `shell_index` in its model input.

Therefore, the current MF-GP fits a geometry-level average over the shell rows rather than a separate response for every shell.

A shell-specific MF-GP would require one of these changes:

- fit one MF-GP per shell;
- include `shell_index` as a model feature;
- preserve `shell_index` in the grouping keys and use a multi-output model.

This behavior should be considered when interpreting the MF-GP output.

---

# Emulation

`emulation.py` generates synthetic source positions and runs them through a trained CNP.

The default generator samples a cylindrical skin outside the detector side wall:

```text
R_max <= s_r <= R_max + width
```

with source z distributed over the detector height.

Top and bottom cap skins can optionally be included.

## Generate emulated source points

```python
from cnp_mfgp.emulation import emulate_detector_points

emulated = emulate_detector_points(
    n_points=100000,
    radius=1490.0,
    height=3966.0,
    width=100.0,
    z_min=0.0,
    include_caps=False,
    seed=42,
)

emulated.to_csv(
    "data/emulation/outside_sidewall.csv",
    index=False,
)
```

The generated table includes:

```text
sx, sy, sz
E0, ETPC
x, y, z
```

The endpoint columns are placeholders used to match the event-file schema.

## Predict on emulated points

```python
from common.config import ShellConfig
from cnp_mfgp.cnp_clean_pipeline import load_runtime_config
from cnp_mfgp.emulation import predict_cnp_from_emulated_csv

runtime = load_runtime_config(
    "cnp_mfgp/config/settings_shell_minibatch.yaml"
)

shell_cfg = ShellConfig(
    R_max=1490.0,
    Z_max=1983.0,
    z_center=1983.0,
    n_shells=100,
    min_candidate_events=1,
    scale_power=1.0 / 3.0,
)

result = predict_cnp_from_emulated_csv(
    emulated_csv="data/emulation/outside_sidewall.csv",
    runtime=runtime,
    model_path="cnp_mfgp/outputs/cnp/cnp_shell_v1_model_15epochs.pth",
    output_dir="cnp_mfgp/outputs/emulation",
    shell_cfg=shell_cfg,
    h5_block_size=100000,
    mc_samples=30,
    chunk_size=20000,
    all_shells=False,
)
```

The helper:

1. derives CNP phi features;
2. creates temporary HDF5 prediction blocks;
3. runs `predict_cnp`;
4. changes truth shell values to `-1`;
5. removes dummy `y_raw` values;
6. empties the aggregate MF-GP CSV by default.

The MF-GP CSV is emptied because dummy shell labels used to satisfy the CNP HDF5 interface are not physical emulation truth.

---

# Visualization

`visualize.py` provides reusable plotting helpers.

Available public functions include:

```text
resolve_input_paths
load_df_with_rt
find_shells
find_shell_occupations
exponential_regression
cylinder
plot_shell_histogram
plot_cnp_pred_shell_occupancy
plot_input_shell_occupancy
plot_input_points_3d
```

## Plot predicted and true shell distributions

```python
from cnp_mfgp.visualize import plot_cnp_pred_shell_occupancy

fig, axes = plot_cnp_pred_shell_occupancy(
    result,
    outpath="cnp_mfgp/outputs/figures/predicted_vs_true_shells.png",
)
```

For emulated data, truth shell indices are `-1`, so only the predicted distribution is meaningful.

## Plot raw input points interactively

```python
from cnp_mfgp.visualize import plot_input_points_3d

plot_input_points_3d(
    inpath="data/raw/*.csv",
    outpath="cnp_mfgp/outputs/figures/input_points.html",
    max_points=50000,
    r_max=1490.0,
    z_min=0.0,
    z_max=3966.0,
    seed=42,
)
```

---

# Recommended notebook workflow

The scripts are the source of truth. Notebooks should orchestrate them.

A notebook run should generally:

1. locate the repository root;
2. define paths and experiment settings;
3. call the preprocessing functions or run the preprocessing script;
4. load the YAML runtime;
5. train or load a CNP checkpoint;
6. predict on LF, HF, and validation datasets;
7. inspect shell-level metrics and plots;
8. fit the MF-GP;
9. compare training and validation geometry trends;
10. save all generated artifacts under `outputs/`.

Avoid copying model, geometry, or HDF5 implementations directly into notebooks.

---

# Output and storage guidance

## Do not save all shells unless required

The all-shell table scales as:

```text
number of events × number of shells
```

For 15 million events and 100 shells, this represents 1.5 billion rows before considering CSV overhead.

Use:

```python
all_shells=False
```

for standard training, validation, and emulation runs.

## Keep dataset builds versioned

Preprocessing deletes its configured output directory. Use distinct directories such as:

```text
data/processed/cnp_mfgp_v1/
data/processed/cnp_mfgp_v2/
```

when previous builds must be retained.

## Separate datasets from model outputs

Recommended locations:

```text
data/                         raw and prepared event data
cnp_mfgp/outputs/cnp/         CNP checkpoints and predictions
cnp_mfgp/outputs/mfgp/        MF-GP artifacts
cnp_mfgp/outputs/emulation/   emulated predictions
cnp_mfgp/outputs/figures/     plots and HTML views
```

---

# Troubleshooting

## Repository root cannot be found

Confirm that the repository root contains:

```text
README.md
PROJECT_EXPERIMENT_GUIDE.md
```

Run commands from that directory.

## `common` cannot be imported

Run from the repository root rather than from inside `cnp_mfgp/`.

Alternatively, install the repository as an editable package once packaging metadata is added.

## Default configuration path does not exist

Pass `--config` explicitly:

```bash
--config cnp_mfgp/config/<settings-file>
```

This is recommended for every script.

## Manifest geometry fails to load

Confirm that every row has numeric values for:

```text
R
Z
z_center
fidelity
```

and that every `filename` exists inside the configured input directory.

## HDF5 label mismatch

The YAML must match the labels stored during preprocessing:

```yaml
theta_headers:
  - detector_R
  - detector_Z

phi_labels:
  - s_r
  - s_z_from_center

target_headers:
  - target_shell
```

Rebuild the HDF5 blocks after changing these definitions.

## CUDA or system memory is exhausted

Reduce:

```yaml
batch_size_train
files_per_batch_train
```

or lower prediction:

```text
--chunk-size
```

## Disk fills during prediction

Confirm that all-shell prediction is disabled.

Also check old dataset builds, output versions, temporary HDF5 files, and the operating-system trash directory.

## MF-GP cannot find both fidelities

Inspect the CNP CSV and verify that it contains rows with the IDs passed as:

```text
--lf-fidelity
--hf-fidelity
```

Also verify the aligned YAML lists:

```yaml
path_to_files_predict
iteration
fidelity
```

## MF-GP auto-discovery finds no CSV

Auto-discovery expects a name matching approximately:

```text
cnp_<version>_output_<epochs>epochs.csv
```

Use:

```text
--output-suffix output
```

during CNP prediction, or pass `--cnp-csv` explicitly.

## Validation output overwrites training output

Use different suffixes:

```text
output
output_validation
```

or different version names.

---

# Command summary

```bash
# Prepare data
python cnp_mfgp/prepare_cnp_mfgp_data.py \
  --config cnp_mfgp/config/pipeline_config.json \
  --manifest file_manifest.csv

# Train CNP
python cnp_mfgp/cnp_clean_pipeline.py \
  --config cnp_mfgp/config/settings_shell_minibatch.yaml \
  --device cuda \
  train

# Predict LF/HF
python cnp_mfgp/cnp_clean_pipeline.py \
  --config cnp_mfgp/config/settings_shell_minibatch.yaml \
  --device cuda \
  predict \
  --model-path cnp_mfgp/outputs/cnp/<checkpoint>.pth \
  --output-suffix output

# Predict held-out validation
python cnp_mfgp/cnp_clean_pipeline.py \
  --config cnp_mfgp/config/settings_shell_validation.yaml \
  --device cuda \
  predict \
  --model-path cnp_mfgp/outputs/cnp/<checkpoint>.pth \
  --output-suffix output_validation

# Fit MF-GP
python cnp_mfgp/mfgp_clean_pipeline.py \
  --config cnp_mfgp/config/settings_shell_minibatch.yaml \
  --cnp-csv cnp_mfgp/outputs/cnp/<training-prediction>.csv \
  --validation-csv cnp_mfgp/outputs/cnp/<validation-prediction>.csv \
  --iteration 0 \
  --lf-fidelity 0 \
  --hf-fidelity 1 \
  --target-transform linear
```

For broader repository organization, see [`../README.md`](../README.md).

For development history and experiment rationale, see [`../PROJECT_EXPERIMENT_GUIDE.md`](../PROJECT_EXPERIMENT_GUIDE.md).
