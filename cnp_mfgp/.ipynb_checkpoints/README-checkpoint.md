# CNP–MF-GP Pipeline

This folder contains the active XLZD event-shell modeling workflow.

The pipeline combines:

1. geometry-aware event preparation;
2. event-level Conditional Neural Process classification;
3. aggregation into one shell-probability distribution per geometry and fidelity;
4. centered-log-ratio PCA of the complete shell distribution;
5. autoregressive multi-fidelity Gaussian processes over the PCA coefficients;
6. reconstruction of valid high-fidelity shell distributions.

The intended use is to learn how complete event-position distributions change across cylindrical detector geometries while combining inexpensive low-fidelity simulations with more expensive high-fidelity simulations.

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

### Primary files

`prepare_cnp_mfgp_data.py`
: Reads `file_manifest.csv`, builds shell labels, preserves the manifest fidelity for every event, splits only high-fidelity events into training and validation data, and writes categorical HDF5 blocks.

`cnp_clean_pipeline.py`
: Defines the event-level CNP, training loop, checkpoint format, MC-dropout prediction, and long-form shell-distribution CSV exports.

`mfgp_clean_pipeline.py`
: Recommended MF-GP implementation. It treats the 100 shell probabilities as one correlated probability distribution, compresses the distribution with centered-log-ratio PCA, fits one autoregressive MF-GP per latent coefficient, and reconstructs normalized shell distributions.

`emulation.py`
: Generates synthetic source positions and passes them through a trained CNP.

`visualize.py`
: Reusable plotting helpers for raw events, shell occupancy, and predicted distributions.

---

# Pipeline overview

```text
Raw event files + file_manifest.csv
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
                 ├── CNP checkpoint
                 ├── training diagnostics
                 ├── best shell per event
                 └── long-form shell probabilities
                              │
                              ▼
          mfgp_clean_pipeline.py
                              │
                              ├── one LF distribution per geometry
                              ├── one HF distribution per geometry
                              ├── CLR transformation
                              ├── PCA coefficients
                              ├── one MF-GP per coefficient
                              └── reconstructed HF distributions
```

The central statistical unit of the MF-GP is now **one detector geometry with one complete shell distribution**, not one shell bin treated as an independent geometry sample.

---

# Model definitions

## Detector geometry: theta

The default detector features are:

```text
theta = [detector_R, detector_Z]
```

- `detector_R` is the detector radial extent.
- `detector_Z` is the maximum absolute centered-z extent, normally the detector half-height.
- Both values come from `file_manifest.csv`.

## Event/source features: phi

The default event-level CNP features are:

```text
phi = [s_r, s_z_from_center]
```

with:

```text
s_r = sqrt(sx^2 + sy^2)
s_z_from_center = abs(sz - z_center)
```

## CNP target

The event-level target is the zero-based shell class:

```text
target_shell = 0, 1, ..., n_shells - 1
```

Human-readable exported shell indices are one-based:

```text
shell_index = 1, 2, ..., n_shells
```

## MF-GP target

For every geometry and fidelity, the CNP output is aggregated into:

```text
p(theta, fidelity) = [p_1, p_2, ..., p_n_shells]
```

where:

```text
p_i >= 0
sum_i p_i = 1
```

For the default 100-shell configuration, one geometry has one 100-dimensional output vector.

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

radius and centered-z extent scale together so that enclosed cylindrical volume increases linearly with shell number. The shells therefore have equal nominal volume.

The shell target is determined from the event endpoint:

```text
r = sqrt(x^2 + y^2)
z_from_center = abs(z - z_center)
```

The CNP input uses source-position features, while the target shell uses the endpoint position.

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

The shared loader may normalize common naming variations. An event ID is preserved or generated.

## File manifest

Place `file_manifest.csv` inside the configured raw-data directory.

Required columns:

| Column | Meaning |
|---|---|
| `filename` | Raw event filename, including extension |
| `R` | Detector radius |
| `Z` | Detector centered-z extent |
| `z_center` | Detector center in the raw z-coordinate system |
| `fidelity` | Exact numeric fidelity: `0` or `1` |

Example:

```csv
filename,R,Z,z_center,fidelity
geometry_A_lf.csv,1490,1983,1983,0
geometry_A_hf.csv,1490,1983,1983,1
geometry_B_lf.csv,1600,2100,2100,0
geometry_B_hf.csv,1600,2100,2100,1
```

Fidelity has one definition throughout the pipeline:

```text
0 = low fidelity
1 = high fidelity
```

Fidelity comes only from `file_manifest.csv`. It is not assigned from directory names, pool fractions, YAML lists, filenames, or numerical ordering.

`R`, `Z`, and `z_center` should be supplied explicitly. They should not be inferred from event coverage because one source component may not span the complete detector.

## Preprocessing configuration

Example `config/pipeline_config.json`:

```json
{
  "file_load": {
    "input_dir": "data/raw",
    "max_rows_per_file": null
  },
  "split": {
    "validation_fraction": 0.4,
    "random_seed": 42
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

`validation_fraction` applies only to manifest-defined high-fidelity events.

For example:

```text
validation_fraction = 0.4
```

means:

- all fidelity-0 events remain in LF training;
- 60% of fidelity-1 events go to HF training;
- 40% of fidelity-1 events go to HF validation.

There are no LF/HF pool fractions and preprocessing never changes an event's fidelity.

`validation_block_size: null` uses the HF block size.

`max_rows_per_file: null` loads the complete file.

## Run preprocessing

From the repository root:

```bash
python cnp_mfgp/prepare_cnp_mfgp_data.py \
  --config cnp_mfgp/config/pipeline_config.json \
  --manifest file_manifest.csv
```

The preparation script deletes the configured output directory before rebuilding it. Use a dedicated or versioned dataset path.

## What preprocessing does

1. Loads each manifest row and validates fidelity as exactly `0` or `1`.
2. Adds centered source and endpoint coordinates.
3. Stores detector geometry and fidelity on every event.
4. Builds the shell table for each geometry.
5. Assigns one zero-based shell class to each valid event.
6. Combines labeled events across files.
7. Sends every fidelity-0 event to LF training.
8. Splits only fidelity-1 events into HF training and HF validation.
9. Divides each dataset into event blocks.
10. Writes categorical HDF5 blocks and summary tables.

## Prepared directory structure

```text
<output_dir>/
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

Typical metadata:

```text
event_index
original_event_id
shell_index
source_file
fidelity
split_name
detector_R
detector_Z
detector_z_center
```

There should be no alternate fidelity fields such as `source_fidelity` or `output_fidelity`.

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

  # Geometry range used for an optional MF-GP prediction grid.
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

  path_out_cnp: ../outputs/cnp
  path_out_mfgp: ../outputs/mfgp
```

There is no fidelity list in the YAML. CNP prediction reads the per-event `fidelity` dataset from HDF5 metadata.

The `iteration` list remains aligned with `path_to_files_predict` when multiple experiment iterations are exported.

---

# 3. Train the CNP

For each event:

```text
x = concatenate(theta, phi)
y = target_shell
```

The model returns 100 shell logits, which become a probability distribution after softmax.

The implementation includes:

- a deterministic CNP encoder and decoder;
- one-hot context shell labels;
- weighted categorical cross entropy;
- class weights based on training shell counts;
- gradient clipping;
- random or fixed context size;
- minibatch or full-pass training;
- MC dropout during prediction.

## Train

```bash
python cnp_mfgp/cnp_clean_pipeline.py \
  --config cnp_mfgp/config/settings_shell_minibatch.yaml \
  --device cuda \
  train
```

CPU:

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

---

# 4. Run CNP prediction

CNP prediction evaluates every configured HDF5 directory and preserves each event's manifest-defined fidelity.

## Training LF/HF prediction

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

## Held-out HF validation prediction

Use a validation YAML whose prediction path is:

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

Use a separate suffix so validation output does not overwrite training output.

## Aggregated MF-GP input

The principal CNP output is:

```text
cnp_<version>_<suffix>_<epochs>epochs.csv
```

The table is long-form and grouped by:

```text
iteration
fidelity
detector_R
detector_Z
shell_index
```

Important columns:

| Column | Meaning |
|---|---|
| `fidelity` | Manifest-defined `0` or `1` |
| `shell_index` | One-based shell number |
| `n_samples` | Events contributing to the aggregate |
| `y_cnp` | Mean CNP probability assigned to the shell |
| `y_cnp_err` | MC-dropout probability uncertainty |
| `y_raw` | Observed fraction of events in the shell |

For each geometry/fidelity combination, the CSV must contain exactly one aggregate row for every shell from `1` through `n_shells`.

## Large diagnostic output

The optional all-shell CSV contains one row per event × shell and can become extremely large. Keep:

```python
all_shells=False
```

for routine runs. The aggregated MF-GP input already contains the complete shell distribution and does not require the event × shell table.

---

# 5. Fit the distribution-aware MF-GP

## Why the structure changed

A 100-bin shell output is one correlated probability distribution:

```text
[p_1, p_2, ..., p_100]
```

The bins are not 100 independent detector observations. They are constrained by:

```text
p_i >= 0
sum_i p_i = 1
```

The recommended pipeline therefore uses one geometry as one statistical sample with a 100-dimensional output.

## Long-form to geometry-level matrix

The CNP CSV is converted from:

```text
detector_R | detector_Z | fidelity | shell_index | probability
```

into one vector per geometry and fidelity:

```text
detector_R | detector_Z | fidelity | p_1 | p_2 | ... | p_100
```

The model uses:

```text
LF distribution = y_cnp for fidelity 0
HF distribution = y_raw for fidelity 1
```

## Centered-log-ratio transformation

Raw probability vectors live on a simplex rather than ordinary Euclidean space. Before PCA, the pipeline:

1. clips each probability with a small epsilon;
2. renormalizes the vector;
3. applies the centered-log-ratio transform.

For a distribution `p`:

```text
clr(p_i) = log(p_i) - mean_j(log(p_j))
```

This makes relative changes among shell probabilities suitable for PCA and GP modeling.

## PCA compression

PCA is fitted on the combined LF and HF training distributions in CLR space.

The 100-shell distribution becomes a smaller coefficient vector:

```text
[p_1, ..., p_100]
        ↓ CLR + PCA
[z_1, z_2, ..., z_k]
```

By default, enough components are retained to explain 99.5% of the training variance. These optional settings can be added to the existing YAML without changing any paths or file layout:

```yaml
mfgp_settings:
  pca_components: 0.995
  pca_epsilon: 1.0e-8
  distribution_mc_samples: 500
```

`pca_components` may also be an integer such as `5`. If the section is omitted, the defaults above are used.

## Autoregressive MF-GP per coefficient

For every retained PCA coefficient `z_k`, the pipeline fits:

```text
z_HF,k(theta) = rho_k * z_LF,k(theta) + delta_k(theta)
```

where:

- the LF GP learns the low-fidelity coefficient over geometry;
- `rho_k` scales the LF prediction at HF geometries;
- the discrepancy GP learns the remaining HF correction.

The coefficients are then reconstructed through inverse PCA and inverse CLR.

The final shell predictions are automatically:

- nonnegative;
- normalized to sum to one;
- represented as complete correlated distributions.

## Uncertainty reconstruction

The pipeline draws Monte Carlo samples from the independent latent coefficient predictions, reconstructs each shell distribution, and reports:

```text
predicted_shell_probabilities
predicted_shell_std
predicted_shell_q025
predicted_shell_q975
```

Each reconstructed draw sums to one. The current implementation fits the latent coefficient GPs independently and samples their predictive uncertainties independently. PCA makes the training coefficients orthogonal, but this approximation does not represent every possible cross-component posterior correlation.

## Run the MF-GP

```bash
python cnp_mfgp/mfgp_clean_pipeline.py \
  --config cnp_mfgp/config/settings_shell_minibatch.yaml \
  --cnp-csv cnp_mfgp/outputs/cnp/cnp_shell_v1_output_15epochs.csv \
  --validation-csv cnp_mfgp/outputs/cnp/cnp_shell_v1_output_validation_15epochs.csv \
  --iteration 0 \
  --target-transform linear \
  --grid-points 30
```

Optional command-line arguments remain compatible with the existing pipeline:

| Option | Meaning |
|---|---|
| `--cnp-csv` | Explicit aggregated CNP training CSV |
| `--validation-csv` | Explicit aggregated HF validation CSV |
| `--iteration` | Iteration value to select |
| `--target-transform linear` | Compatibility option; CLR+PCA is used internally |
| `--grid-points` | Geometry grid points per theta axis |
| `--random-state` | Random seed |
| `--quiet` | Reduce console output |

## Single-geometry smoke test

A single geometry with LF and HF data is accepted automatically.

This verifies:

- long-form shell loading;
- fidelity separation;
- shell-vector pivoting;
- CLR/PCA compression;
- latent MF-GP fitting;
- distribution reconstruction;
- output and plotting code.

It does **not** validate geometry interpolation. With one geometry, detector-length scales and geometry dependence are not statistically identifiable. The script prints a warning when fewer than three geometries exist in either fidelity.

For a scientific geometry-learning run, use several distinct LF and HF geometries. Three is still only a minimal practical threshold; more geometries are preferable.

## MF-GP outputs

The MF-GP output directory receives files such as:

```text
mfgp_<version>_linear_model_iter0.json
mfgp_<version>_linear_metrics_iter0.json
mfgp_<version>_linear_predictions_iter0.csv
mfgp_<version>_linear_grid_iter0.csv
mfgp_<version>_linear_hf_observations_iter0.png
mfgp_<version>_linear_mean_std_iter0.png
mfgp_<version>_linear_validation_shell_distributions_iter0.png
```

The prediction and grid CSVs contain one row per detector geometry. Their list-valued columns store the complete reconstructed shell distribution and uncertainty interval:

```text
predicted_shell_probabilities
predicted_shell_std
predicted_shell_q025
predicted_shell_q975
```

## Model artifacts

The pipeline keeps the existing artifact structure. It writes:

```text
mfgp_<version>_linear_model_iter<iteration>.json
mfgp_<version>_linear_metrics_iter<iteration>.json
mfgp_<version>_linear_predictions_iter<iteration>.csv
mfgp_<version>_linear_grid_iter<iteration>.csv
```

The model JSON records the PCA representation, component-level `rho` values, input geometry headers, fidelity definition, and sample counts. The fitted Python object is not serialized, so there is no `cloudpickle` dependency. The model is fitted and used in memory during the normal pipeline run, matching the previous `mfgp_clean_pipeline.py` workflow.

## Metrics

The metrics JSON includes distribution-level measures:

```text
mean_shell_rmse
mean_shell_mae
mean_total_variation
mean_jensen_shannon
max_sum_error
```

Total variation and Jensen–Shannon divergence compare the complete predicted and true shell distributions rather than evaluating each shell in isolation.

---

# Internal MF-GP data structure

The CNP aggregate CSV remains unchanged and long-form, with one row per geometry, fidelity, and shell. `mfgp_clean_pipeline.py` pivots it internally into one row per detector geometry and fidelity:

```text
detector_R | detector_Z | fidelity | cnp_shell_probabilities
```

`cnp_shell_probabilities` is an ordered list of length `n_shells`. The same internal row also retains the observed `y_raw` shell distribution and CNP uncertainty list. The LF latent model uses fidelity-0 CNP probabilities, while the HF discrepancy uses fidelity-1 observed shell fractions.

This conversion happens only inside the MF-GP loader. It does not change the CNP CSV, prepared-data directories, or repository file layout.

---

# Validation design

Validation is high-fidelity only.

Preprocessing randomly holds out the configured fraction of fidelity-1 events. CNP prediction aggregates those held-out events into one observed HF shell distribution per geometry.

The MF-GP uses:

- training CSV: LF and HF rows;
- validation CSV: HF-only rows.

The validation CSV must contain only:

```text
fidelity = 1
```

It is never used to fit PCA or the latent GPs.

Because preprocessing splits HF **events**, training and validation may describe the same detector geometry using independent event samples. This evaluates reconstruction stability and finite-sample distribution agreement at known geometries. It does not by itself test interpolation to an unseen detector geometry. Geometry-generalization testing requires holding out one or more complete geometry values separately.

---

# Emulation

`emulation.py` generates synthetic source positions and passes them through a trained CNP.

The default source generator samples a cylindrical skin outside the detector side wall:

```text
R_max <= s_r <= R_max + width
```

Emulated endpoint labels are placeholders, so emulation output should not be used as HF truth for MF-GP fitting unless physical truth is available independently.

---

# Recommended notebook workflow

1. Locate the repository root.
2. Define raw, prepared, and output paths.
3. Prepare HDF5 blocks from `file_manifest.csv`.
4. Train or load the CNP.
5. Predict on LF training, HF training, and held-out HF validation data.
6. Confirm that every geometry/fidelity contains all shell indices.
7. Inspect CNP shell distributions.
8. Fit `mfgp_clean_pipeline.py`.
9. Inspect PCA explained variance.
10. Compare reconstructed HF training and validation distributions.
11. Evaluate total variation and Jensen–Shannon divergence.
12. Save all artifacts under `outputs/`.

Notebooks should call pipeline functions rather than copying their implementations.

---

# Storage guidance

## Do not save event × shell output routinely

The optional all-shell table scales as:

```text
number of events × number of shells
```

For 15 million events and 100 shells, this is 1.5 billion rows before CSV overhead.

The MF-GP only needs the aggregated shell CSV, so keep:

```python
all_shells=False
```

## Version prepared datasets

Preprocessing deletes its output directory before rebuilding it. Use paths such as:

```text
data/processed/cnp_mfgp_v1/
data/processed/cnp_mfgp_v2/
```

## Separate datasets from model outputs

```text
data/                         raw and prepared event data
cnp_mfgp/outputs/cnp/         CNP checkpoints and predictions
cnp_mfgp/outputs/mfgp/        MF-GP artifacts
cnp_mfgp/outputs/emulation/   emulated predictions
cnp_mfgp/outputs/figures/     plots and HTML views
```

---

# Troubleshooting

## Manifest fidelity error

Every manifest row must have exactly:

```text
fidelity = 0
```

or:

```text
fidelity = 1
```

Values such as `LF`, `HF`, `2`, `0.5`, blanks, or inferred directory labels are invalid.

## MF-GP cannot find both fidelities

The training CNP aggregate CSV must contain both:

```text
fidelity = 0
fidelity = 1
```

The validation aggregate CSV should contain only:

```text
fidelity = 1
```

## Incomplete shell grid

For every geometry and fidelity, confirm that the aggregate CSV contains exactly:

```text
shell_index = 1, 2, ..., n_shells
```

Missing or duplicate shells cause the distribution pipeline to stop rather than silently constructing an invalid vector.

## Only one LF/HF point is reported

The distribution pipeline counts detector geometries, not shell rows. One detector geometry is intentionally one statistical sample with a 100-dimensional output.

A message reporting:

```text
LF geometries: 1
HF geometries: 1
```

is therefore correct for a one-geometry smoke test.

## Fewer than three geometries warning

This warning means the pipeline can run but cannot reliably learn geometry-dependent length scales or interpolation. Add additional detector geometries for scientific use.

## PCA keeps only one component

With one LF and one HF geometry, only two distribution samples are available to fit PCA, so the rank is at most one after centering. This is expected for a smoke test.

More geometries permit additional distribution-shape modes to be learned.

## Predicted probabilities do not sum to one

The MF-GP normalizes every inverse CLR reconstruction. Sum errors should be near floating-point precision. Check that you are using `mfgp_clean_pipeline.py`, not a custom scalar reconstruction.

## CUDA or system memory is exhausted

Reduce CNP:

```yaml
batch_size_train
files_per_batch_train
```

or reduce prediction:

```text
--chunk-size
```

The MF-GP itself operates on geometry-level aggregate vectors and is normally much smaller than event-level CNP prediction.

## Disk fills during CNP prediction

Confirm that event × shell output is disabled and remove obsolete datasets, prediction versions, temporary HDF5 files, or operating-system trash.

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

# Predict LF and HF training data
python cnp_mfgp/cnp_clean_pipeline.py \
  --config cnp_mfgp/config/settings_shell_minibatch.yaml \
  --device cuda \
  predict \
  --model-path cnp_mfgp/outputs/cnp/<checkpoint>.pth \
  --output-suffix output

# Predict held-out HF validation data
python cnp_mfgp/cnp_clean_pipeline.py \
  --config cnp_mfgp/config/settings_shell_validation.yaml \
  --device cuda \
  predict \
  --model-path cnp_mfgp/outputs/cnp/<checkpoint>.pth \
  --output-suffix output_validation

# Fit the recommended distribution-aware MF-GP
python cnp_mfgp/mfgp_clean_pipeline.py \
  --config cnp_mfgp/config/settings_shell_minibatch.yaml \
  --cnp-csv cnp_mfgp/outputs/cnp/<training-prediction>.csv \
  --validation-csv cnp_mfgp/outputs/cnp/<validation-prediction>.csv \
  --iteration 0 \
  --target-transform linear
```

For broader repository organization, see `../README.md`.

For development history and experiment rationale, see `../PROJECT_EXPERIMENT_GUIDE.md`.
