# XLZD Surrogate Modeling Framework

This repository contains a machine-learning framework for modeling event distributions and geometry-dependent detector response in XLZD simulations.

The framework is organized around a clear separation between **reusable machine-learning models** and the **physics assumptions used to construct their inputs, targets, losses, and aggregated observables**. The Conditional Neural Process (CNP) and Multi-Fidelity Gaussian Process (MF-GP) are implemented as physics-agnostic model cores, while detector- and distribution-specific behavior is implemented in independent physics layers such as the positional-shell workflow.

This structure allows the same model implementations to be reused for other event classifications, detector representations, or physics-informed distributions without embedding those assumptions into the CNP or MF-GP themselves.

---

## Architecture Overview

The repository separates responsibilities into three main layers:

```text
                        PHYSICS / DISTRIBUTION LAYER
                        positional_shells/shells.py
                                  │
              ┌───────────────────┴───────────────────┐
              │                                       │
              ▼                                       ▼
     generic CNP interface                  generic MF-GP interface
          (x, y batches)                    (LF/HF x, y arrays)
              │                                       │
              ▼                                       ▼
            cnp.py                                  mfgp.py
              │                                       │
              └───────────────────┬───────────────────┘
                                  │
                                  ▼
                             common utilities
```

The dependency direction is intentional:

- `positional_shells/` knows what detector geometry, source position, shells, and fidelity mean.
- `cnp.py` knows only how to train and evaluate a categorical Conditional Neural Process.
- `mfgp.py` knows only how to fit and evaluate a generic two-fidelity scalar MF-GP.
- `common/` contains reusable data, I/O, HDF5, configuration, coordinate, and visualization utilities.

**The generic model files do not need to know that the current application uses cylindrical detector shells.**

---

## Repository Layout

```text
XLZD/
├── cnp.py
├── mfgp.py
│
├── positional_shells/
│   ├── shells.py
│   ├── train.ipynb
│   └── validate.ipynb
│
├── common/
│   ├── __init__.py
│   ├── config.py
│   ├── dataset.py
│   ├── geometry.py
│   ├── h5_utils.py
│   ├── io_utils.py
│   ├── pipeline_utils.py
│   ├── theta.py
│   └── visualize.py
│
├── configs/
│   ├── cnp.toml
│   ├── mfgp.toml
│   └── shell.toml
│
├── data/
├── outputs/
├── README.md
└── requirements.txt
```

Generated datasets, checkpoints, predictions, and plots should remain in `data/` and `outputs/` rather than being committed to source control.

---

# 1. Physics-Agnostic CNP

`cnp.py` contains the generalized categorical Conditional Neural Process.

The model receives generic inputs:

```text
x : (N, x_dim) feature matrix
y : (N,) integer class labels
    or
    (N, n_classes) soft categorical targets
```

The feature dimension and number of classes are determined by the external data provider.

### CNP responsibilities

`cnp.py` handles:

- CNP encoder/decoder construction;
- context/target splitting;
- random or fixed context sizes;
- hard or soft categorical targets;
- default categorical cross-entropy;
- optional externally supplied loss functions;
- minibatch or full-epoch training providers;
- validation;
- checkpoint creation and loading;
- fixed inference-context storage;
- MC-dropout probability estimates and uncertainties;
- generic classification diagnostics.

The main training interface is built around externally supplied providers:

```python
cnp.train_cnp(
    train_batch_fn=...,
    validation_batch_fn=...,
    inference_context_fn=...,
    loss_fn=...,
    n_classes=...,
    out_dir=...,
    ...
)
```

The model therefore does not care whether a class means a detector shell, a topology, a region, or another categorical target.

---

# 2. Physics-Agnostic MF-GP

`mfgp.py` contains the generalized two-level autoregressive Multi-Fidelity Gaussian Process.

Its input is represented by the generic `MFGPTrainingData` structure:

```python
MFGPTrainingData(
    x_lf=...,
    y_lf=...,
    x_hf=...,
    y_hf=...,
    y_lf_err=...,
    input_names=...,
)
```

No detector or shell interpretation is built into this structure. The MF-GP only requires:

- low-fidelity input points and scalar targets;
- high-fidelity input points and scalar targets;
- matching feature dimensions;
- optional LF target uncertainty.

The current model is a two-level autoregressive construction:

```text
y_HF(x) = rho * y_LF(x) + delta(x)
```

where one Gaussian Process models the low-fidelity response and a second Gaussian Process models the high-fidelity discrepancy.

The model uses standardized inputs/targets and a Matérn + white-noise kernel. The fitted model, input names, hyperparameters, learned `rho`, kernels, predictions, and metrics are saved independently of the shell application.

### MF-GP responsibilities

`mfgp.py` handles:

- LF and HF input validation;
- input and target standardization;
- low-fidelity GP fitting;
- autoregressive LF/HF coupling;
- discrepancy GP fitting;
- configurable GP noise settings;
- uncertainty propagation;
- checkpoint save/load;
- prediction at arbitrary input points without refitting;
- RMSE, MAE, R², and coverage metrics;
- prediction CSV and metadata output.

The generic training entry point is:

```python
mfgp.run_mfgp_training(config_path, training_data)
```

and a saved model can be evaluated with:

```python
mfgp.run_mfgp_prediction(
    config_path,
    model_path=model_path,
    x=x,
    y_true=y_true,
)
```

---

# 3. Physics-Informed Distribution Layers

Physics-specific assumptions are now kept outside of the generic models.

The first implemented distribution is:

```text
positional_shells/
```

This folder defines how XLZD events are converted into a shell-classification problem and how those event-level predictions are converted into geometry-level MF-GP targets.

A future physics representation can be added as another independent folder using the same generic model interfaces.

For example:

```text
XLZD/
├── cnp.py
├── mfgp.py
├── positional_shells/
├── another_distribution/
└── common/
```

The new distribution would define its own data conversion, labels, optional physics-informed loss, aggregation, and wrappers without requiring changes to the generic CNP or MF-GP implementations.

---

# 4. Positional-shell implementation

`positional_shells/shells.py` contains all shell-specific logic used by the current XLZD workflow.

This includes:

- shell configuration;
- detector-specific shell construction;
- centered-z coordinate handling;
- raw-event conversion;
- LF/HF and validation splitting;
- HDF5 event-class block creation;
- CNP batch providers;
- shell-aware classification loss;
- CNP prediction aggregation;
- conversion from shell predictions to MF-GP training data;
- held-out HF validation data construction;
- wrappers around the generic CNP and MF-GP APIs.

The separation of responsibilities is:

```text
physics assumptions  -> positional_shells/shells.py
model implementation -> cnp.py / mfgp.py
```

## Shell definition

For each detector geometry, the volume is divided into nested cylindrical shells centered on the detector center.

The outer boundary of shell `i` is

```text
R_i = R_max * (i / n_shells)^scale_power
Z_i = Z_max * (i / n_shells)^scale_power
```

and shell `i` is the region inside boundary `i` and outside boundary `i - 1`.

With the default

```text
scale_power = 1/3
```

the radial and axial dimensions scale together according to the configured shell construction.

Internally, the CNP target is a **zero-indexed categorical class**. Human-readable shell outputs use shell numbering beginning at 1.

---

## CNP features for the shell distribution

The shell adapter currently defines:

```python
THETA_HEADERS = ["detector_R", "detector_Z"]
PHI_HEADERS   = ["s_r", "s_z_from_center"]
```

where:

- **theta** describes detector-level/global geometry;
- **phi** describes event-level/source information.

The shell HDF5 provider converts these into the generic CNP representation:

```text
x = concat(theta, phi)
y = target_shell
```

so the CNP sees a normal feature matrix and class label. It is not passed a `ShellConfig`, shell boundaries, or shell geometry.

This is what allows the CNP implementation to remain physics agnostic.

---

## Physics-informed shell loss

The ordered nature of the shell classes is useful physics information, but that assumption does **not** belong in the generic CNP.

Instead, `positional_shells/shells.py` defines a shell-specific loss that combines:

- hard categorical cross-entropy; and
- a Gaussian-smoothed target centered on the true shell.

This lets nearby shell errors be treated differently from distant shell errors while preserving a generic CNP implementation.

The shell wrapper injects this loss into:

```python
cnp.train_cnp(..., loss_fn=shell_loss)
```

Class weighting can also be enabled in `shell.toml` without changing `cnp.py`.

---

# 5. Data preparation

The shell workflow begins from raw simulation files plus a `file_manifest.csv`.

Each manifest row associates a simulation file with its detector geometry and fidelity:

```text
filename,R,Z,z_center,fidelity
```

with the convention:

```text
fidelity = 0  -> low fidelity (LF)
fidelity = 1  -> high fidelity (HF)
```

The preparation stage:

```text
Raw simulation files
        │
        ├── detector geometry from file_manifest.csv
        ├── centered source coordinates
        ├── shell assignment
        ├── LF/HF labeling
        └── held-out HF validation split
        │
        ▼
Shell-labelled events
        │
        ▼
Event-class HDF5 blocks
```

The generated HDF5 blocks store:

```text
theta
theta_labels
phi
phi_labels
target_shell
metadata
```

This HDF5 format is the boundary between shell-specific preparation and the generic CNP interface.

---

# 6. CNP to MF-GP adapter

The MF-GP is deliberately not aware of shells either.

After CNP prediction, `shells.py` aggregates event-level probabilities by detector geometry and converts the results into generic `MFGPTrainingData`.

For a selected containment shell `n`:

### Low fidelity

```text
x_lf     = detector geometry
y_lf     = sum of CNP probabilities for shells 1 ... n
y_lf_err = CNP shell uncertainties combined in quadrature
```

### High fidelity

```text
x_hf = detector geometry
y_hf = sum of raw HF probabilities for shells 1 ... n
```

The resulting arrays are passed to the generic MF-GP without shell metadata or detector-specific behavior.

This adapter is what connects the two reusable model stages:

```text
CNP event classifier
        │
        ▼
physics-specific aggregation
        │
        ▼
MFGPTrainingData
        │
        ▼
generic MF-GP
```

---

# 7. End-to-end shell workflow

The current supported workflow is:

```text
Raw simulation files + file_manifest.csv
                  │
                  ▼
        prepare_shell_cnp_data
                  │
                  ▼
        shell-labelled HDF5 blocks
                  │
                  ▼
        ShellH5EventPool providers
                  │
                  ▼
          generic cnp.train_cnp
                  │
                  ▼
     event-level class probabilities
                  │
                  ▼
       shell prediction aggregation
                  │
                  ▼
       build MFGPTrainingData
                  │
                  ▼
       generic mfgp.run_mfgp_training
                  │
                  ▼
         MF-GP mean + uncertainty
                  │
                  ▼
          held-out HF validation
```

The training and validation notebooks in `positional_shells/` provide the intended high-level interface for this workflow.

---

# 8. Configuration

Configuration has been separated by responsibility.

## `configs/shell.toml`

Contains shell/distribution-specific settings, including:

- raw input and prepared-data locations;
- number of shells;
- shell scaling;
- LF/HF block sizes;
- HF validation split;
- CNP provider settings;
- shell-specific loss settings and class weighting.

## `configs/cnp.toml`

Contains generic CNP settings, including:

- output version and location;
- device;
- epochs and training steps;
- batch size;
- context ratio and mode;
- inference-context size;
- representation dimension;
- hidden dimension;
- dropout;
- optimizer settings.

## `configs/mfgp.toml`

Contains MF-GP training and prediction settings, including:

- output version and location;
- minimum LF/HF point counts;
- GP noise floors;
- white-kernel noise configuration;
- prediction interval settings;
- prediction chunk size.

The current shell workflow also reads the `[shell].n_shell` entry in `mfgp.toml` to choose the outer shell of the containment region. That value is interpreted by the shell adapter before the generic MF-GP receives its training arrays.

---

# 9. Training and validation notebooks

## `positional_shells/train.ipynb`

Provides the end-to-end training workflow:

1. load shell/CNP/MF-GP configuration;
2. prepare shell-labelled datasets;
3. train the CNP;
4. generate training and validation CNP predictions;
5. visualize shell prediction behavior;
6. construct MF-GP training data;
7. train the MF-GP;
8. inspect fitted response and uncertainty.

## `positional_shells/validate.ipynb`

Provides the held-out evaluation workflow for trained models, including MF-GP predictions and validation metrics/plots against high-fidelity truth.

---

# 10. Shared `common/` utilities

Code that is reusable across multiple physical distributions belongs in `common/` rather than in a model or distribution-specific module.

The current shared modules cover areas such as:

- configuration dataclasses;
- dataset/block construction;
- general geometry helpers;
- HDF5 utilities;
- file and table I/O;
- pipeline logging/timing;
- centered coordinate and theta handling;
- common plotting/visualization helpers.

The intended rule is:

```text
specific to one physical representation -> that representation's folder
reusable across representations         -> common/
generic machine-learning implementation -> cnp.py or mfgp.py
```

For example, shell construction, shell labeling, and shell-aware losses belong in `positional_shells/`, while generic file/HDF5 helpers can remain in `common/`.

---

# 11. Adding a new physics-informed distribution

The framework is designed so a new physical representation does not require copying or rewriting either model.

A new distribution should provide the following pieces.

### 1. Define the physical representation

Specify the domain-level features and target.

For a categorical CNP application, this eventually needs to become:

```text
x -> generic feature matrix
y -> categorical target
```

The model does not need to know the physical meaning of either.

### 2. Implement data preparation

Convert raw simulation output into a stable prepared representation and preserve any metadata needed for later aggregation or validation.

### 3. Provide CNP batches

Create providers that return:

```python
(x, y)
```

or `cnp.ClassificationBatch` objects.

### 4. Define an optional physics-informed loss

If the target has structure not represented by ordinary categorical cross-entropy, define the loss in the distribution module and inject it into `cnp.train_cnp`.

Do not add distribution-specific assumptions directly to `cnp.py`.

### 5. Define the geometry-level quantity of interest

Aggregate event-level outputs into the scalar LF/HF response that the MF-GP should learn.

### 6. Construct generic MF-GP data

Return:

```python
mfgp.MFGPTrainingData(...)
```

with no requirement for the MF-GP to understand the source physics.

### 7. Add thin runner functions/notebooks

The distribution layer should own the high-level wrappers that connect its physical data representation to the generic model APIs.

This pattern keeps each new physics problem independent while reusing the same tested model implementations.

---

# 12. Outputs and checkpoints

The two model cores save portable model artifacts separately from the physical data-preparation layer.

## CNP outputs

Depending on configuration, CNP training produces artifacts such as:

```text
cnp_<version>_model.pth
training history CSV
training-history plot
classification monitor plot
aggregated shell prediction CSVs
```

The CNP checkpoint stores the model architecture, feature/class information, and a fixed inference context so predictions can be reproduced without rebuilding the training context manually.

## MF-GP outputs

MF-GP training produces artifacts such as:

```text
mfgp_<version>_model.joblib
mfgp_<version>_model.json
training prediction CSV
metrics JSON
prediction CSVs
```

The metadata includes the feature names, fitted autoregressive coefficient, noise parameters, and fitted GP kernels.

---

# 13. Design principles

The current architecture follows a few important rules:

1. **Models should be physics agnostic.**  
   `cnp.py` and `mfgp.py` operate on generic mathematical inputs rather than detector-specific objects.

2. **Physics assumptions should be explicit.**  
   Shell geometry, shell ordering, containment definitions, and detector/source coordinates live in the positional-shell adapter.

3. **Physics-informed behavior should be injected, not hard-coded into the model.**  
   The shell-aware loss is passed into the generic CNP through its loss interface.

4. **Conversion between model stages belongs to the physics layer.**  
   The shell adapter decides how CNP probabilities become LF/HF scalar targets for the MF-GP.

5. **Shared infrastructure should remain reusable.**  
   General I/O, HDF5, dataset, configuration, coordinate, and visualization logic belongs in `common/` when it can support multiple distributions.

6. **Notebooks orchestrate; modules implement.**  
   `train.ipynb` and `validate.ipynb` should primarily call stable functions from the model and physics modules rather than containing duplicate pipeline logic.

7. **Generated artifacts do not belong in source code.**  
   Raw/prepared datasets and model outputs should remain under ignored `data/` and `outputs/` locations.
