# XLZD Machine-Learning Workflows - Updated July 14th, 2026

This repository contains machine-learning workflows for modeling event distributions in proposed XLZD detector geometries.

The central research goal is to learn how detector geometry and event-source position affect where energy-deposition events occur inside the detector. The active workflow includes:

- a **Conditional Neural Process (CNP)** for event-level shell prediction and a **Multi-Fidelity Gaussian Process (MF-GP)** for learning geometry-level trends from low- and high-fidelity data;
- shared preprocessing, geometry, HDF5, and dataset utilities;
- notebooks and experiments used to evaluate alternative model choices.

The main active implementation is in [`cnp_mfgp/`](cnp_mfgp/).

For the detailed CNP–MFGP workflow, configuration format, commands, outputs, and modeling conventions, see:

- [`cnp_mfgp/README.md`](cnp_mfgp/README.md)

For the history of the project and the reasoning behind earlier experiment branches, see:

- [`PROJECT_EXPERIMENT_GUIDE.md`](PROJECT_EXPERIMENT_GUIDE.md)

---

## Core idea

The detector volume is divided into nested cylindrical shells.

For each simulated event, the model receives:

```text
detector geometry + source position
```

and predicts:

```text
a probability distribution over detector shells
```

The workflow then aggregates those event-level predictions at each detector geometry and combines low- and high-fidelity information with an MFGP.

At a high level:

```text
Raw simulated events
        │
        ▼
Geometry-aware preprocessing
        │
        ▼
LF/HF/Validation HDF5 blocks
        │
        ▼
Conditional Neural Process
        │
        ├── event-level shell predictions
        └── geometry-level aggregated shell probabilities
                         │
                         ▼
              Multi-Fidelity Gaussian Process
                         │
                         ▼
          geometry-response mean and uncertainty
```

This structure separates two related problems:

1. **Event-level modeling:** determine which detector shell an event is likely to occupy.
2. **Geometry-level modeling:** determine how the aggregate event distribution changes as detector geometry changes.

---

## Repository layout

```text
.
├── archive/
├── cnp_mfgp/
├── common/
├── data/
├── settings_examples/
├── PROJECT_EXPERIMENT_GUIDE.md
├── README.md
└── requirements.txt
```

### [`cnp_mfgp/`](cnp_mfgp/)

The active CNP–MF-GP workflow.

This folder contains:

- data preparation;
- CNP training and prediction;
- MF-GP fitting and evaluation;
- synthetic-source emulation;
- visualization helpers;
- active configurations;
- notebooks;
- examples;
- generated pipeline outputs;
- additional experimental branches.

See [`cnp_mfgp/README.md`](cnp_mfgp/README.md) for the complete technical guide.

### [`common/`](common/)

Shared utilities used by the active pipeline.

These modules handle:

- configuration dataclasses;
- raw event loading and normalization;
- LF/HF/validation pool construction;
- block creation;
- shell geometry;
- centered-coordinate calculations;
- HDF5 serialization;
- pipeline logging and timing.

Code that is useful across multiple workflows belongs here rather than being duplicated inside a model-specific folder.

### [`data/`](data/)

Local raw datasets and file preparation requirements.

This data should not be committed to version control

Includes file_manifest.csv - a file that dictates the detector geometry and fidelity of each of the input files

### [`settings_examples/`](settings_examples/)

Example configuration files and reusable settings templates.

Use these as starting points for new runs. Copy or adapt the relevant files into [`cnp_mfgp/config/`](cnp_mfgp/config/) before changing experiment-specific paths and parameters.

### [`archive/`](archive/)

Older implementations, retired layouts, and historical experiments retained for reference.

Archived files are not part of the default workflow and may depend on paths, data formats, or APIs that are no longer current.

### [`PROJECT_EXPERIMENT_GUIDE.md`](PROJECT_EXPERIMENT_GUIDE.md)

A longer record of:

- the development history;
- previous theta and target definitions;
- abandoned or superseded approaches;
- experiment-specific reasoning;
- architectural changes.

Use the README files for the current workflow and the experiment guide for historical context.

### [`requirements.txt`](requirements.txt)

Python dependencies for the repository.

---

## Installation

Create and activate a virtual environment from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

A CUDA-enabled PyTorch installation is optional. The CNP uses CUDA when available unless a device is explicitly selected.

---

## Starting a new run

Run commands from the repository root.

A typical workflow is:

```bash
# 1. Prepare event-level shell-classification blocks
python cnp_mfgp/prepare_cnp_mfgp_data.py \
  --config cnp_mfgp/config/pipeline_config.json \
  --manifest file_manifest.csv

# 2. Train the CNP
python cnp_mfgp/cnp_clean_pipeline.py \
  --config cnp_mfgp/config/settings_shell_minibatch.yaml \
  --device cuda \
  train

# 3. Predict on configured LF/HF datasets
python cnp_mfgp/cnp_clean_pipeline.py \
  --config cnp_mfgp/config/settings_shell_minibatch.yaml \
  --device cuda \
  predict \
  --model-path cnp_mfgp/outputs/cnp/<checkpoint>.pth \
  --output-suffix output

# 4. Fit the MF-GP
python cnp_mfgp/mfgp_clean_pipeline.py \
  --config cnp_mfgp/config/settings_shell_minibatch.yaml \
  --cnp-csv cnp_mfgp/outputs/cnp/<aggregated-output>.csv
```

The exact paths are controlled by the JSON and YAML configuration files.

Detailed setup and command options are documented in [`cnp_mfgp/README.md`](cnp_mfgp/README.md).

---

## Configuration strategy

The repository separates reusable examples from active experiment configuration:

```text
settings_examples/   reusable templates
cnp_mfgp/config/     settings used by the active pipeline
```

The data-preparation stage uses JSON.

The CNP and MF-GP stages use YAML.

Relative paths inside a CNP/MF-GP YAML file are resolved relative to that YAML file. Explicitly passing `--config` is recommended for all command-line runs.

---

## Development conventions

### Run from the repository root

Several modules locate the repository by searching parent directories for:

```text
README.md
PROJECT_EXPERIMENT_GUIDE.md
```

Running from the root also ensures that imports from `common` and `cnp_mfgp` resolve consistently.

### Keep active and historical work separate

Use:

- `cnp_mfgp/` for the active pipeline;
- `cnp_mfgp/additional_experiments/` for current experimental variations;
- `archive/` for retired structures and obsolete implementations.

### Keep generated files out of source folders

Model checkpoints, prediction tables, plots, and interactive HTML files should be written to configured output directories, normally under:

```text
cnp_mfgp/outputs/
```

Large raw and prepared datasets should remain under:

```text
data/
```

### Put reusable code in `common`

Geometry, I/O, HDF5, block-building, and dataset-splitting logic should remain model-independent whenever possible.

---

## Current primary workflow

The supported active path is:

```text
cnp_mfgp/prepare_cnp_mfgp_data.py
        ↓
cnp_mfgp/cnp_clean_pipeline.py
        ↓
cnp_mfgp/mfgp_clean_pipeline.py
```

The supporting modules are:

```text
cnp_mfgp/emulation.py
cnp_mfgp/visualize.py
common/
```

Start with [`cnp_mfgp/README.md`](cnp_mfgp/README.md) before running or modifying the pipeline.
