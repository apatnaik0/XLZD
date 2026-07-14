# XLZD Experiment Guide

This document is meant to help a new reader understand:

- what problem this repository is trying to solve
- how the standard CNP + MF-GP pipeline works
- what each major experiment folder is doing
- what the motivation was for each variation
- how the different branches of work relate to each other

This is not a user quickstart. It is a process and design explanation.

## Table of Contents

- [1. High-Level Goal](#1-high-level-goal)
- [2. Standard Pipeline](#2-standard-pipeline)
  - [2.1 Data preparation](#21-data-preparation)
  - [2.2 Standard theta definition](#22-standard-theta-definition)
  - [2.3 Why the standard theta is useful](#23-why-the-standard-theta-is-useful)
  - [2.4 Limitation of the standard theta](#24-limitation-of-the-standard-theta)
- [3. H5 Conversion](#3-h5-conversion)
- [4. CNP Stage](#4-cnp-stage)
  - [4.1 What the CNP is doing](#41-what-the-cnp-is-doing)
  - [4.2 Why the CNP exists here](#42-why-the-cnp-exists-here)
  - [4.3 Outputs of the CNP stage](#43-outputs-of-the-cnp-stage)
- [5. MF-GP Stage](#5-mf-gp-stage)
  - [5.1 What MF-GP is doing](#51-what-mf-gp-is-doing)
  - [5.2 Why MF-GP exists here](#52-why-mf-gp-exists-here)
  - [5.3 Target transforms](#53-target-transforms)
- [6. Different CNP Training Modes](#6-different-cnp-training-modes)
- [7. LF Augmentation Experiments](#7-lf-augmentation-experiments)
  - [7.1 Motivation](#71-motivation)
  - [7.2 Important idea](#72-important-idea)
  - [7.3 What bootstrap augmentation does](#73-what-bootstrap-augmentation-does)
  - [7.4 What merged augmentation does](#74-what-merged-augmentation-does)
  - [7.5 What these experiments are trying to learn](#75-what-these-experiments-are-trying-to-learn)
  - [7.6 Caveat](#76-caveat)
- [8. Theta Augmentation Experiments](#8-theta-augmentation-experiments)
  - [8.1 Motivation](#81-motivation)
  - [8.2 Local jitter](#82-local-jitter)
  - [8.3 Midpoint](#83-midpoint)
  - [8.4 What these experiments are trying to learn](#84-what-these-experiments-are-trying-to-learn)
- [9. Shell Theta Experiment](#9-shell-theta-experiment)
  - [9.1 Motivation](#91-motivation)
  - [9.2 Initial shell-theta definition](#92-initial-shell-theta-definition)
  - [9.3 Why shell theta is conceptually useful](#93-why-shell-theta-is-conceptually-useful)
  - [9.4 Why shell theta is harder](#94-why-shell-theta-is-harder)
  - [9.5 Variation experiments for shell theta](#95-variation-experiments-for-shell-theta)
  - [9.6 Soft shell target](#96-soft-shell-target)
- [10. Depth Penetration Experiments](#10-depth-penetration-experiments)
  - [10.1 Goal](#101-goal)
  - [10.2 Main models](#102-main-models)
  - [10.3 Why this branch exists](#103-why-this-branch-exists)
  - [10.4 HTML explorers](#104-html-explorers)
- [11. How To Read The Repo](#11-how-to-read-the-repo)
- [12. Short Summary Of Intent](#12-short-summary-of-intent)

## 1. High-Level Goal

The main modeling goal in this repository is:

- build a cheap surrogate from LF data
- anchor and correct it using HF data
- use MF-GP to combine the two fidelities

The broad pattern is:

1. define a theta-dependent target
2. create LF and HF files for many theta values
3. train a CNP to estimate the target from event-level information
4. aggregate the CNP outputs into theta-level `y_cnp`
5. combine `y_cnp` and `y_raw` with MF-GP

So the code is mostly about:

- how theta is defined
- how LF/HF trials are built
- how the CNP is trained
- how MF-GP is fit on the aggregated outputs

## 2. Standard Pipeline

The standard pipeline is the original cumulative-theta workflow.

### 2.1 Data preparation

Main files:

- [`prepare_resum_data.py`](prepare_resum_data.py)
- [`xlzd_resum/config.py`](xlzd_resum/config.py)
- [`xlzd_resum/io_utils.py`](xlzd_resum/io_utils.py)
- [`xlzd_resum/theta.py`](xlzd_resum/theta.py)
- [`xlzd_resum/dataset.py`](xlzd_resum/dataset.py)

What this stage does:

1. load the raw XLZD component event files
2. compute derived geometry columns:
   - `r = sqrt(x^2 + y^2)`
   - `s_r = sqrt(sx^2 + sy^2)`
   - `z_from_center`
   - `s_z_from_center`
3. split raw events into three disjoint pools:
   - LF training
   - HF training
   - HF validation
4. split each pool into event blocks
5. assign theta values to those blocks
6. write per-block CSV/parquet files

### 2.2 Standard theta definition

In the standard workflow:

- `theta = (R_max, Z_max)`

An event is labeled:

- `inside_theta = 1` if:
  - `r <= R_max`
  - `z_from_center <= Z_max`

This means theta defines a centered cumulative detector volume.

### 2.3 Why the standard theta is useful

It is simple and stable.

It gives:

- a clear binary target
- easy event labeling
- a smooth cumulative response surface

### 2.4 Limitation of the standard theta

The main limitation is inclusivity:

- larger theta volumes automatically include smaller ones
- theta targets become nested and strongly correlated

So the model is learning a cumulative occupancy surface, not a local spatial response.

## 3. H5 Conversion

Main file:

- [`convert_csv_to_h5_xlzd.py`](convert_csv_to_h5_xlzd.py)

Purpose:

- convert the generated per-theta CSV/parquet files into HDF5
- H5 is what the CNP pipeline reads directly

Each H5 file stores:

- `theta`
- `phi`
- `target`
- `weights`
- `fidelity`

## 4. CNP Stage

Main files:

- [`src/run_cnp/cnp_clean_pipeline.py`](src/run_cnp/cnp_clean_pipeline.py)
- [`src/run_cnp/cnp_xlzd_workflow.ipynb`](src/run_cnp/cnp_xlzd_workflow.ipynb)
- [`src/run_cnp/cnp_predict_per_signal.py`](src/run_cnp/cnp_predict_per_signal.py)

### 4.1 What the CNP is doing

The CNP works at the event level.

Inputs:

- theta features
- phi features

For the standard pipeline this is typically:

- theta:
  - `R_max`
  - `Z_max`
- phi:
  - `s_r`
  - `s_z_from_center`

Target:

- `inside_theta`

The CNP learns:

- probability that an event belongs to the theta-defined signal region

### 4.2 Why the CNP exists here

The CNP is used to generate a cheap LF estimate:

- event-level predicted probabilities
- aggregated into theta-level `y_cnp`

So the CNP is the LF surrogate model.

### 4.3 Outputs of the CNP stage

Main outputs:

- trained model checkpoint
- training history CSV/plots
- aggregated prediction CSVs:
  - training LF/HF aggregated output
  - validation HF aggregated output

Optional output:

- per-signal/per-event CNP predictions via
  - [`src/run_cnp/cnp_predict_per_signal.py`](src/run_cnp/cnp_predict_per_signal.py)

That per-signal export became important later for LF augmentation experiments.

## 5. MF-GP Stage

Main files:

- [`src/run_mfgp/mfgp_clean_pipeline.py`](src/run_mfgp/mfgp_clean_pipeline.py)
- [`src/run_mfgp/mfgp_xlzd_workflow.ipynb`](src/run_mfgp/mfgp_xlzd_workflow.ipynb)

### 5.1 What MF-GP is doing

The MF-GP stage works on aggregated theta-level quantities.

Inputs:

- LF rows:
  - theta
  - `y_cnp`
  - `y_cnp_err`
- HF rows:
  - theta
  - `y_raw`

Model:

- autoregressive two-level GP
- LF GP on `y_cnp`
- discrepancy GP for HF correction

### 5.2 Why MF-GP exists here

It lets the pipeline:

- use lots of cheap LF information
- use a smaller amount of expensive HF truth
- estimate uncertainty on the final theta-level response surface

### 5.3 Target transforms

The MF-GP workflow supports several target transforms:

- `linear`
- `log_hf`
- `log_lf`
- `log_both`

The purpose of these is to test whether log-scaling improves behavior when targets are small or highly skewed.

## 6. Different CNP Training Modes

Folder:

- [`src/run_cnp/`](src/run_cnp)

Main notebook:

- [`src/run_cnp/cnp_xlzd_workflow.ipynb`](src/run_cnp/cnp_xlzd_workflow.ipynb)

Archived notebooks:

- [`src/run_cnp/additional_experiments/`](src/run_cnp/additional_experiments)

The main notebook supports:

- `default`
- `minibatch`
- `fixedcontext`
- `fullpass`

Purpose of these modes:

- compare training schemes
- compare context selection behavior
- see whether different batching affects CNP quality

The archived fixed-path notebooks are older convenience copies of those runs. The switchable main notebook is now the primary entry point.

## 7. LF Augmentation Experiments

Folder:

- [`lf_augmentations/`](lf_augmentations)

Main files:

- [`lf_augmentations/lf_augmentation.py`](lf_augmentations/lf_augmentation.py)
- [`lf_augmentations/00_lf_augmentation_workflow.ipynb`](lf_augmentations/00_lf_augmentation_workflow.ipynb)
- [`lf_augmentations/01_lf_augmentation_mfgp.ipynb`](lf_augmentations/01_lf_augmentation_mfgp.ipynb)

Main variations:

- bootstrap
  - resamples per-event CNP outputs to create more LF trials at the same theta
- merged
  - combines LF trial summaries into smoother synthetic LF trials at the same theta

### 7.1 Motivation

These experiments came from the question:

- is the current bottleneck the LF side rather than the HF side?

In particular:

- HF data already exists
- LF trial count may be too sparse
- can more LF support improve MF-GP without adding new HF truth?

### 7.2 Important idea

The experiment does **not** invent new HF truth.

It only creates more LF-style aggregated trials from the trained CNP outputs.

So the question being tested is:

- does a denser LF layer help MF-GP?

### 7.3 What bootstrap augmentation does

Bootstrap augmentation works at the event level.

For one theta:

1. take per-signal CNP predictions
2. resample those events with replacement
3. average them into synthetic LF trials

This tests:

- do we just need more LF replications at the same theta?

### 7.4 What merged augmentation does

Merged augmentation works at the trial level.

It combines LF trial summaries into smoother synthetic LF trials.

This tests:

- is the LF estimate too noisy, even if the theta support is the same?

### 7.5 What these experiments are trying to learn

They are not meant as final production pipelines.

They are trying to diagnose:

- whether MF-GP is limited by LF sparsity
- whether more LF rows help
- whether smoothing LF support helps

### 7.6 Caveat

These augmentations can reduce MF-GP uncertainty without genuinely adding new physical information.

So they are best interpreted as:

- diagnostic experiments
- not automatic improvements

## 8. Theta Augmentation Experiments

Folder:

- [`theta_augmentations/`](theta_augmentations)

Subfolders:

- [`theta_augmentations/local_jitter/`](theta_augmentations/local_jitter)
- [`theta_augmentations/midpoint/`](theta_augmentations/midpoint)

Main variations:

- local jitter
  - adds nearby synthetic theta values by small random perturbations around existing LF theta points
- midpoint
  - adds synthetic theta values halfway between neighboring LF theta points

### 8.1 Motivation

LF augmentation at existing thetas does not increase theta coverage.

So the next question was:

- maybe the real issue is sparse LF coverage in theta space

These experiments test that idea.

### 8.2 Local jitter

For each original LF theta:

- create a nearby synthetic theta by a small random perturbation
- recompute event labels for that theta
- rerun the trained CNP
- aggregate the new LF trial

This tests:

- does denser local LF coverage help MF-GP?

### 8.3 Midpoint

For nearby LF thetas:

- create midpoint theta values
- recompute event labels there
- rerun the trained CNP
- build new LF rows

This tests:

- does filling LF gaps between existing thetas help MF-GP?

### 8.4 What these experiments are trying to learn

They are exploring whether:

- LF support is too sparse in theta space
- MF-GP would improve if the LF surface were denser

Again, these are exploratory branches, not the main pipeline.

## 9. Shell Theta Experiment

Folder:

- [`xlzd_shell_theta/`](xlzd_shell_theta)

Main files:

- [`xlzd_shell_theta/prepare_shell_theta_data.py`](xlzd_shell_theta/prepare_shell_theta_data.py)
- [`xlzd_shell_theta/convert_shell_theta_to_h5.py`](xlzd_shell_theta/convert_shell_theta_to_h5.py)
- [`xlzd_shell_theta/00_shell_theta_cnp_workflow.ipynb`](xlzd_shell_theta/00_shell_theta_cnp_workflow.ipynb)
- [`xlzd_shell_theta/01_shell_theta_mfgp.ipynb`](xlzd_shell_theta/01_shell_theta_mfgp.ipynb)

### 9.1 Motivation

The standard cumulative theta has the inclusivity problem:

- larger theta includes smaller theta
- target is cumulative
- local spatial structure is hard to isolate

Shell theta was introduced to remove that built-in nesting.

### 9.2 Initial shell-theta definition

Initial shell theta was defined as:

- `theta = (r_shell, z_shell)`

Target:

- event is near the shell-centered neighborhood

Initially this was a hard-box target:

- `|r - r_shell| <= delta_r`
- `|z_from_center - z_shell| <= delta_z`

So shell theta asks:

- how many events land near this local region?

instead of:

- how many events are inside everything up to this region?

### 9.3 Why shell theta is conceptually useful

It makes theta more local and less nested.

That means:

- different theta values are less dependent
- the model is forced to learn local response structure
- the response surface is more informative spatially

### 9.4 Why shell theta is harder

It also makes the target:

- sparser
- noisier
- more uncertain away from HF anchors

So shell theta is more informative, but statistically harder than cumulative theta.

### 9.5 Variation experiments for shell theta

Folder:

- [`xlzd_shell_theta/variations/`](xlzd_shell_theta/variations)

These were introduced to test whether shell-theta behavior improves if the shell construction is changed.

Implemented variations:

1. larger shell width
2. smaller theta grid spacing
3. asymmetric shell width
4. higher support threshold
5. soft Gaussian shell target

What each one does:

- larger shell width
  - increases `delta_r` and `delta_z` so each shell target is less sparse
- smaller theta grid spacing
  - decreases `r_shell_step` and `z_shell_step` so theta coverage is denser
- asymmetric shell width
  - uses different `delta_r` and `delta_z` to test radial vs axial locality separately
- higher support threshold
  - increases `min_candidate_events` so weak shell-theta candidates are filtered out
- soft Gaussian shell target
  - replaces the hard `0/1` near-shell label with a smooth proximity score

The purpose of these is to test whether:

- the base shell target is too sparse
- the theta grid is too coarse
- the hard 0/1 shell label is too sharp

### 9.6 Soft shell target

The soft shell variation changes the meaning of the target.

Instead of:

- `near_shell = 0/1`

it uses a smooth Gaussian proximity score.

This is meant to:

- smooth the target
- reduce harsh label boundaries
- make CNP and MF-GP behavior more stable

## 10. Depth Penetration Experiments

Folder:

- [`depth_penetration_experiments/`](depth_penetration_experiments)

This is a different line of work from the theta/CNP/MF-GP experiments.

### 10.1 Goal

Instead of modeling occupancy under a theta definition, these experiments ask:

- how deeply do events penetrate toward the TPC center?

### 10.2 Main models

Implemented notebooks:

- global `d_center` model
- grouped axial/radial model
- deterministic MLP baseline
- MLP + VBLL head

### 10.3 Why this branch exists

This is a physics-side ranking and interpretation branch.

It is useful for:

- identifying components with deeper penetration
- comparing uncertainty-aware and deterministic regressors
- building visualization tools for component behavior

### 10.4 HTML explorers

Several Plotly HTML generators were added for this branch.

These are for:

- 2D and 3D spatial exploration
- prediction maps
- component-level comparison

## 11. How To Read The Repo

If someone is trying to understand the codebase for the first time, the best reading order is:

1. [`README.md`](README.md)
2. [`prepare_resum_data.py`](prepare_resum_data.py)
3. [`xlzd_resum/dataset.py`](xlzd_resum/dataset.py)
4. [`src/run_cnp/cnp_clean_pipeline.py`](src/run_cnp/cnp_clean_pipeline.py)
5. [`src/run_mfgp/mfgp_clean_pipeline.py`](src/run_mfgp/mfgp_clean_pipeline.py)

Then branch into:

- [`lf_augmentations/`](lf_augmentations)
- [`theta_augmentations/`](theta_augmentations)
- [`xlzd_shell_theta/`](xlzd_shell_theta)
- [`depth_penetration_experiments/`](depth_penetration_experiments)

## 12. Short Summary Of Intent

The repository now contains several layers of work:

- the original cumulative-theta LF/HF CNP + MF-GP pipeline
- experiments to densify LF support
- experiments to change theta geometry
- a separate shell-theta branch to reduce cumulative nesting
- a separate penetration-modeling branch for detector interpretation

So the codebase is no longer just one pipeline. It is a main pipeline plus several experiment branches that explore:

- how theta should be defined
- how LF should be strengthened
- how uncertainty behaves
- how event geometry should be modeled
