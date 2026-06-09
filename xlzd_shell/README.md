# Cylindrical Shell Theta

This folder contains a centered cylindrical shell-theta experiment that is separate from the original cumulative-volume theta pipeline and separate from the earlier local shell-grid experiment in [`xlzd_shell_theta`](/Users/anishpatnaik/Documents/RareAI%20Lab/XLZD/xlzd_shell_theta).

The purpose of this experiment is to build shell regions that:

- are centered on the TPC center
- cover the detector from inner to outer regions
- have equal shell-band volume
- can be used as LF/HF theta values for CNP and MF-GP

The target in this experiment is:

- `inside_shell = 1` if an event lies within a specific shell band
- `inside_shell = 0` otherwise

This is a shell-band occupancy experiment, not a shell-surface proximity experiment.

## Why This Experiment Exists

The original cumulative theta setup uses:

- `theta = (R_max, Z_max)`
- target = fraction of events inside that cumulative centered volume

That construction has a built-in inclusivity problem:

- larger theta values always include smaller theta values
- target values are nested and strongly correlated by construction

The earlier shell-grid experiment in [`xlzd_shell_theta`](/Users/anishpatnaik/Documents/RareAI%20Lab/XLZD/xlzd_shell_theta) improved locality, but the shell candidates were still based on local shell points. That can leave the inner part of the TPC underrepresented, especially on the HF side.

This equal-volume shell experiment is trying to fix that by making the shell family itself centered and geometrically balanced.

## Core Geometry

We define a family of centered cylindrical shell boundaries.

Each boundary is described by:

- outer radius `R_i`
- outer axial extent `Z_i`

The family is constrained to keep a fixed aspect ratio:

- `alpha = Z_max / R_max`
- `Z_i = alpha * R_i`

So as the shell family expands outward:

- `R` increases
- `Z` increases proportionally

This means the shell boundary points lie on a straight line in `(R, Z)` space. That is intentional in the current construction. The experiment is not varying shell shape at fixed volume. It is varying shell scale while keeping shell shape fixed.

## Equal-Volume Shell Math

The centered cylindrical volume enclosed by boundary `i` is proportional to:

- `V_i = 2 * pi * Z_i * R_i^2`

Substitute `Z_i = alpha * R_i`:

- `V_i = 2 * pi * alpha * R_i^3`

Now define shell band `i` as the region between:

- inner boundary `(R_{i-1}, Z_{i-1})`
- outer boundary `(R_i, Z_i)`

Then shell-band volume is:

- `V_shell,i = V_i - V_{i-1}`
- `V_shell,i = 2 * pi * alpha * (R_i^3 - R_{i-1}^3)`

To force every shell band to have the same volume, require:

- `R_i^3 - R_{i-1}^3 = constant`

If the full outer boundary is `(R_max, Z_max)` and we use `n_shells` shell bands, then:

- `R_i^3 - R_{i-1}^3 = R_max^3 / n_shells`

This gives the closed-form shell boundaries:

- `R_i = R_max * (i / n_shells)^(1/3)`
- `Z_i = Z_max * (i / n_shells)^(1/3)`

for `i = 0, 1, ..., n_shells`.

So:

- shell 1 is thick near the center
- outer shells become progressively thinner

That is the main mechanism that gives the inner TPC region more representation.

## Normalized Thickness Interpretation

If you normalize by the full detector scale, the outer boundary of shell `i` sits at:

- `(i / n_shells)^(1/3)`

So the normalized thickness of shell `i` is:

- `(i / n_shells)^(1/3) - ((i - 1) / n_shells)^(1/3)`

These thicknesses sum to `1` by construction.

This is why the first few shells are relatively thick and outer shells are thin.

## Target Construction

Each event already has final-position coordinates:

- `r`
- `z_from_center`

For shell `i`, define:

- `inside_outer = (r <= R_i) and (z_from_center <= Z_i)`
- `inside_inner = (r <= R_{i-1}) and (z_from_center <= Z_{i-1})`

Then:

- `inside_shell = inside_outer and not inside_inner`

This means the event belongs to the shell band itself, not just the shell surface.

That is an important design choice.

This experiment does **not** use:

- distance-to-shell
- tolerance around shell surface
- shell-surface proximity score

It uses shell-band occupancy directly.

## What The Model Learns

The CNP sees:

- `theta = (R_shell, Z_shell)` where these are the outer shell boundaries
- `phi = (s_r, s_z_from_center)` as event-level source geometry features
- target = `inside_shell`

The CNP therefore learns:

- probability that an event belongs to that equal-volume shell band

Then MF-GP learns a low/high-fidelity response surface over shell-theta space using the CNP outputs.

## Why This Might Help With Inner HF Coverage

In the earlier shell experiments, inner regions could be starved because:

- shell candidates near the center had very little support
- or many candidate regions were effectively too small or too sparse

In this equal-volume shell-band construction:

- inner shells are thicker
- inner shell bands can capture more events
- candidate shells are distributed across shell scale more evenly

So the experiment is trying to improve:

- availability of valid inner shell candidates
- chance that HF training and validation files occupy inner shell bands

This does not guarantee strong inner HF coverage, but it makes it more likely than a shell construction where inner regions are too thin.

## Pipeline Files

- [prepare_equal_volume_shell_data.py](/Users/anishpatnaik/Documents/RareAI%20Lab/XLZD/xlzd_equal_volume_shell_theta/prepare_equal_volume_shell_data.py)
  - loads raw events
  - splits them into LF/HF/validation pools
  - creates equal-volume shell candidates
  - assigns shells to LF/HF/validation files
  - writes shell-band CSV/parquet files
- [convert_equal_volume_shell_to_h5.py](/Users/anishpatnaik/Documents/RareAI%20Lab/XLZD/xlzd_equal_volume_shell_theta/convert_equal_volume_shell_to_h5.py)
  - converts shell-band files into H5 for the CNP code
- [run_equal_volume_shell.py](/Users/anishpatnaik/Documents/RareAI%20Lab/XLZD/xlzd_equal_volume_shell_theta/run_equal_volume_shell.py)
  - convenience driver that runs preprocess + H5 conversion
- [settings_equal_volume_shell_minibatch.yaml](/Users/anishpatnaik/Documents/RareAI%20Lab/XLZD/xlzd_equal_volume_shell_theta/settings_equal_volume_shell_minibatch.yaml)
  - CNP/MF-GP training config
- [settings_equal_volume_shell_validation_minibatch.yaml](/Users/anishpatnaik/Documents/RareAI%20Lab/XLZD/xlzd_equal_volume_shell_theta/settings_equal_volume_shell_validation_minibatch.yaml)
  - validation prediction config
- [predict_cnp_on_csv.py](/Users/anishpatnaik/Documents/RareAI%20Lab/XLZD/xlzd_equal_volume_shell_theta/predict_cnp_on_csv.py)
  - runs a trained CNP checkpoint on a new source/event CSV
  - uses existing empirical H5 files as the CNP context set
  - writes event-wise CNP probabilities without rerunning the full pipeline
- [00_equal_volume_shell_workflow.ipynb](/Users/anishpatnaik/Documents/RareAI%20Lab/XLZD/xlzd_equal_volume_shell_theta/00_equal_volume_shell_workflow.ipynb)
  - single baseline notebook with:
    - notebook-parameter override
    - dataset generation
    - CNP training/prediction
    - MF-GP fitting and plots

## Default Run Flow

The intended flow is:

1. Open [00_equal_volume_shell_workflow.ipynb](/Users/anishpatnaik/Documents/RareAI%20Lab/XLZD/xlzd_equal_volume_shell_theta/00_equal_volume_shell_workflow.ipynb)
2. Edit the notebook parameters
3. Run all cells

The notebook writes a runtime config to:

- [runtime_notebook_config.json](/Users/anishpatnaik/Documents/RareAI%20Lab/XLZD/xlzd_equal_volume_shell_theta/config/runtime_notebook_config.json)

and then runs:

- [run_equal_volume_shell.py](/Users/anishpatnaik/Documents/RareAI%20Lab/XLZD/xlzd_equal_volume_shell_theta/run_equal_volume_shell.py)

using that generated config.

The dataset directory is always:

- `outputs_equal_volume_shell_theta`

and the prep step now clears and rebuilds that folder each time. That avoids filling disk with stale experiment copies.

## Predicting On A New CSV With A Trained CNP

Use `predict_cnp_on_csv.py` when you already have a trained CNP model and want event-wise predictions for a new CSV of source points.

This script does not train a model and does not rebuild the shell dataset. It:

- reads a source/event CSV
- derives or reads the CNP phi columns, usually `s_r` and `s_z_from_center`
- attaches one or more theta queries, usually `R_shell` and `Z_shell`
- samples empirical context rows from the existing H5 training directory
- writes event-wise `y_cnp` and `y_cnp_err`

Example for one shell-theta query:

```bash
python xlzd_equal_volume_shell_theta/predict_cnp_on_csv.py \
  --input-csv path/to/source_points.csv \
  --output-csv data/out/cnp/source_points_shell750_1000_predictions.csv \
  --config xlzd_equal_volume_shell_theta/settings_equal_volume_shell_minibatch.yaml \
  --theta 750 1000 \
  --overwrite
```

If the model checkpoint is not at the default config-derived location, pass it explicitly:

```bash
python xlzd_equal_volume_shell_theta/predict_cnp_on_csv.py \
  --input-csv path/to/source_points.csv \
  --output-csv data/out/cnp/source_points_predictions.csv \
  --config xlzd_equal_volume_shell_theta/settings_equal_volume_shell_minibatch.yaml \
  --model-path data/out/cnp/cnp_xlzd_equal_volume_shell_v1_minibatch_model_15epochs.pth \
  --theta 750 1000 \
  --overwrite
```

For multiple theta queries, create a CSV with columns matching the config theta headers:

```text
R_shell,Z_shell
500,667
750,1000
1000,1333
```

Then run:

```bash
python xlzd_equal_volume_shell_theta/predict_cnp_on_csv.py \
  --input-csv path/to/source_points.csv \
  --output-csv data/out/cnp/source_points_multi_shell_predictions.csv \
  --theta-csv path/to/theta_queries.csv \
  --overwrite
```

The input CSV can provide `s_r` and `s_z_from_center` directly. If not, the script tries to derive them from common source coordinate columns:

- `s_r` from `s_r`, `r_start`, `source_r`, `r`, or from `sx/sy`
- `s_z_from_center` from `s_z_from_center`, `z_start_from_center`, `source_z_from_center`, `z_from_center`, or from `sz - z_center`

For source CSVs with columns such as:

```text
sx,sy,sz,E0,ETPC,x,y,z
```

the script uses:

- `s_r = sqrt(sx^2 + sy^2)`
- `s_z_from_center = abs(sz - z_center)`

The `x,y,z` columns are not treated as predicted outputs by this CNP. They are ignored unless you explicitly include them in the output with `--include-input-columns`.

To scan every equal-volume shell theta instead of passing one theta manually:

```bash
python xlzd_equal_volume_shell_theta/predict_cnp_on_csv.py \
  --input-csv path/to/source_points.csv \
  --output-csv data/out/cnp/source_points_all_shell_predictions.csv \
  --config xlzd_equal_volume_shell_theta/settings_equal_volume_shell_minibatch.yaml \
  --all-shells \
  --include-input-columns \
  --overwrite
```

`--all-shells` reads `R_max`, `Z_max`, and `n_shells` from `xlzd_equal_volume_shell_theta/config/pipeline_config.json` unless you override them with `--R-max`, `--Z-max`, or `--n-shells`.

Important caveat: the CNP needs a context set. This script uses empirical H5 training files as the context, so synthetic CSV rows do not need known targets and do not contaminate the context.

## Notebook Parameters

The main parameters exposed in the notebook are:

- `R_max`
- `Z_max`
- `n_shells`
- `min_candidate_events`
- `lf_block_size`
- `hf_block_size`
- `validation_block_size`

### `R_max`

This is the outer radial extent of the full shell family.

Increasing it:

- pushes the shell family farther outward
- may include shells closer to detector limits

Decreasing it:

- truncates the outer shell family
- focuses the experiment on a smaller radial region

Usually this should stay tied to the TPC geometry unless you are intentionally changing the physical outer boundary.

### `Z_max`

This is the outer axial extent of the full shell family.

Increasing it:

- pushes shells farther in `z`
- changes the shell aspect ratio only if `R_max` stays fixed

Decreasing it:

- compresses the shell family axially

Like `R_max`, this is usually a geometry parameter, not a tuning knob for routine sweeps.

### `n_shells`

This is the most important resolution parameter.

Increasing `n_shells`:

- creates thinner shell bands
- increases geometric resolution
- usually decreases per-shell support
- tends to increase sparsity

Decreasing `n_shells`:

- creates thicker shell bands
- reduces geometric resolution
- increases support per shell
- makes the target less sparse

If the shell occupancy is too sparse, reducing `n_shells` is one of the first things to try.

### `min_candidate_events`

A shell band is considered valid only if the LF support pool contains at least this many events inside it.

Increasing `min_candidate_events`:

- removes weakly supported shells
- makes surviving shells more reliable
- can eliminate inner/sparse shells
- reduces the number of valid candidates

Decreasing `min_candidate_events`:

- allows more shell candidates to survive
- increases shell diversity
- helps keep inner shells
- can admit noisier shells

This is usually the first parameter to lower if the experiment is not producing enough inner shell candidates.

Setting it to `0` is generally not a good idea because it allows completely empty shell candidates.

### `lf_block_size`

This controls how many LF files are created.

Increasing `lf_block_size`:

- creates fewer LF files
- reduces the number of LF shell assignments required
- makes each LF file stronger
- usually reduces the chance that file count must be shrunk

Decreasing `lf_block_size`:

- creates more LF files
- requires more LF shell assignments
- increases pressure on shell diversity

If you have too many LF files relative to valid shell candidates, increasing `lf_block_size` is the cleanest fix.

### `hf_block_size`

This controls how many HF training files are created.

Increasing `hf_block_size`:

- creates fewer HF training files

Decreasing `hf_block_size`:

- creates more HF training files

Usually this is less sensitive than LF file count, but it affects how many HF anchors MF-GP has.

### `validation_block_size`

If this is `None`, validation uses the same block size as HF training.

Decreasing it:

- creates more validation files

Increasing it:

- creates fewer validation files

Validation file count matters because it affects how many unique validation shell assignments are needed.

## What Happens If There Are Not Enough Valid Shell Candidates

This is one of the key design details in the current implementation.

The number of valid shell candidates is determined by:

- shell geometry
- `n_shells`
- `min_candidate_events`
- LF support density

The number of files that need shell assignments is determined by:

- LF pool size and `lf_block_size`
- HF pool size and `hf_block_size`
- validation pool size and `validation_block_size`

If there are fewer valid shell candidates than the requested LF + validation file counts, the current code uses **shrinkage**, not replacement.

That means:

- LF file count is reduced
- validation file count is reduced
- HF file count is reduced proportionally to keep the LF/HF ratio roughly consistent

The code then writes only that reduced number of files.

It does **not** currently:

- reuse shell candidates across multiple files
- invent unsupported shell candidates

This is the more conservative choice and is easier to interpret scientifically.

### Why Shrinkage Was Chosen

If shell candidates were reused heavily:

- the number of trials would stay high
- but theta diversity would be artificially low

Shrinkage gives:

- fewer files
- cleaner mapping from files to shell bands
- a more honest reflection of how much shell diversity the data actually supports

So the current policy is:

- preserve shell uniqueness as much as possible
- reduce file count when the candidate set is too small

## Practical Tuning Strategy

If a run produces too few valid shell candidates or shrinks too aggressively, the best adjustment order is usually:

1. lower `min_candidate_events`
2. increase `lf_block_size`
3. reduce `n_shells`

That order typically works better than changing geometry first.

## What This Experiment Is Trying To Tell You

This baseline is testing whether a centered, equal-volume shell-band theta construction:

- produces more meaningful shell support toward the inside of the TPC
- gives better-balanced shell candidates than the earlier shell-point experiment
- can support a useful LF/HF CNP + MF-GP pipeline without relying on cumulative inclusivity

If it works well, the main expected signs are:

- inner shell bands survive candidate filtering
- HF files are not concentrated only at the outermost shells
- CNP learns a less pathological target than the cumulative-volume version
- MF-GP uncertainty is not entirely dominated by outer-region shell placement

If it works poorly, the main failure modes are:

- shell occupancy remains too sparse
- candidate shell count collapses
- heavy shrinkage reduces theta coverage too much
- MF-GP becomes numerically unstable or uninformative in log space

## Current Status

The baseline currently uses:

- centered cylindrical shell bands
- equal shell volume
- fixed shell shape, variable shell scale
- binary shell-band occupancy target
- notebook-controlled parameter overrides
- dataset-folder overwrite on each prep run
- shrinkage when candidate shell count is insufficient

It does not yet include:

- shell-surface proximity targets
- varying shell shape at fixed volume
- LF augmentation
- synthetic theta augmentation

Those would be separate follow-up experiments.
