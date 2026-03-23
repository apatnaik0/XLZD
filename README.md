# XLZD Pool/Block Theta Data Preparation

XLZD is a multi-purpose liquid-xenon rare-event observatory with physics goals including dark matter, neutrinoless double-beta decay, and neutrino studies.

This codebase prepares XLZD background simulation data for the current pool/block-based spatial workflow.

The input files are already preselected:

- narrow ROI around the Q-value
- single-site only
- veto already applied

## Raw Data Columns

Each row is one event with:

- `global_event_id`
- `E0`
- `sx`, `sy`, `sz`
- `ETPC`
- `x`, `y`, `z`

Interpretation:

- `global_event_id`: unique event identifier
- `E0`: initial energy
- `sx`, `sy`, `sz`: initial coordinates
- `ETPC`: energy deposited in the TPC
- `x`, `y`, `z`: final coordinates

The loader preserves the first column as `global_event_id`.

## Final Position Coordinates

The code computes:

- `r = sqrt(x^2 + y^2)`

and also computes a centered axial coordinate:

- `z_from_center = abs(z - z_center)`

where:

- `z_center` is inferred by default as the midpoint of the observed `z` range
- or can be provided explicitly in the config

This is used because theta is defined relative to the chamber center in the current workflow.

From inspecting the raw files:

- `x` and `y` both take positive and negative values and behave like centered transverse coordinates
- `z` runs approximately from `0` to the chamber extent rather than being centered around `0`

So the code treats raw `z` as an axial coordinate and derives a centered quantity:

- `z_from_center = abs(z - z_center)`

Reference material:

- XLZD collaboration site: https://xlzd.org/
- XLZD design report: https://link.springer.com/article/10.1140/epjc/s10052-025-14810-w
- STFC XLZD overview: https://www.ppd.stfc.ac.uk/Pages/XLZD.aspx

## Centered Theta Definition

Theta is centered and uses only two parameters:

- `R_max`
- `Z_max`

An event is inside theta if:

- `r <= R_max`
- `z_from_center <= Z_max`

So theta defines a centered detector volume grown outward from the chamber center.

Theta in this workflow is only this 2D pair:

- `(R, Z) = (R_max, Z_max)`

There is no separate theta identifier in the generated data files.

## Three Disjoint Pools

The full dataset is shuffled once and split into three non-overlapping pools:

- LF training pool
- HF training pool
- HF validation pool

By default the proportions are:

- LF training pool: `20%`
- HF training pool: `40%`
- HF validation pool: remaining `40%`

These are configurable.

## Approximate Counts With The Current Dataset

Using the original seven-file dataset counts discussed during development, the total event count is approximately:

- `5,325,583` events

With the current default pool proportions:

- LF training pool (`20%`): about `1.07M` events
- HF training pool (`40%`): about `2.13M` events
- HF validation pool (remainder): about `2.13M` events

These are planning figures. The exact counts depend on the exact dataset size at runtime.

## Blocks

Each pool is split into equal-size blocks:

- LF training blocks of size `10,000`
- HF training blocks of size `100,000`
- HF validation blocks of size `100,000`

These block sizes are configurable. Extra rows are spread across the blocks so almost all available events are used.

With the current defaults and a dataset of about `5.33M` events, the expected file counts are approximately:

- `n ≈ 106` LF training files
- `m ≈ 21` HF training files
- `k ≈ 21` HF validation files

## Example Run Summary

With the current full dataset and default settings, a representative run produced:

| Quantity | Value |
| --- | --- |
| Total events | `5,325,583` |
| Inferred `z_center` | `1982.48` |
| Final-position `z` range | `[0.112398, 3964.85]` |
| Final-position `r` range | `[1.22868, 1489.87]` |

| Pool | Size |
| --- | ---: |
| LF training | `1,065,116` |
| HF training | `2,130,233` |
| HF validation | `2,130,234` |

| Split | Files | Block size range | Unused leftover rows |
| --- | ---: | --- | ---: |
| LF training | `106` | `10048-10049` | `0` |
| HF training | `21` | `101439-101440` | `0` |
| HF validation | `21` | `101439-101440` | `0` |

| Generated File Statistic | Value |
| --- | ---: |
| Number of generated files | `148` |
| Mean sample size | `35983.67` |
| Std sample size | `41342.19` |
| Min sample size | `10048` |
| Max sample size | `101440` |
| Mean `inside_theta_count` | `203.43` |
| Std `inside_theta_count` | `652.82` |
| Max `inside_theta_count` | `3604` |
| Mean `inside_theta_fraction` | `0.006648` |
| Std `inside_theta_fraction` | `0.021646` |
| Max `inside_theta_fraction` | `0.194945` |

## How Theta Values Are Assigned

Let:

- `n` = number of LF blocks
- `m` = number of HF training blocks
- `k` = number of HF validation blocks

Then:

- sample `n` LF theta values from the LF support
- choose `m` of those LF theta values for the HF training blocks
- assign HF-based theta values to the `k` validation blocks

This preserves the important rule:

- every HF theta is also present in LF

while allowing validation to be held out in a separate pool.

## What Each Output File Contains

Each LF/HF/validation block file contains:

- the sampled event rows from that block
- original event columns
- `r`
- `z_from_center`
- `source_component`
- `source_file`
- theta metadata:
  - `R_max`
  - `Z_max`
  - `split_name`
  - `fidelity`
- `inside_theta`

where:

- `inside_theta = 1` if the event is inside the centered theta region
- `inside_theta = 0` otherwise

## Code Layout

- [prepare_resum_data.py](/Users/anishpatnaik/Documents/XLZD/prepare_resum_data.py): runnable entrypoint, config loading, pool/block pipeline, and summary printing
- [xlzd_resum/config.py](/Users/anishpatnaik/Documents/XLZD/xlzd_resum/config.py): config dataclasses
- [xlzd_resum/io_utils.py](/Users/anishpatnaik/Documents/XLZD/xlzd_resum/io_utils.py): robust input loading and saving
- [xlzd_resum/theta.py](/Users/anishpatnaik/Documents/XLZD/xlzd_resum/theta.py): centered theta definitions and `z_from_center` logic
- [xlzd_resum/dataset.py](/Users/anishpatnaik/Documents/XLZD/xlzd_resum/dataset.py): pool splitting, block splitting, theta assignment, and file writing
- [config/pipeline_config.json](/Users/anishpatnaik/Documents/XLZD/config/pipeline_config.json): main config
- [config/pipeline_config_smoke.json](/Users/anishpatnaik/Documents/XLZD/config/pipeline_config_smoke.json): smoke-test config

## How To Run

Normal run:

```bash
python3 prepare_resum_data.py
```

Smoke test:

```bash
python3 prepare_resum_data.py --config config/pipeline_config_smoke.json
```

## Main Config Fields

Edit [config/pipeline_config.json](/Users/anishpatnaik/Documents/XLZD/config/pipeline_config.json).

Most important fields:

- `file_load.input_dir`
- `file_load.max_rows_per_file`
- `split.lf_pool_fraction`
- `split.hf_pool_fraction`
- `split.random_seed`
- `theta.z_center`
- `theta.z_lower`, `theta.z_upper`
- `theta.r_lower`, `theta.r_upper`
- `theta.min_z_width`, `theta.min_r_width`
- `sampling.lf_block_size`
- `sampling.hf_block_size`
- `sampling.validation_block_size`
- `output.output_dir`
- `output.output_format`

## Progress During Run

The script prints stage-level progress with elapsed time for:

- file loading and normalization
- centered-z computation
- disjoint pool splitting
- equal-size block splitting
- theta generation
- LF block writing
- HF training block writing
- HF validation block writing
- output writing

It also shows progress bars for writing the LF, HF training, and HF validation block files.

## Outputs

The script writes:

- `processed_all_events.csv` or `.parquet`
- `lf_training_pool.csv` or `.parquet`
- `hf_training_pool.csv` or `.parquet`
- `hf_validation_pool.csv` or `.parquet`
- `training/lf/*.csv` or `.parquet`
- `training/hf/*.csv` or `.parquet`
- `validation/hf/*.csv` or `.parquet`
- `theta_file_manifest.csv` or `.parquet`

### Pool Files

These contain the disjoint raw-event pools:

- `lf_training_pool`
- `hf_training_pool`
- `hf_validation_pool`

### Per-Block Files

These are the per-theta block files:

- `training/lf/`
- `training/hf/`
- `validation/hf/`

Filename style:

- `lf_R123p456_Z789p000.csv`
- `hf_R123p456_Z789p000.csv`

### Manifest

The manifest has one row per generated block file and includes:

- `block_index`
- `random_seed_used`
- `split_name`
- `fidelity`
- `file_name`
- `file_path`
- `sample_size`
- `R_max`
- `Z_max`
- `inside_theta_count`
- `inside_theta_fraction`

## Assumptions And Places To Modify

### File parsing assumptions

- the first column is preserved as `global_event_id`
- the remaining 8 physics columns are parsed as numeric
- malformed rows are dropped with a warning instead of aborting the entire run

### Changing pool proportions

Edit:

- `split.lf_pool_fraction`
- `split.hf_pool_fraction`

The validation pool is the remainder.

### Changing block sizes

Edit:

- `sampling.lf_block_size`
- `sampling.hf_block_size`
- `sampling.validation_block_size`

### Changing centered theta behavior

Edit:

- `theta.z_center` if you want to force a chamber center
- `theta.z_lower`, `theta.z_upper`
- `theta.r_lower`, `theta.r_upper`

### Per-component processing

The code still preserves `source_component`.

If you later want to run the same pool/block logic separately per component:

- start from `load_event_collection(...)`
- loop over `loaded.per_component.items()`
- run the same pool/block pipeline separately on each component dataframe
