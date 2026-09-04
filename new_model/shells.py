"""
Shell specific data preparation and CNP data providers

This file has all the XLZD shell-based data handling used with the generic CNP
    Raw event files + file_manifest.csv -> labelled shell events
    LF/HF training and HF validation splitting
    Event-class HDF5 block writing
    HDF5 sampling/batch providers used by CNP

Shell Construction
    Centered on the detector center
    Outer shell boundaries follow
        R_i = R_max * (i/n_shells) ^ scale_power
        Z_i = Z_max * (i/n_shells) ^ scale_power
    Shell i is the region inside outer boundary i and outside i-1

Target shell = zero-indexed shell class label

Dataset Splits
    Fidelity = 0 is LF
    Fidelity = 1 is HF
    Validation_Fraction is fraction of HF to move to validation
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator, Optional, Sequence
from functools import partial

import numpy as np
import pandas as pd
import h5py
from tqdm.auto import tqdm

try:
    import tomllib
except:
    import tomli as tomllib

def find_repo_root(start: Path | None = None) -> Path:
    start = (start or Path.cwd()).resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "PROJECT_EXPERIMENT_GUIDE.md").exists() and (candidate / "README.md").exists():
            return candidate
    raise RuntimeError("Could not find the XLZD repo root from the current working directory.")


REPO_ROOT = find_repo_root()
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from common.config import FileLoadConfig, DEFAULT_FILE_STEMS, SamplingConfig, OutputConfig, EVENT_ID_COLUMN
from common.dataset import split_pool_into_blocks
from common.io_utils import load_event_file, save_dataframe
from common.theta import add_centered_z_coordinate, Z_FROM_CENTER_COLUMN
from common.pipeline_utils import log_stage, finish_stage

import cnp
import mfgp

# -----------------------------------------------------------------------------
# Names
# -----------------------------------------------------------------------------
TARGET_COLUMN = "target_shell"
THETA_HEADERS = ["detector_R", "detector_Z"]
PHI_HEADERS = ["s_r", "s_z_from_center"]
MANIFEST_NAME = "file_manifest.csv"

# -----------------------------------------------------------------------------
# Dataclasses
# -----------------------------------------------------------------------------
@dataclass
class HFValidationSplitConfig:
    """Config for holding out a fraction of HF events for validation"""
    validation_fraction: float = 0.4
    random_seed: int = 42

    def validate(self) -> None:
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError(f"Validation Fraction must be between 0 and 1, got {self.validation_fraction}")

@dataclass(frozen=True)
class PreparationConfig:
    file_load: FileLoadConfig
    split: HFValidationSplitConfig
    sampling: SamplingConfig
    shell: ShellConfig
    output: OutputConfig

    def validate(self) -> None:
        self.split.validate()
        for section in (self.file_load, self.sampling, self.shell, self.output):
            validate = getattr(section, "validate", None)
            if callable(validate):
                validate()

@dataclass(frozen=True)
class PreparationResult:
    """Paths produced by prepare_shell_cnp_data"""
    output_dir: Path
    training_dir: Path
    validation_dir: Path
    shell_table_path: Path
    manifest_path: Path

@dataclass(slots=True)
class ShellConfig:
    """Top level config that houses all data about shell distributions"""
    R_max: float | None = None
    Z_max: float | None = None
    n_shells: int = 100
    min_candidate_events: int = 25
    z_center: float | None = None
    scale_power: float = 1.0/3.0

    def validate(self) -> None:
        if self.R_max is not None and self.R_max <=0:
            raise ValueError("R_max must be positive")
        if self.Z_max is not None and self.Z_max <=0:
            raise ValueError("Z_max must be positive")
        if self.n_shells <= 0:
            raise ValueError("n_shells must be positive.")
        if self.min_candidate_events <= 0:
            raise ValueError("min_candidate_events must be positive.")

@dataclass(slots=True)
class ShellEventBlock:
    features: np.ndarray
    truth_shell: np.ndarray
    human_shell: np.ndarray
    event_index: np.ndarray
    valid_events: pd.DataFrame

@dataclass(frozen=True)
class ShellPredictionResults:
    """Outputs from running the CNP on a shell event pool"""
    prediction_path: Path
    n_events: int
    n_groups: int
    
# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def _validate_fidelity_series(values: pd.Series, *, context: str) -> pd.Series:
    """Return integer fidelity values and reject anything other than 0 or 1"""
    numeric = pd.to_numeric(values, errors="coerce")
    invalid_numeric = numeric.isna() | ~np.isfinite(numeric.to_numpy(dtype=float))
    non_integer = numeric.notna() & ~np.isclose(
        numeric.to_numpy(dtype=float),
        np.rint(numeric.to_numpy(dtype=float)),
    )
    invalid_binary = numeric.notna() & ~numeric.isin({0,1})
    invalid = invalid_numeric | non_integer | invalid_binary

    if invalid.any():
        bad_rows = values.index[invalid].tolist()
        bad_values = values.loc[invalid].tolist()
        raise ValueError(
            f"{context} contains invalid fidelity values at rows {bad_rows}: {bad_values}. "
            "Fidelity must be exactly 0 (low fidelity) or 1 (high fidelity)."
        )

    return numeric.astype(np.int32)

def _as_h5_array(value: object) -> np.ndarray:
    arr = np.asarray(value)
    if arr.dtype.kind in {"U", "O"}:
        arr = arr.astype("S")
    return arr

def block_range(blocks: Sequence[pd.DataFrame]) -> tuple[int, int]:
    if not blocks: 
        return 0,0
    sizes = [len(block) for block in blocks]
    return int(min(sizes)), int(max(sizes))

def print_summary(
    *,
    files_loaded: list[Path],
    total_events_loaded: int,
    lf_training_pool: pd.DataFrame,
    hf_training_pool: pd.DataFrame,
    hf_validation_pool: pd.DataFrame,
    block_summaries: dict[str, dict[str, int | tuple[int, int]]],
    shell_cfg: ShellConfig,
    validation_fraction: float,
) -> None:
    print("\n=== XLZD Cylindrical Shell Summary ===")
    print(f"Total files loaded: {len(files_loaded)}")
    print(f"Total events loaded: {total_events_loaded:,}")
    print(f"Shell classes: {shell_cfg.n_shells:,}")
    print(f"LF training events (fidelity=0): {len(lf_training_pool):,}")
    print(f"HF training events (fidelity=1): {len(hf_training_pool):,}")
    print(f"HF validation events (fidelity=1): {len(hf_validation_pool):,}")
    print(f"Requested HF validation fraction: {validation_fraction:.2%}")

    for label, info in block_summaries.items():
        size_range = info["size_range"]
        print(f"{label}: blocks={info['block_count']}, block_size_range={size_range[0]}-{size_range[1]}, unused_leftover_rows={info['leftover_rows']}")

def _prediction_meta_array(
    meta: dict[str, np.ndarray],
    key: str,
    n_events: int,
    *,
    file_path: Path,
) -> np.ndarray:
    if key not in meta:
        raise ValueError(f"{file_path.name}: missing required metadata field {key!r}")
        
    values = np.asarray(meta[key]).reshape(-1)
    if len(values) == 1 and n_events != 1:
        values = np.repeat(values[0], n_events)
    if len(values) != n_events:
        raise ValueError(f"{file_path.name}: metadata {key!r} has {len(values)} values for {n_events} events")

    if values.dtype.kind == "S":
        values = np.char.decode(values, "utf-8")
    elif values.dtype.kind == "O":
        values = np.asarray([
            value.decode("utf-8")
            if isinstance(value, (bytes, np.bytes_))
            else value
            for value in values
        ])

    return values

# -----------------------------------------------------------------------------
# Configs
# -----------------------------------------------------------------------------
def load_config(path: Path) -> dict:
    """Loads either a JSON or a TOML config"""
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist at {path}")

    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    elif suffix == ".toml":
        with path.open('rb') as f:
            return tomllib.load(f)
    raise ValueError(f"Unsupported config format {suffix!r}. Expected json or toml")
        
def load_file_manifest(input_dir: Path, manifest_name: str = MANIFEST_NAME) -> pd.DataFrame:
    """
    Loads data manifest of input files

    Structure:
        filename   Name of file, extension included
        R          Maximum radius of the detector 
        Z          Maximum Z (half-height) of the detector
        z_center   Z-value of the center of the detector (usually just equal to the half-height)
        fidelity   0 for LF, 1 for HF
    """
    manifest_path = input_dir / manifest_name
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest file {manifest_path}")
    manifest = pd.read_csv(manifest_path,
                           na_values=["", "None", "none", "NULL", "null", "NaN", "nan"],
                           keep_default_na=True)

    # Check if all columns are there
    required = {"filename", "R", "Z", "z_center", "fidelity"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest is missing required columns {sorted(missing)}")

    # Clean it up
    manifest = manifest.copy()
    manifest["filename"] = manifest["filename"].astype(str).str.strip()
    manifest["R"] = pd.to_numeric(manifest["R"], errors="coerce")
    manifest["Z"] = pd.to_numeric(manifest["Z"], errors="coerce")
    manifest["z_center"] = pd.to_numeric(manifest["z_center"], errors="coerce")
    manifest["fidelity"] = _validate_fidelity_series(
        manifest["fidelity"],
        context="file_manifest.csv",
    )
    if manifest["filename"].isna().any() or (manifest["filename"] == "").any():
        raise ValueError("Manifest contains missing filename values.")
    manifest["filename"] = manifest["filename"].astype(str)
    manifest["R"] = manifest["R"].astype(float)
    manifest["Z"] = manifest["Z"].astype(float)
    manifest["z_center"] = manifest["z_center"].astype(float)
    manifest["fidelity"] = manifest["fidelity"].astype(np.int32)

    return manifest

def get_manifest_file_path(input_dir: Path, file_name: str) -> Path:
    path = input_dir / file_name
    if not path.exists():
        raise FileNotFoundError(f"Manifest entry points to missing file {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Manifest entry is not a file {path}")

    return path

def build_config(config_path: str | Path) -> PreparationConfig:
    config_path = Path(config_path)
    raw = load_config(config_path)
    data = raw.get("data", {})
    split = raw.get("split", {})
    blocks = raw.get("blocks", {})
    shell = raw.get("shell", {})
    
    if "validation_fraction" not in split:
        raise ValueError("Config must define validation_fraction under [split]")
    if "input_dir" not in data:
        raise ValueError("Config must define input_dir under [data]")
    if "output_dir" not in data:
        raise ValueError("Config must define output_dir under [data]")

    config = PreparationConfig(
        file_load=FileLoadConfig(
            input_dir=Path(data["input_dir"]),
            file_stems=data.get("file_stems", list(DEFAULT_FILE_STEMS)),
            max_rows_per_file=data.get("max_rows_per_file")),
        split=HFValidationSplitConfig(
            validation_fraction=float(split["validation_fraction"]),
            random_seed=int(split.get("random_seed", 42))),
        sampling=SamplingConfig(
            hf_block_size=int(blocks.get("hf_block_size", 100000)),
            lf_block_size=int(blocks.get("lf_block_size", 20000)),
            validation_block_size=None if blocks.get("validation_block_size") is None else int(blocks["validation_block_size"]),
            progress=bool(blocks.get("progress", True))),
        shell=ShellConfig(
            R_max=None,
            Z_max=None,
            n_shells=int(shell.get("n_shells", 100)),
            min_candidate_events=int(shell.get("min_candidate_events", 25)),
            z_center=None,
            scale_power=float(shell.get("scale_power", 1.0 / 3.0))),
        output=OutputConfig(
            output_dir=Path(data["output_dir"]),
            output_format=str(data.get("output_format", "csv")),
        ))

    config.validate()
    return config

def build_shell_config_for_manifest_row(
    row: pd.Series,
    base_shell_cfg: ShellConfig,
) -> ShellConfig:
    return ShellConfig(
        R_max=float(row["R"]),
        Z_max=float(row["Z"]),
        n_shells=base_shell_cfg.n_shells,
        min_candidate_events=base_shell_cfg.min_candidate_events,
        z_center=float(row["z_center"]),
        scale_power=base_shell_cfg.scale_power,
    )

# -----------------------------------------------------------------------------
# Data Management
# -----------------------------------------------------------------------------
def split_hf_training_validation(
    events: pd.DataFrame,
    *,
    validation_fraction: float,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Keep every LF event for training and split on HF events

    Fidelity read from file manifest"""
    if "fidelity" not in events.columns:
        raise ValueError("Events are missing the fidelity column")
    events = events.copy()
    events["fidelity"] = _validate_fidelity_series(events["fidelity"], context="Prepared events")
    if not 0.0 < float(validation_fraction) < 1.0:
        raise ValueError(f"Validation fraction must be between 0 and 1")

    # Separate out HF
    lf_training = events[events["fidelity"] == 0].copy()
    hf_events = events[events["fidelity"] == 1].copy().reset_index(drop=True)
    if hf_events.empty:
        raise ValueError("No HF events were found")

    n_hf = len(hf_events)
    n_validation = int(round(n_hf * float(validation_fraction)))
    if n_hf < 2:
        raise ValueError(f"At least two HF points required")
    n_validation = min(max(1, n_validation), n_hf - 1)

    # Create random split
    rng = np.random.default_rng(int(random_seed))
    order = rng.permutation(n_hf)
    validation_idx = order[:n_validation]
    training_idx = order[n_validation:]

    hf_training = hf_events.iloc[training_idx].copy().reset_index(drop=True)
    hf_validation = hf_events.iloc[validation_idx].copy().reset_index(drop=True)
    lf_training = lf_training.reset_index(drop=True)

    # Printout
    print(f"[split] LF training (fidelity=0): {len(lf_training):,}")
    print(f"[split] HF training (fidelity=1): {len(hf_training):,}")
    print(f"[split] HF validation (fidelity=1): {len(hf_validation):,} ({validation_fraction:.2%} requested)")

    return lf_training, hf_training, hf_validation

def write_h5_class_block(
    *,
    output_path: Path,
    theta: np.ndarray,
    phi: np.ndarray,
    target_shell: np.ndarray,
    theta_headers: Sequence[str],
    phi_headers: Sequence[str],
    meta: dict[str, np.ndarray],
) -> None:
    """Write one event-level h5 block

    Datasets
        theta:          float32, shape (N, theta_dim)
        phi:            float32, shape (N, phi_dim)
        target_shell:   int64, shape (N, ) zero-indexed class labels
        meta:           event_index and one-based shell index
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    theta = np.asarray(theta, dtype=np.float32)
    phi = np.asarray(phi, dtype=np.float32)
    target_shell = np.asarray(target_shell, dtype=np.int64).reshape(-1)

    # Buncha checks
    if theta.ndim != 2:
        raise ValueError(f"theta must have shape (N, theta_dim), got {theta.shape}")
    if phi.ndim != 2:
        raise ValueError(f"phi must have shape (N, phi_dim), got {phi.shape}")
    if target_shell.ndim != 1:
        raise ValueError(f"target_shell must have shape (N,), got {target_shell.shape}")
    if len(theta) != len(phi) or len(phi) != len(target_shell):
        raise ValueError(f"theta/phi/target length mismatch: theta={len(theta)}, phi={len(phi)}, target_shell={len(target_shell)}")
    if theta.shape[1] != len(theta_headers):
        raise ValueError(f"theta dim/header mismatch: theta.shape={theta.shape}, theta_headers={list(theta_headers)}")
    if phi.shape[1] != len(phi_headers):
        raise ValueError(f"phi dim/header mismatch: phi.shape={phi.shape}, phi_headers={list(phi_headers)}")

    # Actually write the h5
    with h5py.File(output_path, 'w') as f:
        f.create_dataset("theta", data=theta, compression="gzip", compression_opts=4)
        f.create_dataset("phi", data=phi, compression="gzip", compression_opts=4)
        f.create_dataset("target_shell", data=target_shell, compression="gzip", compression_opts=4)
        f.create_dataset("theta_labels", data=np.asarray(theta_headers, dtype="S"))
        f.create_dataset("phi_labels", data=np.asarray(phi_headers, dtype="S"))
        f.create_dataset("target_headers", data=np.asarray([TARGET_COLUMN], dtype="S"))

        meta_group = f.create_group("meta")
        for key, value in meta.items():
            meta_group.create_dataset(key, data=_as_h5_array(value), compression="gzip", compression_opts=4)
    
def write_h5_all_class_blocks(
    *,
    blocks: Sequence[pd.DataFrame],
    output_dir: Path,
    split_name: str,
    file_prefix: str,
    theta_headers: Sequence[str],
    phi_headers: Sequence[str],
    n_shells: int,
) -> pd.DataFrame:
    """Write the h5 for all the blocks one at a time"""
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []

    required_columns = {
        *theta_headers,
        *phi_headers,
        TARGET_COLUMN,
        "shell_index",
        EVENT_ID_COLUMN,
        "fidelity",
    }

    for block_index, block_df in enumerate(blocks):
        if block_df.empty:
            continue
        missing = required_columns - set(block_df.columns)
        if missing:
            raise ValueError(f"Block {block_index} is missing required columns: {sorted(missing)}")

        fidelity_values = _validate_fidelity_series(block_df["fidelity"], context=f"Block {block_index}").to_numpy(dtype=np.int32)
        unique_fidelities = np.unique(fidelity_values)
        if len(unique_fidelities) != 1:
            raise ValueError(f"Block {block_index} mixes fidelities {unique_fidelities.tolist()}. Blocks must be created within one manifest-defined fidelity.")
        block_fidelity = int(unique_fidelities[0])

        theta = block_df[list(theta_headers)].to_numpy(dtype=np.float32)
        phi = block_df[list(phi_headers)].to_numpy(dtype=np.float32)
        target_shell = block_df[TARGET_COLUMN].to_numpy(dtype=np.int64).reshape(-1)
        source_file = (
            block_df["source_file"].astype(str).to_numpy()
            if "source_file" in block_df.columns
            else np.asarray(["unknown"] * len(block_df))
        )

        meta = {
            "event_index": block_df[EVENT_ID_COLUMN].to_numpy(dtype=np.int64),
            "original_event_id": (
                block_df["original_event_id"].to_numpy(dtype=np.int64)
                if "original_event_id" in block_df.columns
                else block_df[EVENT_ID_COLUMN].to_numpy(dtype=np.int64)
            ),
            "shell_index": block_df["shell_index"].to_numpy(dtype=np.int64),
            "source_file": np.asarray(source_file, dtype="S"),
            "fidelity": fidelity_values,
            "split_name": np.asarray([split_name] * len(block_df), dtype="S"),
            "detector_R": block_df["detector_R"].to_numpy(dtype=np.float32),
            "detector_Z": block_df["detector_Z"].to_numpy(dtype=np.float32),
            "detector_z_center": block_df["detector_z_center"].to_numpy(dtype=np.float32),
        }

        output_path = output_dir / f"{file_prefix}_block{block_index:04d}_event_classes.h5"
        write_h5_class_block(
            output_path=output_path,
            theta=theta,
            phi=phi,
            target_shell=target_shell,
            theta_headers=theta_headers,
            phi_headers=phi_headers,
            meta=meta,
        )

        class_counts = np.bincount(target_shell, minlength=n_shells)
        records.append(
            {
                "split_name": split_name,
                "fidelity": block_fidelity,
                "block_index": block_index,
                "file_name": output_path.name,
                "file_path": str(output_path),
                "original_block_rows": int(len(block_df)),
                "saved_event_rows": int(len(target_shell)),
                "dropped_rows": 0,
                "n_shells": int(n_shells),
                "min_class_index": int(target_shell.min()),
                "max_class_index": int(target_shell.max()),
                "nonzero_classes": int(np.count_nonzero(class_counts)),
            }
        )

    return pd.DataFrame.from_records(records)
    
            
# -----------------------------------------------------------------------------
# Shell Management
# -----------------------------------------------------------------------------
def assign_shell_labels_for_file(
    *,
    events: pd.DataFrame,
    shell_table_df: pd.DataFrame,
    phi_headers: Sequence[str],
) -> pd.DataFrame:
    work = events.copy()
    if EVENT_ID_COLUMN in work.columns:
        work["original_event_id"] = work[EVENT_ID_COLUMN].to_numpy()
    else:
        work["original_event_id"] = np.arange(len(work), dtype=np.int64)

    # Force unique temp event id so mapping works
    work[EVENT_ID_COLUMN] = np.arange(len(work), dtype=np.int64)
    shell_block = build_shell_event_block(
        block_df = work,
        shell_table_df = shell_table_df,
        feature_columns = phi_headers,
        keep_event_data = False,
    )

    if len(shell_block.truth_shell) == 0:
        return work.iloc[:0].copy()
    valid_row_positions = np.asarray(shell_block.event_index, dtype=np.int64)

    labeled = work.iloc[valid_row_positions].copy()
    labeled[TARGET_COLUMN] = np.asarray(
        shell_block.truth_shell,
        dtype=np.int64,
    )
    
    labeled["shell_index"] = np.asarray(
        shell_block.human_shell,
        dtype=np.int64,
    )

    return labeled.reset_index(drop=True)

def compute_shell_class_weights(
    pool: ShellH5EventPool,
    *,
    beta: float = 0.5,
    max_weight: float | None = 20.0,
):
    """Compute the same optional inverse-frequency shell weights as the old pipeline."""
    import torch

    counts = np.zeros(pool.n_shells, dtype=np.float64)
    for file_path in pool.files:
        with h5py.File(file_path, "r") as f:
            target_shell = np.asarray(f[TARGET_COLUMN], dtype=np.int64).reshape(-1)
        counts += np.bincount(target_shell, minlength=pool.n_shells)

    safe_counts = np.maximum(counts, 1.0)
    weights = safe_counts ** (-float(beta))
    present = counts > 0
    if np.any(present):
        weights[present] /= weights[present].mean()
    weights[~present] = 0.0
    if max_weight is not None:
        weights = np.clip(weights, 0.0, float(max_weight))

    return torch.tensor(weights, dtype=torch.float32)

def shell_classification_loss(
    logits,
    target,
    *,
    sigma: float = 1.25,
    hard_fraction: float = 0.5,
    class_weights=None,
):
    """Shell-aware hard + Gaussian-smoothed categorical cross entropy.

    This preserves the ordered-shell loss from the previous CNP pipeline while
    keeping that shell-specific assumption outside the generic ``cnp.py``.
    """
    import torch
    import torch.nn.functional as F

    if sigma <= 0:
        raise ValueError("sigma must be greater than zero")
    if not 0.0 <= hard_fraction <= 1.0:
        raise ValueError("hard_fraction must be between 0 and 1")

    target = target.long().reshape(-1)
    n_shells = logits.shape[-1]
    shell_indices = torch.arange(n_shells, device=logits.device, dtype=logits.dtype)
    distances = shell_indices.unsqueeze(0) - target.to(logits.dtype).unsqueeze(1)
    gaussian_target = torch.exp(-0.5 * torch.square(distances / sigma))
    gaussian_target = gaussian_target / gaussian_target.sum(
        dim=-1,
        keepdim=True,
    ).clamp_min(1e-12)

    hard_target = F.one_hot(target, num_classes=n_shells).to(dtype=logits.dtype)
    soft_target = hard_fraction * hard_target + (1.0 - hard_fraction) * gaussian_target

    log_probabilities = F.log_softmax(logits, dim=-1)
    per_event_loss = -torch.sum(soft_target * log_probabilities, dim=-1)

    if class_weights is not None:
        weights = class_weights.to(device=logits.device, dtype=logits.dtype)
        sample_weights = weights[target]
        return torch.sum(sample_weights * per_event_loss) / sample_weights.sum().clamp_min(1e-12)

    return per_event_loss.mean()

def prepare_shell_cnp_data(
    config: PreparationConfig,
    *,
    manifest_name: str = MANIFEST_NAME,
) -> PreparationResult:
    """Run the existing raw-event -> shell-labelled HDF5 preparation pipeline."""
    config.validate()
    validation_fraction = config.split.validation_fraction
    total_start = time.perf_counter()

    input_dir = config.file_load.input_dir
    output_dir = config.output.output_dir
    output_format = config.output.output_format

    if output_dir.exists():
        stage_start = log_stage(f"Clearing existing dataset directory: {output_dir}")
        shutil.rmtree(output_dir)
        finish_stage(stage_start, "Removed previous shell dataset")
    output_dir.mkdir(parents=True, exist_ok=True)

    stage_start = log_stage("Loading file manifest")
    input_manifest = load_file_manifest(input_dir, manifest_name=manifest_name)
    finish_stage(
        stage_start,
        f"Loaded {len(input_manifest)} manifest rows from {input_dir / manifest_name}",
    )

    files_loaded: list[Path] = []
    labeled_event_parts: list[pd.DataFrame] = []
    shell_table_parts: list[pd.DataFrame] = []
    total_events_loaded = 0

    for manifest_index, row in input_manifest.iterrows():
        source_name = str(row["filename"])
        fidelity = int(row["fidelity"])
        data_path = get_manifest_file_path(input_dir, source_name)

        print("\n" + "=" * 80)
        print(f"Processing manifest row {manifest_index + 1}")
        print(f"source_file={source_name}")
        print(f"manifest R={row.get('R')}, Z={row.get('Z')}")
        print(f"fidelity={fidelity}")
        print("=" * 80)

        stage_start = log_stage(f"Loading {data_path.name}")
        events = load_event_file(
            data_path,
            max_rows=config.file_load.max_rows_per_file,
        )
        finish_stage(stage_start, f"Loaded {len(events):,} rows")

        files_loaded.append(data_path)
        total_events_loaded += int(len(events))
        shell_cfg = build_shell_config_for_manifest_row(row, config.shell)

        stage_start = log_stage("Using detector geometry and fidelity from manifest")
        detector_R = float(shell_cfg.R_max)
        detector_Z = float(shell_cfg.Z_max)
        z_center = float(shell_cfg.z_center)
        events = add_centered_z_coordinate(events, z_center)

        events["source_file"] = source_name
        events["fidelity"] = fidelity
        events["detector_R"] = detector_R
        events["detector_Z"] = detector_Z
        events["detector_z_center"] = z_center

        finish_stage(
            stage_start,
            f"Using manifest values: fidelity={fidelity}, "
            f"z_center={z_center:.6g}, R={detector_R:.6g}, Z={detector_Z:.6g}",
        )

        stage_start = log_stage("Building shell table and assigning target shells")
        shell_table_df = build_shell_table(events, shell_cfg)
        shell_table_out = shell_table_df.copy()
        shell_table_out["source_file"] = source_name
        shell_table_out["fidelity"] = fidelity
        shell_table_out["detector_R"] = detector_R
        shell_table_out["detector_Z"] = detector_Z
        shell_table_out["detector_z_center"] = z_center
        shell_table_parts.append(shell_table_out)

        labeled_events = assign_shell_labels_for_file(
            events=events,
            shell_table_df=shell_table_df,
            phi_headers=PHI_HEADERS,
        )
        finish_stage(
            stage_start,
            f"Assigned shells for {len(labeled_events):,}/{len(events):,} events",
        )

        labeled_event_parts.append(labeled_events)
        del events, labeled_events, shell_table_df

    stage_start = log_stage("Concatenating labeled events across all detector geometries")
    if not labeled_event_parts:
        raise RuntimeError("No labeled events were produced from the manifest.")

    all_events = pd.concat(labeled_event_parts, ignore_index=True, sort=False)
    all_events["mixed_event_index"] = np.arange(len(all_events), dtype=np.int64)
    all_events[EVENT_ID_COLUMN] = all_events["mixed_event_index"]
    all_events["fidelity"] = _validate_fidelity_series(
        all_events["fidelity"],
        context="Combined prepared events",
    )
    finish_stage(
        stage_start,
        f"Combined labeled event table has {len(all_events):,} rows",
    )

    stage_start = log_stage("Splitting only high-fidelity events for validation")
    lf_training_pool, hf_training_pool, hf_validation_pool = split_hf_training_validation(
        all_events,
        validation_fraction=validation_fraction,
        random_seed=config.split.random_seed,
    )
    finish_stage(
        stage_start,
        f"HF split complete (validation_fraction={validation_fraction:.4f})",
    )

    validation_block_size = (
        config.sampling.hf_block_size
        if config.sampling.validation_block_size is None
        else config.sampling.validation_block_size
    )

    pool_specs = (
        (
            "LF training",
            lf_training_pool,
            output_dir / "training" / "lf",
            "training",
            "lf",
            config.sampling.lf_block_size,
        ),
        (
            "HF training",
            hf_training_pool,
            output_dir / "training" / "hf",
            "training",
            "hf",
            config.sampling.hf_block_size,
        ),
        (
            "HF validation",
            hf_validation_pool,
            output_dir / "validation" / "hf",
            "validation",
            "hf",
            validation_block_size,
        ),
    )

    manifest_parts: list[pd.DataFrame] = []
    block_summaries: dict[str, dict[str, int | tuple[int, int]]] = {}

    for label, pool_df, pool_output_dir, split_name, file_prefix, block_size in pool_specs:
        stage_start = log_stage(f"Writing {label} event-class H5 blocks")
        block_result = split_pool_into_blocks(pool_df, block_size=block_size)
        blocks = list(block_result.blocks)

        block_manifest = write_h5_all_class_blocks(
            blocks=blocks,
            output_dir=pool_output_dir,
            split_name=split_name,
            file_prefix=file_prefix,
            theta_headers=THETA_HEADERS,
            phi_headers=PHI_HEADERS,
            n_shells=config.shell.n_shells,
        )
        if not block_manifest.empty:
            manifest_parts.append(block_manifest)

        block_summaries[label] = {
            "block_count": int(len(blocks)),
            "size_range": block_range(blocks),
            "leftover_rows": int(block_result.leftover_rows),
        }
        finish_stage(stage_start, f"{len(block_manifest)} {label} blocks written")

    stage_start = log_stage("Writing pool tables and manifests")
    save_dataframe(all_events, output_dir / "processed_all_events", output_format)
    save_dataframe(lf_training_pool, output_dir / "lf_training_pool", output_format)
    save_dataframe(hf_training_pool, output_dir / "hf_training_pool", output_format)
    save_dataframe(hf_validation_pool, output_dir / "hf_validation_pool", output_format)

    if shell_table_parts:
        shell_table_by_theta = pd.concat(shell_table_parts, ignore_index=True, sort=False)
    else:
        shell_table_by_theta = pd.DataFrame()

    shell_table_path = save_dataframe(
        shell_table_by_theta,
        output_dir / "shell_table_by_theta",
        output_format,
    )

    manifest_df = (
        pd.concat(manifest_parts, ignore_index=True, sort=False)
        if manifest_parts
        else pd.DataFrame()
    )
    manifest_path = save_dataframe(
        manifest_df,
        output_dir / "event_class_manifest",
        output_format,
    )
    finish_stage(stage_start, "Output files written")

    print_summary(
        files_loaded=files_loaded,
        total_events_loaded=total_events_loaded,
        lf_training_pool=lf_training_pool,
        hf_training_pool=hf_training_pool,
        hf_validation_pool=hf_validation_pool,
        block_summaries=block_summaries,
        shell_cfg=config.shell,
        validation_fraction=validation_fraction,
    )

    print(
        f"\nArtifacts:"
        f"\n- shell table by theta: {shell_table_path}"
        f"\n- manifest: {manifest_path}"
        f"\n- training H5 root: {output_dir / 'training'}"
        f"\n- validation H5 root: {output_dir / 'validation'}"
    )

    total_elapsed = time.perf_counter() - total_start
    print(f"\nTotal elapsed: {total_elapsed:.2f}s")

    return PreparationResult(
        output_dir=output_dir,
        training_dir=output_dir / "training",
        validation_dir=output_dir / "validation",
        shell_table_path=Path(shell_table_path),
        manifest_path=Path(manifest_path),
    )

def build_shell_table(df_for_support: pd.DataFrame, shell_cfg: ShellConfig) -> pd.DataFrame:
    boundaries = shell_boundaries(shell_cfg)
    support_rows: list[dict[str, float | int]] = []

    for i in range(1, shell_cfg.n_shells + 1):
        prev = boundaries.iloc[i - 1]
        curr = boundaries.iloc[i]
        mask = inside_shell(
            df_for_support,
            R_inner=float(prev["R_boundary"]),
            Z_inner=float(prev["Z_boundary"]),
            R_outer=float(curr["R_boundary"]),
            Z_outer=float(curr["Z_boundary"]),
        )
        outer_volume = 2.0*np.pi*float(curr["Z_boundary"])*float(curr["R_boundary"])**2
        inner_volume = 2.0*np.pi*float(prev["Z_boundary"])*float(prev["R_boundary"])**2
        shell_volume = outer_volume - inner_volume
        support_rows.append(
            {
                "shell_index": int(i),
                "class_index": int(i-1),
                "R_inner": float(prev["R_boundary"]),
                "Z_inner": float(prev["Z_boundary"]),
                "R_shell": float(curr["R_boundary"]),
                "Z_shell": float(curr["Z_boundary"]),
                "candidate_events": int(mask.sum()),
                "shell_volume": float(shell_volume),
            }
        )

    out = pd.DataFrame(support_rows)
    low_support = out[out["candidate_events"] < shell_cfg.min_candidate_events]
    if not low_support.empty:
        print("[warn] Some shell classes have low support, but they are kept for categorical CE because class IDs must remain fixed")

    return out.sort_values(["shell_index"]).reset_index(drop=True)

def shell_boundaries(shell_cfg: ShellConfig) -> pd.DataFrame:
    idx = np.arange(0, shell_cfg.n_shells + 1, dtype=float)
    frac = idx / float(shell_cfg.n_shells)
    scale = frac ** shell_cfg.scale_power
    r = shell_cfg.R_max * scale
    z = shell_cfg.Z_max * scale
    return pd.DataFrame(
        {
            "shell_level": idx.astype(int),
            "R_boundary": r.astype(float),
            "Z_boundary": z.astype(float),
        }
    )

def inside_shell(
    df: pd.DataFrame,
    *,
    R_inner: float,
    Z_inner: float,
    R_outer: float,
    Z_outer: float,
) -> np.ndarray:
    required = {"r", Z_FROM_CENTER_COLUMN}
    if not required.issubset(df.columns):
        raise ValueError(f"Block dataframe must contain columns {required}.")

    r = df["r"].to_numpy(dtype=float)
    z = df[Z_FROM_CENTER_COLUMN].to_numpy(dtype=float)

    inside_outer = (r <= R_outer) & (z <= Z_outer)
    inside_inner = (r <= R_inner) & (z <= Z_inner)
    return inside_outer & ~inside_inner

def build_shell_event_block(
    *,
    block_df: pd.DataFrame,
    shell_table_df: pd.DataFrame,
    feature_columns: Sequence[str],
    keep_event_data: bool=True,
) -> ShellEventBlock:
    missing_features = [col for col in feature_columns if col not in block_df.columns]
    if missing_features:
        raise ValueError(f"Block is missing required feature columns: {missing_features}")

    positive_shell_one_based = positive_shells_for_block(
        block_df=block_df, shell_table_df=shell_table_df,
    )

    valid_mask = positive_shell_one_based.notna()
    if not valid_mask.any():
        raise RuntimeError("This block has no events with a valid shell")
    valid_events = block_df.loc[valid_mask].copy()

    human_shell = positive_shell_one_based.loc[valid_mask].astype(np.int32).to_numpy()
    truth_shell = human_shell.astype(np.int64) - 1
    features = valid_events[list(feature_columns)].to_numpy(dtype=np.float32)
    event_index = valid_events.index.to_numpy(dtype=np.int64)

    if keep_event_data:
        valid_events["event_index"] = event_index
        valid_events["human_shell_index"] = human_shell
        valid_events["truth_shell"] = truth_shell

    return ShellEventBlock(
        features=features,
        truth_shell = truth_shell.astype(np.int64),
        human_shell = human_shell.astype(np.int32),
        event_index = event_index.astype(np.int64),
        valid_events = valid_events
    )

def positive_shells_for_block(
    block_df: pd.DataFrame,
    shell_table_df: pd.DataFrame,
) -> pd.Series:
    """
    Return one positive shell index per event
    
    The returned shell is one-indexed
    Events outside the detector bounds are labelled NaN
    """
    positive_shell = pd.Series(np.nan, index=block_df.index, dtype="float")
    
    for row in shell_table_df.itertuples(index=False):
        mask = inside_shell(
            block_df,
            R_inner=float(row.R_inner),
            Z_inner=float(row.Z_inner),
            R_outer=float(row.R_shell),
            Z_outer=float(row.Z_shell),
        )
        positive_shell.loc[mask] = int(row.shell_index)

    return positive_shell

# -----------------------------------------------------------------------------
# CNP Functions
# -----------------------------------------------------------------------------
Batch = tuple[np.ndarray, np.ndarray]
BatchProvider = Callable[[int], Batch]
EpochProvider = Callable[[int], Iterable[Batch]]

class ShellH5EventPool:
    """
    Read shell HDF5 blocks and write to CNP (x,y) batches

    CNP sees:
        x = concat(theta, phi)
        y = target_shell

    Does not pass shell knowledge to CNP
    """
    def __init__(
        self,
        directory: str | Path,
        *,
        theta_headers: Sequence[str] = THETA_HEADERS,
        phi_headers: Sequence[str] = PHI_HEADERS,
        n_shells: int,
        seed: int = 42,
        cache_files: bool = True,
    ) -> None:
        self.directory = Path(directory)
        self.theta_headers = list(theta_headers)
        self.phi_headers = list(phi_headers)
        self.n_shells = int(n_shells)
        self.rng = np.random.default_rng(seed)
        self.cache_files = bool(cache_files)

        self._cache: dict[Path, tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]] = {}
        self._row_count_cache: dict[Path, int] = {}

        if not self.directory.exists():
            raise FileNotFoundError(f"H5 directory does not exist: {self.directory}")

        self.files = sorted(p for p in self.directory.rglob("*.h5") if p.is_file())
        if not self.files:
            raise FileNotFoundError(f"No H5 files found in {self.directory}")

    @staticmethod
    def _decode_labels(arr: np.ndarray) -> list[str]:
        return [
            item.decode("utf-8") if isinstance(item, (bytes, np.bytes_)) else str(item)
            for item in arr
        ]

    @staticmethod
    def _read_meta(f: h5py.File) -> dict[str, np.ndarray]:
        if "meta" not in f:
            return {}
        return {key: np.asarray(f["meta"][key]) for key in f["meta"].keys()}

    def _load_one(
        self,
        file_path: Path,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        if self.cache_files and file_path in self._cache:
            return self._cache[file_path]

        with h5py.File(file_path, "r") as f:
            required = {"theta", "phi", TARGET_COLUMN}
            missing = required - set(f.keys())
            if missing:
                raise ValueError(f"{file_path.name}: missing HDF5 datasets {sorted(missing)}")

            # Read the H5
            theta = np.asarray(f["theta"], dtype=np.float32)
            phi = np.asarray(f["phi"], dtype=np.float32)
            target_shell = np.asarray(f[TARGET_COLUMN], dtype=np.int64).reshape(-1)
            meta = self._read_meta(f)

            # Validate the data
            if theta.ndim != 2:
                raise ValueError(f"{file_path.name}: expected theta shape (N, theta_dim), got {theta.shape}")
            if phi.ndim != 2:
                raise ValueError(f"{file_path.name}: expected phi shape (N, phi_dim), got {phi.shape}")
            if len(theta) != len(phi) or len(phi) != len(target_shell):
                raise ValueError(f"{file_path.name}: row count mismatch: theta={len(theta)}, phi={len(phi)}, target_shell={len(target_shell)}")
            if theta.shape[1] != len(self.theta_headers):
                raise ValueError(f"{file_path.name}: theta dim/header mismatch. Expected {len(self.theta_headers)} columns {self.theta_headers}, got {theta.shape}")
            if phi.shape[1] != len(self.phi_headers):
                raise ValueError(f"{file_path.name}: phi dim/header mismatch. Expected {len(self.phi_headers)} columns {self.phi_headers}, got {phi.shape}")

            if len(target_shell):
                y_min = int(target_shell.min())
                y_max = int(target_shell.max())
                if y_min < 0 or y_max >= self.n_shells:
                    raise ValueError(f"{file_path.name}: target_shell must be in [0, {self.n_shells - 1}], got min={y_min}, max={y_max}")

            if "theta_labels" in f:
                labels = self._decode_labels(np.asarray(f["theta_labels"]))
                if labels != self.theta_headers:
                    raise ValueError(f"Theta labels mismatch in {file_path.name}. Expected {self.theta_headers}, got {labels}")

            if "phi_labels" in f:
                labels = self._decode_labels(np.asarray(f["phi_labels"]))
                if labels != self.phi_headers:
                    raise ValueError(f"Phi labels mismatch in {file_path.name}. Expected {self.phi_headers}, got {labels}")

            if "target_headers" in f:
                labels = self._decode_labels(np.asarray(f["target_headers"]))
                if labels != [TARGET_COLUMN]:
                    raise ValueError(f"Target headers mismatch in {file_path.name}. Expected {[TARGET_COLUMN]}, got {labels}")

            if "fidelity" not in meta:
                raise ValueError(f"{file_path.name}: missing required meta/fidelity")
            fidelity = pd.Series(np.asarray(meta["fidelity"]).reshape(-1))
            if len(fidelity) == 1 and len(target_shell) != 1:
                fidelity = pd.Series(np.repeat(fidelity.iloc[0], len(target_shell)))
            if len(fidelity) != len(target_shell):
                raise ValueError(f"{file_path.name}: fidelity has {len(fidelity)} values for {len(target_shell)} events")
            meta["fidelity"] = _validate_fidelity_series(fidelity, context=f"{file_path.name} meta/fidelity",).to_numpy(dtype=np.int32)

            # Actually assign x
            x = np.concatenate([theta, phi], axis=1).astype(np.float32)

        result = (x, target_shell, meta)
        if self.cache_files:
            self._cache[file_path] = result
        return result

    def _count_rows_one(self, file_path: Path) -> int:
        if file_path in self._row_count_cache:
            return self._row_count_cache[file_path]
        with h5py.File(file_path, "r") as f:
            count = int(f[TARGET_COLUMN].shape[0])
        self._row_count_cache[file_path] = count
        return count

    def _choose_files(self, files_per_batch: int) -> list[Path]:
        if files_per_batch <= 0:
            raise ValueError("Files per batch must be positive")
        count = min(int(files_per_batch), len(self.files))
        if count == len(self.files):
            return list(self.files)
        indices = self.rng.choice(len(self.files), size=count, replace=False)
        return [self.files[int(i)] for i in indices]

    def sample_batch(self, batch_size: int, files_per_batch: int = 32) -> Batch:
        """Sample a random event batch from several H5 blocks"""
        if batch_size <= 0:
            raise ValueError("Batch size must be positive")

        # Choose which files to take from
        chosen = self._choose_files(files_per_batch)
        per_file = max(1, int(np.ceil(batch_size / len(chosen))))
        x_parts: list[np.ndarray] = []
        y_parts: list[np.ndarray] = []

        # Grab the actual x and y from those files
        for file_path in chosen:
            x, y, _meta = self._load_one(file_path)
            if len(y) == 0:
                continue
            indices = self.rng.integers(0, len(y), size=per_file)
            x_parts.append(x[indices])
            y_parts.append(y[indices])
        if not x_parts:
            raise RuntimeError("Could not sample a non-empty batch from HDF5 files")
        
        # Make the batches that the CNP uses
        x_batch = np.vstack(x_parts).astype(np.float32)[:batch_size]
        y_batch = np.concatenate(y_parts).astype(np.int64)[:batch_size]
        return x_batch, y_batch

    def iter_epoch_batches(
        self,
        batch_size: int,
        files_per_batch: int = 32,
        *,
        shuffle: bool = True,
        drop_last: bool = False,
    ) -> Iterator[Batch]:
        """Yield a full pass over every prepared event exactly once per epoch"""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if files_per_batch <= 0:
            raise ValueError("files_per_batch must be positive")

        # Make sure there is a concrete order
        file_order = list(self.files)
        if shuffle and len(file_order) > 1:
            order = self.rng.permutation(len(file_order))
            file_order = [file_order[int(i)] for i in order]

        x_buffer: list[np.ndarray] = []
        y_buffer: list[np.ndarray] = []
        for file_start in range(0, len(file_order), files_per_batch):
            for file_path in file_order[file_start : file_start + files_per_batch]:
                x, y, _meta = self._load_one(file_path)
                if len(y) == 0:
                    continue
                if shuffle and len(y) > 1:
                    order = self.rng.permutation(len(y))
                    x = x[order]
                    y = y[order]
                x_buffer.append(x)
                y_buffer.append(y)
            if not x_buffer:
                continue

            x_all = np.vstack(x_buffer).astype(np.float32)
            y_all = np.concatenate(y_buffer).astype(np.int64)
            if shuffle and len(y_all) > 1:
                order = self.rng.permutation(len(y_all))
                x_all = x_all[order]
                y_all = y_all[order]

            while len(y_all) >= batch_size:
                yield x_all[:batch_size], y_all[:batch_size]
                x_all = x_all[batch_size:]
                y_all = y_all[batch_size:]

            x_buffer = [x_all] if len(y_all) else []
            y_buffer = [y_all] if len(y_all) else []
        
        if y_buffer and len(y_buffer[0]) and not drop_last:
            yield x_buffer[0], y_buffer[0]

    def total_events(self) -> int:
        return sum(self._count_rows_one(file_path) for file_path in self.files)

    def iter_file_data(
        self,
    ) -> Iterator[tuple[Path, np.ndarray, np.ndarray, dict[str, np.ndarray]]]:
        for file_path in self.files:
            x, y, meta = self._load_one(file_path)
            yield file_path, x.astype(np.float32), y.astype(np.int64), meta

@dataclass
class ShellCNPProviders:
    """Shell specific callables that plug into the CNP"""
    training_pool: ShellH5EventPool
    validation_pool: Optional[ShellH5EventPool]
    inference_pool: ShellH5EventPool
    files_per_batch_train: int = 32
    files_per_batch_validation: int = 16

    def train_batch_fn(self, batch_size: int) -> Batch:
        return self.training_pool.sample_batch(
            batch_size,
            files_per_batch=self.files_per_batch_train)

    def validation_batch_fn(self, batch_size: int) -> Batch:
        pool = self.validation_pool or self.training_pool
        return pool.sample_batch(
            batch_size,
            files_per_batch=self.files_per_batch_validation,
        )

    def epoch_batches_fn(self, batch_size: int) -> Iterable[Batch]:
        return self.training_pool.iter_epoch_batches(
            batch_size,
            files_per_batch=self.files_per_batch_train,
            shuffle=True,
            drop_last=False,
        )

    def inference_context_fn(self, context_size: int) -> Batch:
        return self.inference_pool.sample_batch(
            context_size,
            files_per_batch=self.files_per_batch_train,
        )

def build_shell_cnp_providers(
    prepared_dir: str | Path,
    *,
    n_shells: int,
    theta_headers: Sequence[str] = THETA_HEADERS,
    phi_headers: Sequence[str] = PHI_HEADERS,
    files_per_batch_train: int = 32,
    files_per_batch_validation: Optional[int] = None,
    seed: int = 42,
    cache_training_files: bool = True,
) -> ShellCNPProviders:
    """Build all providers for the CNP"""
    prepared_dir = Path(prepared_dir)
    validation_files_per_batch = int(
        files_per_batch_validation
        if files_per_batch_validation is not None
        else max(1, files_per_batch_train // 2)
    )

    training_pool = ShellH5EventPool(
        prepared_dir / "training",
        theta_headers=theta_headers,
        phi_headers=phi_headers,
        n_shells=n_shells,
        seed=seed,
        cache_files=cache_training_files,
    )
    validation_dir = prepared_dir / "validation"
    validation_files = (
        sorted(p for p in validation_dir.rglob("*.h5") if p.is_file())
        if validation_dir.exists()
        else []
    )
    if validation_files:
        validation_pool: Optional[ShellH5EventPool] = ShellH5EventPool(
            validation_dir,
            theta_headers=theta_headers,
            phi_headers=phi_headers,
            n_shells=n_shells,
            seed=seed + 1,
            cache_files=False,
        )
    else:
        validation_pool = None
        print(
            "[warn] No held-out validation HDF5 blocks were found; "
            "validation_batch_fn will sample from the training pool."
        )
    # Keep inference context selection independent of the training sampler state.
    inference_pool = ShellH5EventPool(
        prepared_dir / "training",
        theta_headers=theta_headers,
        phi_headers=phi_headers,
        n_shells=n_shells,
        seed=seed + 104729,
        cache_files=False,
    )

    return ShellCNPProviders(
        training_pool=training_pool,
        validation_pool=validation_pool,
        inference_pool=inference_pool,
        files_per_batch_train=int(files_per_batch_train),
        files_per_batch_validation=validation_files_per_batch,
    )

# -----------------------------------------------------------------------------
# MFGP Functions
# -----------------------------------------------------------------------------
@dataclass(slots=True)
class ShellMFGPValidationData:
    """Shell-specific held-out data converted to generic MF-GP validation inputs."""

    x: np.ndarray
    y_true: np.ndarray
    frame: pd.DataFrame
    input_names: list[str]

# -----------------------------------------------------------------------------
# MF-GP Adapters
# -----------------------------------------------------------------------------
def _load_shell_mfgp_rows(
    cnp_prediction_path: str | Path,
    *,
    shell_n: int,
    iteration: int = 0,
) -> pd.DataFrame:
    """Load shell CNP output and validate it for conversion to MF-GP inputs."""
    cnp_prediction_path = Path(cnp_prediction_path)
    if not cnp_prediction_path.exists():
        raise FileNotFoundError(f"CNP prediction file does not exist: {cnp_prediction_path}")
    if shell_n <= 0:
        raise ValueError(f"shell_n must be positive, got {shell_n}")

    df = pd.read_csv(cnp_prediction_path)
    required = {
        *THETA_HEADERS,
        "iteration",
        "fidelity",
        "shell_index",
        "y_cnp",
        "y_cnp_err",
        "y_raw",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"CNP prediction file is missing required columns: {missing}")

    # Convert required numeric values.
    numeric_columns = [
        *THETA_HEADERS,
        "iteration",
        "fidelity",
        "shell_index",
        "y_cnp",
        "y_cnp_err",
        "y_raw",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    if df[numeric_columns].isna().any().any():
        bad_columns = [column for column in numeric_columns if df[column].isna().any()]
        raise ValueError(f"CNP prediction file contains missing or non-numeric values in {bad_columns}")

    # Validate fidelity using the same convention as the rest of the shell code.
    df["fidelity"] = _validate_fidelity_series(
        df["fidelity"],
        context=f"MF-GP adapter input {cnp_prediction_path.name}",
    ).astype(np.int32)

    # shell_index must be a positive integer.
    shell_values = df["shell_index"].to_numpy(dtype=float)
    if not np.allclose(shell_values, np.rint(shell_values)):
        raise ValueError("shell_index contains non-integer values")
    df["shell_index"] = np.rint(shell_values).astype(np.int32)
    if (df["shell_index"] <= 0).any():
        raise ValueError("shell_index must use human shell numbering starting at 1")
    max_shell = int(df["shell_index"].max())

    if shell_n > max_shell:
        raise ValueError(f"Requested shell_n={shell_n}, but CNP output only contains shells through {max_shell}")
    
    # Select requested CNP iteration.
    df = df[df["iteration"].astype(int) == int(iteration)].copy()
    if df.empty:
        raise ValueError(f"No CNP prediction rows found for iteration={iteration}")

    # Keep the containment region shell 1 ... shell_n.
    df = df[df["shell_index"] <= int(shell_n)].copy()
    if df.empty:
        raise ValueError(f"No shell rows found in containment region 1..{shell_n}")

    # Every geometry/fidelity must have exactly one row for every selected shell.
    expected_shells = set(range(1, int(shell_n) + 1))
    group_columns = [*THETA_HEADERS, "fidelity"]
    problems = []
    for key, group in df.groupby(group_columns, dropna=False, sort=False):
        shells = group["shell_index"].astype(int)

        actual_shells = set(shells.tolist())
        missing_shells = sorted(expected_shells - actual_shells)
        duplicate_shells = sorted(shells[shells.duplicated()].unique().tolist())
        if missing_shells or duplicate_shells:
            problems.append(f"{key}: missing={missing_shells}, duplicates={duplicate_shells}")

    if problems:
        raise ValueError(
            "Each detector geometry/fidelity must contain exactly one row "
            f"for every shell 1..{shell_n}.\n"
            + "\n".join(problems[:10])
        )

    return df

def build_shell_mfgp_training_data(
    cnp_prediction_path: str | Path,
    *,
    shell_n: int,
    iteration: int = 0,
) -> mfgp.MFGPTrainingData:
    """Convert shell-wise CNP training output into generic MF-GP training data.

    Low fidelity:
        x_lf = detector geometry
        y_lf = sum of CNP probabilities for shells 1..shell_n
        y_lf_err = shell uncertainties combined in quadrature

    High fidelity:
        x_hf = detector geometry
        y_hf = sum of raw probabilities for shells 1..shell_n
    """

    df = _load_shell_mfgp_rows(
        cnp_prediction_path,
        shell_n=shell_n,
        iteration=iteration,
    )

    lf = df[df["fidelity"] == 0].copy()
    hf = df[df["fidelity"] == 1].copy()

    if lf.empty:
        raise ValueError("Shell CNP training output contains no fidelity=0 low-fidelity rows")
    if hf.empty:
        raise ValueError("Shell CNP training output contains no fidelity=1 high-fidelity rows")

    # Collapse shells 1..shell_n into one containment probability per geometry.
    lf = (
        lf.groupby(THETA_HEADERS, as_index=False, sort=True).agg(
            y_lf=("y_cnp", "sum"),
            y_lf_err=(
                "y_cnp_err",
                lambda values: float(np.sqrt(np.sum(np.square(values.to_numpy(dtype=float)))))))
        .sort_values(THETA_HEADERS, kind="mergesort")
        .reset_index(drop=True))

    hf = (
        hf.groupby(THETA_HEADERS, as_index=False, sort=True)
        .agg(y_hf=("y_raw", "sum"),)
        .sort_values(THETA_HEADERS, kind="mergesort")
        .reset_index(drop=True))

    x_lf = lf[THETA_HEADERS].to_numpy(dtype=float)
    y_lf = lf["y_lf"].to_numpy(dtype=float)
    y_lf_err = lf["y_lf_err"].to_numpy(dtype=float)

    x_hf = hf[THETA_HEADERS].to_numpy(dtype=float)
    y_hf = hf["y_hf"].to_numpy(dtype=float)

    return mfgp.MFGPTrainingData(
        x_lf=x_lf,
        y_lf=y_lf,
        x_hf=x_hf,
        y_hf=y_hf,
        y_lf_err=y_lf_err,
        input_names=list(THETA_HEADERS),
    )

def build_shell_mfgp_validation_data(
    cnp_prediction_path: str | Path,
    *,
    shell_n: int,
    iteration: int = 0,
) -> ShellMFGPValidationData:
    """Convert held-out shell CNP output into MF-GP validation inputs."""

    df = _load_shell_mfgp_rows(
        cnp_prediction_path,
        shell_n=shell_n,
        iteration=iteration,
    )

    # MF-GP validation compares against held-out HF truth.
    hf = df[df["fidelity"] == 1].copy()

    if hf.empty:
        raise ValueError("Shell CNP validation output contains no fidelity=1 high-fidelity rows")
    validation = (
        hf.groupby(THETA_HEADERS, as_index=False, sort=True)
        .agg(y_true=("y_raw", "sum"))
        .sort_values(THETA_HEADERS, kind="mergesort",)
        .reset_index(drop=True))

    x = validation[THETA_HEADERS].to_numpy(dtype=float)
    y_true = validation["y_true"].to_numpy(dtype=float)
    
    return ShellMFGPValidationData(
        x=x,
        y_true=y_true,
        frame=validation,
        input_names=list(THETA_HEADERS),
    )

# -----------------------------------------------------------------------------
# Runners
# -----------------------------------------------------------------------------
def run_shell_cnp_training(
    shell_config_path: str | Path,
    cnp_config_path: str | Path,
) -> cnp.TrainResult:
    """Wrapper for cnp training with specifically shell inputs"""
    # Load the configs
    shell_raw = load_config(shell_config_path)
    cnp_raw = load_config(cnp_config_path)
    shell_cfg = shell_raw["shell"]
    provider_cfg = shell_raw.get("provider", {})
    loss_cfg = shell_raw.get("loss", {})
    run_cfg = cnp_raw.get("run", {})
    training_cfg = cnp_raw.get("training", {})
    context_cfg = cnp_raw.get("context", {})
    model_cfg = cnp_raw.get("model", {})
    optimizer_cfg = cnp_raw.get("optimizer", {})

    # Grab prepared shell data
    prepared_dir = Path(shell_raw["data"]["output_dir"])
    n_shells = int(shell_cfg.get("n_shells", 100))

    # Build shell-specific CNP Providers
    providers = build_shell_cnp_providers(
        prepared_dir=prepared_dir,
        n_shells=n_shells,
        files_per_batch_train=int(provider_cfg.get("files_per_batch_train", 32)),
        files_per_batch_validation=None if provider_cfg.get("files_per_batch_validation") is None else int(provider_cfg["files_per_batch_validation"]),
        seed=int(run_cfg.get("seed", 42)),
        cache_training_files=bool(provider_cfg.get("cache_training_files", True)),
    )

    # Set class weights
    class_weights = None
    if bool(loss_cfg.get("use_class_weights", False)):
        class_weights = compute_shell_class_weights(
            providers.training_pool, 
            beta=float(loss_cfg.get("class_weight_beta", 0.5)), 
            max_weight=loss_cfg.get("class_weight_max", 20.0))

    # Shell specific loss
    loss_fn = partial(
        shell_classification_loss, 
        sigma=float(loss_cfg.get("sigma", 1.25)), 
        hard_fraction=float(loss_cfg.get("hard_fraction", 0.5)), 
        class_weights=class_weights)

    # Device
    device = run_cfg.get("device", "auto")
    if device == "auto":
        device = None

    # Run the actual training
    return cnp.train_cnp(
        train_batch_fn=providers.train_batch_fn,
        validation_batch_fn=providers.validation_batch_fn,
        epoch_batches_fn=providers.epoch_batches_fn if bool(training_cfg.get("use_full_epochs", False)) else None,
        inference_context_fn=providers.inference_context_fn,
        loss_fn=loss_fn,
        n_classes=n_shells,
        out_dir=Path(run_cfg.get("output_dir", "cnp_outputs")),
        version=str(run_cfg.get("version", "default")),
        epochs=int(training_cfg.get("epochs", 15)),
        steps_per_epoch=int(training_cfg.get("steps_per_epoch", 5000)),
        batch_size=int(training_cfg.get("batch_size", 12000)),
        validation_batch_size=None if training_cfg.get("validation_batch_size") is None else int(training_cfg["validation_batch_size"]),
        context_ratio=float(context_cfg.get("ratio", 1.0 / 3.0)),
        context_mode=str(context_cfg.get("mode", "random")),
        inference_context_size=int(context_cfg.get("inference_size", 4096)),
        learning_rate=float(optimizer_cfg.get("learning_rate", 1e-4)),
        weight_decay=float(optimizer_cfg.get("weight_decay", 0.0)),
        repr_dim=int(model_cfg.get("representation_dim", 32)),
        hidden=int(model_cfg.get("hidden_dim", 128)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        monitor_every=int(training_cfg.get("monitor_every", 1000)),
        seed=int(run_cfg.get("seed", 42)),
        device=device,
        input_names=[*THETA_HEADERS, *PHI_HEADERS],
        class_names=[f"shell_{i}" for i in range(1, n_shells + 1)],
        checkpoint_metadata={"distribution": "position_shells", "n_shells": n_shells, "scale_power": float(shell_cfg.get("scale_power", 1.0 / 3.0))},
    )
        
def run_shell_cnp_prediction(
    shell_config_path: str | Path,
    cnp_config_path: str | Path, 
    *,
    split: str = "validation",
    model_path: str | Path | None = None, 
    output_path: str | Path | None = None,
    iteration: int = 0
) -> ShellPredictionResults:
    """Wrapper for CNP prediction based on shells. Handles pool construction, generic CNP inference, shell aggregation, and CSV output."""
    # Load configs
    shell_raw = load_config(shell_config_path)
    cnp_raw = load_config(cnp_config_path)
    shell_cfg = shell_raw["shell"]
    run_cfg = cnp_raw.get("run", {})
    prediction_cfg = cnp_raw.get("prediction", {})

    n_shells = int(shell_cfg.get("n_shells", 100))
    prepared_dir = Path(shell_raw["data"]["output_dir"])

    # Choose prediction data
    split = split.lower()
    if split == "training":
        pool_dir = prepared_dir / "training"
    elif split == "validation":
        pool_dir = prepared_dir / "validation"
    else:
        raise ValueError("split must be either 'training' or 'validation'")

    # Build prediction pool
    pool = ShellH5EventPool(
        pool_dir, 
        theta_headers=THETA_HEADERS, 
        phi_headers=PHI_HEADERS, 
        n_shells=n_shells, 
        seed=int(run_cfg.get("seed", 42)), 
        cache_files=False,
    )

    # CNP output/model information
    cnp_output_dir = Path(run_cfg.get("output_dir", "cnp_outputs"))
    cnp_output_dir.mkdir(parents=True, exist_ok=True)
    version = str(run_cfg.get("version", "default"))
    if model_path is None:
        model_path = cnp_output_dir / f"cnp_{version}_model.pth"
    if output_path is None:
        output_path = cnp_output_dir / (f"cnp_{version}_output.csv" if split == "training" else f"cnp_{version}_output_validation.csv")

    model_path = Path(model_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not model_path.exists():
        raise FileNotFoundError(f"CNP model does not exist: {model_path}")

    # Device
    device = run_cfg.get("device", "auto")
    if device == "auto":
        device = None

    mc_samples = int(prediction_cfg.get("mc_samples", 30))
    chunk_size = int(prediction_cfg.get("chunk_size", 20000))

    # Accumulate predictions by split/fidelity/detector geometry
    accumulators: dict[tuple, dict[str, object]] = {}
    total_events = 0

    pbar = tqdm(pool.iter_file_data(), total=len(pool.files), desc=f"Predicting {split}", unit="file")
    for file_path, x, true_shell, meta in pbar:
        n_events = len(true_shell)
        if n_events == 0:
            continue
    
        pbar.set_postfix(file=file_path.name, events=f"{n_events:,}")

        # Generic CNP prediction
        prediction = cnp.predict_distribution(model_path=model_path, x=x, mc_samples=mc_samples, chunk_size=chunk_size, device=device)

        probabilities = np.asarray(prediction.probabilities, dtype=np.float64)
        uncertainties = np.asarray(prediction.uncertainty, dtype=np.float64)
        true_shell = np.asarray(true_shell, dtype=np.int64).reshape(-1)

        expected_shape = (n_events, n_shells)
        if probabilities.shape != expected_shape:
            raise ValueError(f"{file_path.name}: prediction shape {probabilities.shape} does not match expected {expected_shape}")
        if uncertainties.shape != expected_shape:
            raise ValueError(f"{file_path.name}: uncertainty shape {uncertainties.shape} does not match expected {expected_shape}")

        # Get shell/detector metadata
        split_name = _prediction_meta_array(meta, "split_name", n_events, file_path=file_path).astype(str)
        fidelity = _prediction_meta_array(meta, "fidelity", n_events, file_path=file_path).astype(np.int32)
        detector_R = _prediction_meta_array(meta, "detector_R", n_events, file_path=file_path).astype(float)
        detector_Z = _prediction_meta_array(meta, "detector_Z", n_events, file_path=file_path).astype(float)
        detector_z_center = _prediction_meta_array(meta, "detector_z_center", n_events, file_path=file_path).astype(float)

        group_data = pd.DataFrame({
            "split_name": split_name, 
            "fidelity": fidelity, 
            "detector_R": detector_R, 
            "detector_Z": detector_Z, 
            "detector_z_center": detector_z_center
        })
        group_columns = ["split_name", "fidelity", "detector_R", "detector_Z", "detector_z_center"]

        # Aggregate event-level predictions by detector geometry/fidelity
        for group_key, indices in group_data.groupby(group_columns, sort=False).indices.items():
            indices = np.asarray(indices, dtype=np.int64)

            if not isinstance(group_key, tuple):
                group_key = (group_key,)

            if group_key not in accumulators:
                accumulators[group_key] = {
                    "n_samples": 0,
                    "probability_sum": np.zeros(n_shells, dtype=np.float64),
                    "uncertainty_sq_sum": np.zeros(n_shells, dtype=np.float64),
                    "truth_count": np.zeros(n_shells, dtype=np.float64),
                }

            state = accumulators[group_key]
            state["n_samples"] += len(indices)
            state["probability_sum"] += probabilities[indices].sum(axis=0)
            state["uncertainty_sq_sum"] += np.square(uncertainties[indices]).sum(axis=0)
            state["truth_count"] += np.bincount(true_shell[indices], minlength=n_shells)

        total_events += n_events

    # Convert accumulated values into one row per shell
    rows: list[dict] = []
    for group_key, state in accumulators.items():
        split_value, fidelity_value, detector_R_value, detector_Z_value, detector_z_center_value = group_key
        n_samples = int(state["n_samples"])

        if n_samples <= 0:
            continue

        y_cnp = state["probability_sum"] / n_samples
        y_cnp_err = np.sqrt(state["uncertainty_sq_sum"] / n_samples)
        y_raw = state["truth_count"] / n_samples

        for class_index in range(n_shells):
            rows.append({
                "iteration": int(iteration),
                "split_name": str(split_value),
                "fidelity": int(fidelity_value),
                "n_samples": n_samples,
                "detector_R": float(detector_R_value),
                "detector_Z": float(detector_Z_value),
                "detector_z_center": float(detector_z_center_value),
                "shell_index": class_index + 1,
                "y_cnp": float(y_cnp[class_index]),
                "y_cnp_err": float(y_cnp_err[class_index]),
                "y_raw": float(y_raw[class_index]),
                "source_file": "aggregated_event_shell_predictions",
            })

    if not rows:
        raise RuntimeError(f"CNP prediction produced no shell rows for split {split!r}")

    prediction_df = pd.DataFrame(rows)
    prediction_df = prediction_df.sort_values(["split_name", "fidelity", "detector_R", "detector_Z", "shell_index"]).reset_index(drop=True)
    prediction_df.to_csv(output_path, index=False)

    print()
    print(f"Prediction split:  {split}")
    print(f"Predicted events:  {total_events:,}")
    print(f"Prediction groups:  {len(accumulators):,}")
    print(f"Saved predictions: {output_path}")

    return ShellPredictionResults(
        prediction_path=output_path,
        n_events=total_events,
        n_groups=len(accumulators)
    )

def run_shell_mfgp_training(
    cnp_prediction_path: str | Path,
    mfgp_config_path: str | Path,
    *,
    iteration: int = 0,
) -> mfgp.MFGPTrainResult:
    """Train the generic MF-GP using shell containment data."""
    raw = mfgp.load_config(mfgp_config_path)
    shell_cfg = raw.get("shell", {})
    n_shell = int(shell_cfg.get("n_shell", 20))
    if n_shell <= 0:
        raise ValueError(f"MFGP config shell.n_shell must be positive, got {n_shell}")

    training_data = build_shell_mfgp_training_data(
        cnp_prediction_path,
        shell_n=n_shell,
        iteration=iteration,
    )

    return mfgp.run_mfgp_training(
        mfgp_config_path,
        training_data,
    )

def run_shell_mfgp_prediction(
    cnp_prediction_path: str | Path,
    mfgp_config_path: str | Path,
    *,
    model_path: str | Path,
    iteration: int = 0,
    output_path: str | Path | None = None,
    metrics_path: str | Path | None = None,
) -> mfgp.MFGPPredictionResults:
    """Evaluate a trained generic MF-GP on held-out shell data."""
    raw = mfgp.load_config(mfgp_config_path)
    shell_cfg = raw.get("shell", {})
    n_shell = int(shell_cfg.get("n_shell", 20))
    if n_shell <= 0:
        raise ValueError(f"MFGP config shell.n_shell must be positive, got {n_shell}")

    validation_data = build_shell_mfgp_validation_data(
        cnp_prediction_path,
        shell_n=n_shell,
        iteration=iteration,
    )
    
    return mfgp.run_mfgp_prediction(
        mfgp_config_path,
        model_path=model_path,
        x=validation_data.x,
        y_true=validation_data.y_true,
        output_path=output_path,
        metrics_path=metrics_path,
    )