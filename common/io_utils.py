"""Input/output helpers for XLZD event files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .config import EVENT_ID_COLUMN, EXPECTED_COLUMNS, FileLoadConfig

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - fallback when tqdm is unavailable
    def tqdm(iterable, **_: object):  # type: ignore[misc]
        return iterable


@dataclass(slots=True)
class LoadedEventCollection:
    """Container holding per-component dataframes and the concatenated table."""

    per_component: dict[str, pd.DataFrame]
    concatenated: pd.DataFrame
    files_loaded: list[Path]


def _discover_files(config: FileLoadConfig) -> list[Path]:
    """Find input files from the configured directory and stem list."""

    available = {path.stem: path for path in config.input_dir.glob(config.glob_pattern)}
    selected: list[Path] = []
    missing: list[str] = []

    for stem in config.file_stems:
        path = available.get(stem)
        if path is None:
            missing.append(stem)
            continue
        selected.append(path)

    if missing:
        missing_text = ", ".join(missing)
        raise FileNotFoundError(
            f"Could not find the following expected files in {config.input_dir}: {missing_text}"
        )

    if not selected:
        raise FileNotFoundError(f"No input files were discovered in {config.input_dir}.")

    return selected


def _try_read_table(path: Path, nrows: Optional[int]) -> pd.DataFrame:
    """Read a tabular event file with a couple of delimiter/header strategies."""

    read_attempts = (
        {"sep": ",", "nrows": nrows, "low_memory": False},
        {"sep": None, "engine": "python", "nrows": nrows},
        {"sep": r"\s+", "engine": "python", "nrows": nrows},
        {"header": None, "sep": ",", "nrows": nrows, "low_memory": False},
        {"header": None, "sep": None, "engine": "python", "nrows": nrows},
        {"header": None, "sep": r"\s+", "engine": "python", "nrows": nrows},
    )

    errors: list[str] = []
    for kwargs in read_attempts:
        try:
            return pd.read_csv(path, **kwargs)
        except Exception as exc:  # pragma: no cover - defensive path
            errors.append(f"{kwargs}: {exc}")

    joined_errors = "\n".join(errors)
    raise ValueError(f"Unable to parse tabular file {path}.\nTried:\n{joined_errors}")


def _normalize_columns(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Normalize column names while preserving the leading global event id column."""

    normalized = df.copy()
    normalized.columns = [str(column).strip() for column in normalized.columns]

    rename_map = {
        "": EVENT_ID_COLUMN,
        "Unnamed: 0": EVENT_ID_COLUMN,
        "unnamed: 0": EVENT_ID_COLUMN,
        "eventid": EVENT_ID_COLUMN,
        "event_id": EVENT_ID_COLUMN,
        "global_event_id": EVENT_ID_COLUMN,
        "eTPC": "ETPC",
        "etpc": "ETPC",
        "Etpc": "ETPC",
    }
    normalized = normalized.rename(columns=rename_map)

    expected_with_id = [EVENT_ID_COLUMN, *EXPECTED_COLUMNS]
    expected_set = set(EXPECTED_COLUMNS)
    current_set = set(normalized.columns)

    if set(expected_with_id).issubset(current_set):
        normalized = normalized.loc[:, expected_with_id]
    elif expected_set.issubset(current_set) and EVENT_ID_COLUMN not in current_set:
        normalized.insert(0, EVENT_ID_COLUMN, np.arange(len(normalized), dtype=np.int64))
        normalized = normalized.loc[:, expected_with_id]
    elif normalized.shape[1] == len(expected_with_id):
        normalized.columns = expected_with_id
    elif normalized.shape[1] == len(EXPECTED_COLUMNS):
        normalized.columns = list(EXPECTED_COLUMNS)
        normalized.insert(0, EVENT_ID_COLUMN, np.arange(len(normalized), dtype=np.int64))
        normalized = normalized.loc[:, expected_with_id]
    else:
        raise ValueError(
            f"{path} could not be mapped onto the expected event schema. "
            f"Observed columns: {list(normalized.columns)}"
        )

    normalized[EVENT_ID_COLUMN] = pd.to_numeric(normalized[EVENT_ID_COLUMN], errors="coerce")
    for column in EXPECTED_COLUMNS:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

    required_columns = [EVENT_ID_COLUMN, *EXPECTED_COLUMNS]
    invalid_mask = normalized[required_columns].isna().any(axis=1)
    invalid_count = int(invalid_mask.sum())
    if invalid_count:
        normalized = normalized.loc[~invalid_mask].copy()
        if normalized.empty:
            raise ValueError(
                f"{path} contains only non-numeric or malformed rows after parsing."
            )
        print(
            f"[warn] Dropped {invalid_count} malformed row(s) from {path.name} "
            f"after numeric conversion."
        )

    normalized[EVENT_ID_COLUMN] = normalized[EVENT_ID_COLUMN].astype(np.int64)

    return normalized


def infer_component_name(path: Path) -> str:
    """Infer detector component name from the file stem."""

    stem = path.stem
    marker = "_2447keVgamma"
    if marker in stem:
        return stem.split(marker, maxsplit=1)[0]
    return stem


def load_event_file(path: Path, *, max_rows: Optional[int] = None) -> pd.DataFrame:
    """Load a single event file and append metadata columns."""

    raw = _try_read_table(path, nrows=max_rows)
    df = _normalize_columns(raw, path)
    df["r"] = np.sqrt(df["x"].to_numpy() ** 2 + df["y"].to_numpy() ** 2)
    df["s_r"] = np.sqrt(df["sx"].to_numpy() ** 2 + df["sy"].to_numpy() ** 2)
    df["source_component"] = infer_component_name(path)
    df["source_file"] = path.name
    return df


def load_event_collection(config: FileLoadConfig) -> LoadedEventCollection:
    """Load all configured event files and return both grouped and concatenated views."""

    config.validate()
    files = _discover_files(config)
    per_component: dict[str, pd.DataFrame] = {}

    iterator = tqdm(files, total=len(files), desc="Loading event files")
    for path in iterator:
        component = infer_component_name(path)
        per_component[component] = load_event_file(path, max_rows=config.max_rows_per_file)

    concatenated = pd.concat(per_component.values(), ignore_index=True, sort=False)
    return LoadedEventCollection(
        per_component=per_component,
        concatenated=concatenated,
        files_loaded=files,
    )


def save_dataframe(df: pd.DataFrame, path: Path, output_format: str) -> Path:
    """Persist a dataframe as CSV or Parquet and return the written path."""

    output_format = output_format.lower()
    if output_format not in {"csv", "parquet"}:
        raise ValueError("output_format must be 'csv' or 'parquet'.")

    path.parent.mkdir(parents=True, exist_ok=True)

    if output_format == "csv":
        final_path = path.with_suffix(".csv")
        df.to_csv(final_path, index=False)
        return final_path

    final_path = path.with_suffix(".parquet")
    try:
        df.to_parquet(final_path, index=False)
    except Exception as exc:  # pragma: no cover - depends on optional parquet engine
        raise RuntimeError(
            "Failed to write parquet output. Install a parquet engine such as pyarrow "
            "or switch --output-format to csv."
        ) from exc
    return final_path
