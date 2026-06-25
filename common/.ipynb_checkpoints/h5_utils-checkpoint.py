"""
This file stores common utilities used to save files to h5 file format from input CSVs
"""
from __future__ import annotations

import re

import h5py
import numpy as np
import pandas as pd

def safe_h5_dataset_name(name: object, used_names: set[str]) -> str:
    raw = str(name)
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("_")

    if not safe:
        safe = "column"

    candidate = safe
    counter = 1

    while candidate in used_names:
        candidate = f"{safe}_{counter}"
        counter += 1

    used_names.add(candidate)
    return candidate

def write_shell_table_group(
    h5: h5py.File | h5py.Group,
    shell_table_df: pd.DataFrame,
    group_name: str = "shell_table",
) -> None:
    shell_group = h5.create_group(group_name)

    for col in shell_table_df.columns:
        values = shell_table_df[col]

        if pd.api.types.is_integer_dtype(values):
            data = values.to_numpy(dtype=np.int64)
        else:
            data = values.to_numpy(dtype=np.float64)

        shell_group.create_dataset(
            col,
            data=data,
            compression="gzip",
            compression_opts=4,
        )

def write_dataframe_group(
    h5: h5py.File | h5py.Group,
    group_name: str,
    df: pd.DataFrame,
) -> None:
    group = h5.create_group(group_name)

    used_names: set[str] = set()
    column_names = [str(col) for col in df.columns]
    dataset_names = [safe_h5_dataset_name(col, used_names) for col in column_names]

    group.create_dataset("column_names", data=np.asarray(column_names, dtype="S"))
    group.create_dataset("dataset_names", data=np.asarray(dataset_names, dtype="S"))

    for original_col, dataset_name in zip(df.columns, dataset_names):
        values = df[original_col]

        if pd.api.types.is_bool_dtype(values):
            data = values.to_numpy(dtype=np.bool_)

        elif pd.api.types.is_integer_dtype(values):
            data = values.to_numpy(dtype=np.int64)

        elif pd.api.types.is_float_dtype(values):
            data = values.to_numpy(dtype=np.float64)

        else:
            string_dtype = h5py.string_dtype(encoding="utf-8")
            data = values.astype("string").fillna("").astype(str).to_numpy()
            group.create_dataset(dataset_name, data=data, dtype=string_dtype)
            continue

        group.create_dataset(dataset_name, data=data, compression="gzip", compression_opts=4,)