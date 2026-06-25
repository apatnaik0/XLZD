"""
Stores functions to help with geometry of shells

This counts whether they are inside a shell, what the shell boundaries are, etc.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common.config import ShellConfig
from common.theta import Z_FROM_CENTER_COLUMN

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

def build_shell_table(df_for_support: pd.DataFrame, shell_cfg: ShellConfig) -> pd.DataFrame:
    boundaries = shell_boundaries(shell_cfg)
    support_rows: list[dict[str, float | int]] = []
    shell_volume = 2.0 * np.pi * shell_cfg.Z_max * (shell_cfg.R_max**2) / float(shell_cfg.n_shells)

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

def update_detector_maximums(
    df: pd.DataFrame,
    shell_cfg: ShellConfig,
    z_center: float,
) -> None:
    # Find Z and R Maximum
    if "r" not in df.columns or "z" not in df.columns:
        raise ValueError("Dataframe must contain 'z' and 'r' to infer detector maximums")
    if shell_cfg.R_max is None:
        shell_cfg.R_max = df['r'].max()
    if shell_cfg.Z_max is None:
        shell_cfg.Z_max = np.abs(df['z'] - z_center).max()
    
def infer_centered_z_coordinate(df: pd.DataFrame, shell_cfg: ShellConfig) -> float:
    if "z" not in df.columns:
        raise ValueError("Dataframe must contain 'z' to infer centered coordinates.")
    if shell_cfg.z_center is not None:
        return float(shell_cfg.z_center)
    else:
        z_center = 0.5 * (df["z"].min() + df["z"].max())
        shell_cfg.z_center = z_center
        return z_center