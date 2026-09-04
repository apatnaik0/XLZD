"""
Stores functions to help with geometry of shells

This counts whether they are inside a shell, what the shell boundaries are, etc.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common.config import ShellConfig
from common.theta import Z_FROM_CENTER_COLUMN

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