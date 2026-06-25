"""
Stores functions that help build the event class blocks
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from common.geometry import positive_shells_for_block

@dataclass(slots=True)
class ShellEventBlock:
    features: np.ndarray
    truth_shell: np.ndarray
    human_shell: np.ndarray
    event_index: np.ndarray
    valid_events: pd.DataFrame

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

    