"""XLZD RESuM-style data preparation package."""

from .config import (
    DEFAULT_FILE_STEMS,
    FileLoadConfig,
    OutputConfig,
    PipelineConfig,
    SamplingConfig,
    SplitConfig,
    ThetaSamplingConfig,
)
from .dataset import (
    BlockSet,
    PoolSplitResult,
    build_theta_sets_from_blocks,
    split_into_disjoint_pools,
    split_pool_into_blocks,
    write_theta_block_files,
)
from .io_utils import load_event_collection, save_dataframe
from .theta import (
    Z_FROM_CENTER_COLUMN,
    ThetaRegion,
    add_centered_z_coordinate,
    infer_theta_bounds,
    infer_z_center,
    sample_theta_region,
    theta_mask,
)

__all__ = [
    "DEFAULT_FILE_STEMS",
    "FileLoadConfig",
    "OutputConfig",
    "PipelineConfig",
    "SamplingConfig",
    "SplitConfig",
    "ThetaSamplingConfig",
    "BlockSet",
    "PoolSplitResult",
    "ThetaRegion",
    "Z_FROM_CENTER_COLUMN",
    "add_centered_z_coordinate",
    "build_theta_sets_from_blocks",
    "infer_theta_bounds",
    "infer_z_center",
    "load_event_collection",
    "sample_theta_region",
    "save_dataframe",
    "split_into_disjoint_pools",
    "split_pool_into_blocks",
    "theta_mask",
    "write_theta_block_files",
]
