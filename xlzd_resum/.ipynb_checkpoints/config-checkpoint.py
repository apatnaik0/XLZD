"""Configuration objects for the XLZD pool/block theta-sampling pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

EVENT_ID_COLUMN = "global_event_id"
EXPECTED_COLUMNS: tuple[str, ...] = ("E0", "sx", "sy", "sz", "ETPC", "x", "y", "z")

DEFAULT_FILE_STEMS: tuple[str, ...] = (
    "CathodeGrid_2447keVgamma_ROI1sigma_SScut_vetoapplied",
    "FieldCage_2447keVgamma_ROI1sigma_SScut_vetoapplied",
    "GXe_2447keVgamma_ROI1sigma_SScut_vetoapplied",
    "ICV_2447keVgamma_ROI1sigma_SScut_vetoapplied",
    "OCV_2447keVgamma_ROI1sigma_SScut_vetoapplied",
    "TPCPMTTBottom_2447keVgamma_ROI1sigma_SScut_vetoapplied",
    "TPCPMTTop_2447keVgamma_ROI1sigma_SScut_vetoapplied",
)


@dataclass(slots=True)
class FileLoadConfig:
    """File discovery and parsing options."""

    input_dir: Path
    file_stems: Sequence[str] = field(default_factory=lambda: list(DEFAULT_FILE_STEMS))
    glob_pattern: str = "*.csv"
    max_rows_per_file: Optional[int] = None
    concatenate_components: bool = True

    def validate(self) -> None:
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory does not exist: {self.input_dir}")
        if not self.input_dir.is_dir():
            raise NotADirectoryError(f"Input path is not a directory: {self.input_dir}")
        if self.max_rows_per_file is not None and self.max_rows_per_file <= 0:
            raise ValueError("max_rows_per_file must be positive when provided.")


@dataclass(slots=True)
class SplitConfig:
    """Disjoint raw pool proportion settings."""

    lf_pool_fraction: float = 0.2
    hf_pool_fraction: float = 0.4
    random_seed: int = 42
    stratify_by_component: bool = False

    def validate(self) -> None:
        if not 0.0 < self.lf_pool_fraction < 1.0:
            raise ValueError("lf_pool_fraction must be strictly between 0 and 1.")
        if not 0.0 < self.hf_pool_fraction < 1.0:
            raise ValueError("hf_pool_fraction must be strictly between 0 and 1.")
        if self.lf_pool_fraction + self.hf_pool_fraction >= 1.0:
            raise ValueError("lf_pool_fraction + hf_pool_fraction must be smaller than 1.0.")


@dataclass(slots=True)
class ThetaSamplingConfig:
    """Centered-theta sampling bounds and geometric constraints."""

    z_lower: Optional[float] = None
    z_upper: Optional[float] = None
    r_lower: Optional[float] = None
    r_upper: Optional[float] = None
    min_z_width: float = 1.0
    min_r_width: float = 1.0
    z_center: Optional[float] = None

    def validate(self) -> None:
        if self.min_z_width <= 0:
            raise ValueError("min_z_width must be positive.")
        if self.min_r_width <= 0:
            raise ValueError("min_r_width must be positive.")
        if self.z_lower is not None and self.z_upper is not None and self.z_lower >= self.z_upper:
            raise ValueError("z_lower must be smaller than z_upper.")
        if self.r_lower is not None and self.r_upper is not None and self.r_lower >= self.r_upper:
            raise ValueError("r_lower must be smaller than r_upper.")


@dataclass(slots=True)
class SamplingConfig:
    """LF/HF/validation block settings."""

    hf_block_size: int = 100000
    lf_block_size: int = 10000
    validation_block_size: Optional[int] = None
    progress: bool = True

    def validate(self) -> None:
        if self.hf_block_size <= 0:
            raise ValueError("hf_block_size must be positive.")
        if self.lf_block_size <= 0:
            raise ValueError("lf_block_size must be positive.")
        if self.validation_block_size is not None and self.validation_block_size <= 0:
            raise ValueError("validation_block_size must be positive when provided.")


@dataclass(slots=True)
class OutputConfig:
    """Output file and serialization options."""

    output_dir: Path
    output_format: str = "csv"

    def validate(self) -> None:
        output_format = self.output_format.lower()
        if output_format not in {"csv", "parquet"}:
            raise ValueError("output_format must be either 'csv' or 'parquet'.")


@dataclass(slots=True)
class PipelineConfig:
    """Top-level configuration for the end-to-end pipeline."""

    file_load: FileLoadConfig
    split: SplitConfig
    theta: ThetaSamplingConfig
    sampling: SamplingConfig
    output: OutputConfig

    def validate(self) -> None:
        self.file_load.validate()
        self.split.validate()
        self.theta.validate()
        self.sampling.validate()
        self.output.validate()
