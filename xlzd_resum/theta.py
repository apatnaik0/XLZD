"""Centered theta definitions and membership utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .config import ThetaSamplingConfig

Z_FROM_CENTER_COLUMN = "z_from_center"


@dataclass(slots=True)
class ThetaRegion:
    """Centered theta region parameterized by radial and axial extents."""

    Z_max: float
    R_max: float

    def validate(self) -> None:
        if self.Z_max <= 0:
            raise ValueError("ThetaRegion requires Z_max > 0.")
        if self.R_max <= 0:
            raise ValueError("ThetaRegion requires R_max > 0.")

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(slots=True)
class ThetaBounds:
    """Sampling bounds for centered theta regions."""

    z_lower: float
    z_upper: float
    r_lower: float
    r_upper: float

    def validate(self) -> None:
        if self.z_lower >= self.z_upper:
            raise ValueError("ThetaBounds requires z_lower < z_upper.")
        if self.r_lower >= self.r_upper:
            raise ValueError("ThetaBounds requires r_lower < r_upper.")


def infer_z_center(df: pd.DataFrame, config: ThetaSamplingConfig) -> float:
    """Infer the chamber center in z from the observed range unless provided."""

    if "z" not in df.columns:
        raise ValueError("Dataframe must contain a 'z' column to infer z_center.")
    if config.z_center is not None:
        return float(config.z_center)
    return float(0.5 * (df["z"].min() + df["z"].max()))


def add_centered_z_coordinate(df: pd.DataFrame, z_center: float) -> pd.DataFrame:
    """Add the absolute distance from the inferred/provided z center."""

    out = df.copy()
    out[Z_FROM_CENTER_COLUMN] = np.abs(out["z"].to_numpy(dtype=float) - float(z_center))
    return out


def infer_theta_bounds(df: pd.DataFrame, config: ThetaSamplingConfig) -> ThetaBounds:
    """Infer centered-theta bounds from the data unless manual bounds are supplied."""

    required = {"r", Z_FROM_CENTER_COLUMN}
    if not required.issubset(df.columns):
        raise ValueError(f"Dataframe must contain columns {required} to infer theta bounds.")

    z_lower = float(config.min_z_width if config.z_lower is None else config.z_lower)
    z_upper = float(df[Z_FROM_CENTER_COLUMN].max()) if config.z_upper is None else float(config.z_upper)
    r_lower = float(config.min_r_width if config.r_lower is None else config.r_lower)
    r_upper = float(df["r"].max()) if config.r_upper is None else float(config.r_upper)

    bounds = ThetaBounds(
        z_lower=z_lower,
        z_upper=z_upper,
        r_lower=r_lower,
        r_upper=r_upper,
    )
    bounds.validate()
    return bounds


def sample_theta_region(rng: np.random.Generator, bounds: ThetaBounds) -> ThetaRegion:
    """Sample a valid centered theta region within the configured bounds."""

    bounds.validate()
    theta = ThetaRegion(
        Z_max=float(rng.uniform(bounds.z_lower, bounds.z_upper)),
        R_max=float(rng.uniform(bounds.r_lower, bounds.r_upper)),
    )
    theta.validate()
    return theta


def theta_mask(df: pd.DataFrame, theta: ThetaRegion) -> pd.Series:
    """Inclusive centered-theta membership mask using final-position coordinates only."""

    theta.validate()
    return (df[Z_FROM_CENTER_COLUMN] <= theta.Z_max) & (df["r"] <= theta.R_max)
