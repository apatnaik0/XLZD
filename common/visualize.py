"""
Visualization utilities for the XLZD CNP / shell workflow.

This module is intentionally visualization-only.

Shell definitions and shell assignment are owned by ``position_shells.py``.
This file may construct a ShellConfig for a plot, but it does not independently
implement shell boundaries or shell membership.

Current plotting functionality:
    - shell histograms
    - CNP predicted-vs-true shell occupancy
    - input shell occupancy with exponential fit
    - interactive 3D source/end-point visualization
"""

from __future__ import annotations

import glob
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.optimize import curve_fit
from matplotlib.colors import LogNorm
from matplotlib.ticker import FuncFormatter

from shells import (
    ShellConfig,
    build_shell_table,
    positive_shells_for_block,
)

from common.theta import add_centered_z_coordinate


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def resolve_input_paths(
    inpath: str | Path | Sequence[str | Path],
) -> list[Path]:
    """Resolve explicit paths and glob patterns used by visualization functions."""
    if isinstance(inpath, (str, Path)):
        inpath = [inpath]

    paths: list[Path] = []

    for item in inpath:
        matches = glob.glob(str(item))

        if matches:
            paths.extend(Path(match) for match in matches)
        else:
            paths.append(Path(item))

    return paths

def _load_plot_dataframe(
    inpath: str | Path | Sequence[str | Path],
) -> pd.DataFrame:
    """
    Load one or more CSV files for plotting.

    This helper only creates quantities needed by the visualizations. It does
    not define shells or assign shell labels.
    """
    csv_files = resolve_input_paths(inpath)
    if not csv_files:
        raise ValueError("No input files provided.")
    frames: list[pd.DataFrame] = []

    for file_path in csv_files:
        if not file_path.exists():
            raise FileNotFoundError(file_path)
        temp_df = pd.read_csv(file_path)
        required = {"x", "y", "z"}
        missing = required - set(temp_df.columns)
        if missing:
            raise ValueError(f"{file_path.name} is missing required plotting columns: {sorted(missing)}")
        temp_df = temp_df.copy()
        temp_df["r"] = np.sqrt(temp_df["x"].to_numpy(dtype=float) ** 2 + temp_df["y"].to_numpy(dtype=float) ** 2)
        temp_df["filename"] = file_path.stem
        frames.append(temp_df)
        
    return pd.concat(frames, ignore_index=True, sort=False)

def _mfgp_frame(result_or_frame):
    """Accept either MFGPPredictionResults or a DataFrame."""
    if isinstance(result_or_frame, pd.DataFrame):
        return result_or_frame.copy()
    if hasattr(result_or_frame, "frame"):
        return result_or_frame.frame.copy()
    raise TypeError("Expected MFGPPredictionResults or pandas DataFrame")

# -----------------------------------------------------------------------------
# Plot Utilities
# -----------------------------------------------------------------------------
def _exponential(x: np.ndarray, amplitude: float, exponent: float) -> np.ndarray:
    """Exponential model used only for the shell-occupancy visualization."""
    return amplitude * np.exp(exponent * x)

def _exponential_regression(
    x_data: np.ndarray,
    y_data: np.ndarray,
    p0: tuple[float, float] = (0.1, 0.2),
    maxfev: int = 10_000,
) -> tuple[float, float]:
    """Fit the exponential curve displayed by ``plot_input_shell_occupancy``."""
    popt, _pcov = curve_fit(
        _exponential,
        np.asarray(x_data, dtype=float),
        np.asarray(y_data, dtype=float),
        p0=p0,
        maxfev=maxfev,
    )

    return float(popt[0]), float(popt[1])

def _cylinder_surface(
    r: float,
    h: float,
    a: float = 0.0,
    nt: int = 100,
    nv: int = 50,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create display coordinates for a cylindrical Plotly surface."""
    theta = np.linspace(0.0, 2.0 * np.pi, nt)
    vertical = np.linspace(a, a + h, nv)
    theta, vertical = np.meshgrid(theta, vertical)

    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = vertical

    return x, y, z

def _positive_plot_floor(*arrays):
    """Small positive value used only when displaying data on a log scale."""
    positive = []

    for array in arrays:
        values = np.asarray(array, dtype=float)
        values = values[np.isfinite(values) & (values > 0)]
        if len(values):
            positive.append(values)

    if not positive:
        return 1e-12

    return max(min(np.min(values) for values in positive) * 0.5, 1e-12)

from matplotlib.ticker import FuncFormatter

def _format_log_colorbar(cbar, vmin, vmax, n_ticks=5):
    """Use evenly spaced log ticks with short, rounded labels."""

    ticks = np.geomspace(vmin, vmax, n_ticks)
    cbar.set_ticks(ticks)

    def format_tick(x, pos=None):
        if x == 0:
            return "0"

        exponent = int(np.floor(np.log10(abs(x))))
        coefficient = x / 10**exponent

        if -2 <= exponent <= 2:
            return f"{x:.3g}"

        return rf"${coefficient:.1f}\times10^{{{exponent}}}$"

    cbar.ax.yaxis.set_major_formatter(FuncFormatter(format_tick))
    cbar.ax.minorticks_off()
# -----------------------------------------------------------------------------
# CHP Plots
# -----------------------------------------------------------------------------
def plot_shell_histogram(
    data: np.ndarray,
    shell_end: int,
    title: str,
    shell_start: int = 0,
    ax=None,
) -> None:
    """Plot an integer-binned shell-index histogram."""
    if ax is None:
        _fig, ax = plt.subplots(figsize=(10, 4))

    bins = np.arange(shell_start - 0.5, shell_end + 1.5, 1)

    ax.hist(data, bins=bins, edgecolor="black")
    ax.set_xlim(shell_start - 0.5, shell_end + 0.5)
    ax.set_xlabel("Shell Number")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

def plot_cnp_pred_shell_occupancy(
    mfgp_path: str | Path,
    outpath: str | Path | None = None,
) -> tuple[plt.Figure, list[plt.Axes]]:
    """
    Plot shell occupancy using the aggregated prediction CSV.

    Required columns:
        shell_index
        y_cnp
        y_raw
        n_samples

    Panels:
        1. True shell occupancy
        2. Summed predicted shell probabilities
        3. Fractional difference: (predicted - true) / true
    """
    mfgp_path = Path(mfgp_path)
    if not mfgp_path.exists():
        raise FileNotFoundError(f"Prediction CSV does not exist: {mfgp_path}")
    prob_data = pd.read_csv(mfgp_path)
    required_columns = {
        "shell_index",
        "y_cnp",
        "y_raw",
        "n_samples",
    }

    missing_columns = required_columns - set(prob_data.columns)
    if missing_columns:
        raise ValueError(f"Prediction CSV is missing required columns: {sorted(missing_columns)}")
    if prob_data.empty:
        raise ValueError(f"Prediction CSV contains no rows: {mfgp_path}")
    for column in required_columns:
        prob_data[column] = pd.to_numeric(prob_data[column], errors="coerce")
    if prob_data[list(required_columns)].isna().any().any():
        bad_columns = [
            column
            for column in required_columns
            if prob_data[column].isna().any()
        ]
        raise ValueError(f"Prediction CSV contains missing or non-numeric values in: {sorted(bad_columns)}")

    if (prob_data["n_samples"] < 0).any():
        raise ValueError("Prediction CSV contains negative n_samples values")
    if (prob_data["shell_index"] < 1).any():
        raise ValueError("shell_index must use one-based shell numbering starting at 1")

    # y_cnp and y_raw are group means. Multiplying by n_samples restores
    # the expected count contribution from each group.
    prob_data["predicted_count"] = prob_data["y_cnp"] * prob_data["n_samples"]
    prob_data["true_count"] = prob_data["y_raw"] * prob_data["n_samples"]
    shell_totals = (
        prob_data
        .groupby("shell_index", as_index=True)
        .agg(
            predicted_count=("predicted_count", "sum"),
            true_count=("true_count", "sum"),
        ).sort_index())

    shell_end = int(shell_totals.index.max())
    shell_numbers = np.arange(1, shell_end + 1, dtype=np.int32)
    shell_totals = shell_totals.reindex(shell_numbers, fill_value=0.0)
    predicted_counts = shell_totals["predicted_count"].to_numpy(dtype=float)
    true_counts = shell_totals["true_count"].to_numpy(dtype=float)

    # Avoid inf/nan for shells with no true events.
    count_difference = np.divide(
        predicted_counts - true_counts,
        true_counts,
        out=np.full_like(true_counts, np.nan, dtype=float),
        where=true_counts != 0,
    )

    predicted_total = float(predicted_counts.sum())
    true_total = float(true_counts.sum())
    total_difference = predicted_total - true_total
    print(f"Predicted minus truth total: {total_difference:,.2f}")
    fig, (ax_truth, ax_predicted, ax_difference,) = plt.subplots(3, 1, figsize=(12, 11), sharex=True)

    positive_values = np.concatenate([true_counts[true_counts > 0], predicted_counts[predicted_counts > 0]])
    max_val = float(positive_values.max()) if len(positive_values) else 1.0

    # True shell occupancy
    ax_truth.bar(
        shell_numbers,
        true_counts,
        edgecolor="black",
    )
    ax_truth.set_title("True Shell Distribution")
    ax_truth.set_ylabel("Event Count")
    ax_truth.set_yscale("log")
    ax_truth.set_ylim(1, max(1.0, max_val))
    ax_truth.set_axisbelow(True)
    ax_truth.grid(True, which="major", alpha=0.3)

    # Summed predicted shell probabilities
    ax_predicted.bar(
        shell_numbers,
        predicted_counts,
        edgecolor="black",
    )
    ax_predicted.set_title("Summed Predicted Shell Probabilities")
    ax_predicted.set_ylabel("Expected Count")
    ax_predicted.set_yscale("log")
    ax_predicted.set_ylim(1, max(1.0, max_val))
    ax_predicted.set_axisbelow(True)
    ax_predicted.grid(True, which="major", alpha=0.3)

    # Fractional difference between prediction and truth
    ax_difference.bar(
        shell_numbers,
        count_difference,
        edgecolor="black",
    )
    ax_difference.set_title("Occupancy Difference: Predicted - True")
    ax_difference.set_xlabel("Shell Number")
    ax_difference.set_ylabel("Fractional Difference")
    ax_difference.set_axisbelow(True)
    ax_difference.grid(True, which="major", alpha=0.3)

    axes = [
        ax_truth,
        ax_predicted,
        ax_difference,
    ]

    for ax in axes:
        ax.set_xlim(0.5, shell_end + 0.5)
    fig.suptitle("CNP Predicted vs True Shell Occupancy", fontsize=14)
    fig.tight_layout()
    if outpath is not None:
        outpath = Path(outpath)
        outpath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            outpath,
            dpi=200,
            bbox_inches="tight")

    return fig, axes

def plot_input_shell_occupancy(
    nshells: int,
    inpath: str | Path | Sequence[str | Path],
    outpath: str | Path,
    scale_power: float = 1.0 / 3.0,
    *,
    R_max: float | None = None,
    Z_max: float | None = None,
    z_center: float | None = None,
) -> None:
    """
    Plot the true shell occupancy of raw input events.

    Shell construction and shell membership are delegated to
    ``position_shells.py``.

    Parameters
    ----------
    nshells:
        Number of shell classes.
    inpath:
        Input CSV path, glob, or collection of paths.
    outpath:
        Output image path.
    scale_power:
        Shell scale power passed directly to ``ShellConfig``.
    R_max, Z_max, z_center:
        Optional detector geometry.

        If omitted, the same geometry assumptions used by the original
        visualization are retained:
            R_max    = ceil(max event radius)
            Z_max    = ceil(max z / 2)
            z_center = Z_max
    """
    df = _load_plot_dataframe(inpath)

    if R_max is None:
        R_max = float(np.ceil(df["r"].max()))

    if Z_max is None:
        Z_max = float(np.ceil(df["z"].max() / 2.0))

    if z_center is None:
        z_center = float(Z_max)

    shell_cfg = ShellConfig(
        R_max=float(R_max),
        Z_max=float(Z_max),
        n_shells=int(nshells),
        min_candidate_events=1,
        z_center=float(z_center),
        scale_power=float(scale_power),
    )
    shell_cfg.validate()

    # Centered-z calculation is shared preprocessing; shell definitions and
    # assignments remain in position_shells.py.
    shell_df = add_centered_z_coordinate(
        df,
        float(z_center),
    )

    shell_table = build_shell_table(
        shell_df,
        shell_cfg,
    )

    shell_index = positive_shells_for_block(
        block_df=shell_df,
        shell_table_df=shell_table,
    )

    valid_shells = (
        shell_index
        .dropna()
        .astype(np.int32)
    )

    x_data = np.arange(1, nshells + 1, dtype=np.int32)

    occupancy = (
        valid_shells
        .value_counts()
        .reindex(range(1, nshells + 1), fill_value=0)
        .sort_index()
        .to_numpy(dtype=float)
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(
        x_data,
        occupancy,
        edgecolor="black",
        label="Shell occupancy",
    )

    # Keep the same exponential overlay as the original visualization.
    # A fit can fail for a degenerate / tiny dataset, in which case the
    # occupancy plot remains useful by itself.
    try:
        amplitude, exponent = _exponential_regression(
            x_data,
            occupancy,
        )
        regression = _exponential(
            x_data,
            amplitude,
            exponent,
        )

        ax.plot(
            x_data,
            regression,
            label="Exponential fit",
        )

        ax.annotate(
            f"Regression: {amplitude:.4g} * e^({exponent:.4g}x)",
            xy=(0.05, 0.90),
            xycoords="axes fraction",
        )
    except (RuntimeError, ValueError, FloatingPointError) as exc:
        print(f"[warn] Exponential shell-occupancy fit failed: {exc}")

    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_xlabel("Shell Number")
    ax.set_ylabel("Count")
    ax.set_title("Shell Occupation")
    ax.set_xlim(0.5, nshells + 0.5)

    fig.tight_layout()

    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        outpath,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)

# -----------------------------------------------------------------------------
# MFGP Plots
# -----------------------------------------------------------------------------
def plot_mfgp_mean_uncertainty(
    grid_result,
    x_col="detector_R",
    y_col="detector_Z",
    hf_points=None,
    outpath=None,
):
    """Plot MF-GP predicted HF mean and 1-sigma uncertainty over a 2D grid."""
    df = _mfgp_frame(grid_result)

    required = {x_col, y_col, "mf_prediction", "mf_std"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    x = np.sort(df[x_col].unique())
    y = np.sort(df[y_col].unique())
    if len(x) * len(y) != len(df):
        raise ValueError("MF-GP prediction points must form a complete rectangular grid")

    df = df.sort_values([y_col, x_col]).reset_index(drop=True)
    mean = df["mf_prediction"].to_numpy(dtype=float).reshape(len(y), len(x))
    std = df["mf_std"].to_numpy(dtype=float).reshape(len(y), len(x))

    mean_eps = _positive_plot_floor(mean)
    std_eps = _positive_plot_floor(std)

    mean_plot = np.maximum(mean, mean_eps)
    std_plot = np.maximum(std, std_eps)

    mean_min, mean_max = mean_plot.min(), mean_plot.max()
    std_min, std_max = std_plot.min(), std_plot.max()

    if np.isclose(mean_min, mean_max):
        mean_max = mean_min * 1.01
    if np.isclose(std_min, std_max):
        std_max = std_min * 1.01

    mean_levels = np.geomspace(mean_min, mean_max, 25)
    std_levels = np.geomspace(std_min, std_max, 25)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharex=True, sharey=True)

    im0 = axes[0].contourf(x, y, mean_plot, levels=mean_levels,
                           norm=LogNorm(vmin=mean_min, vmax=mean_max))
    im1 = axes[1].contourf(x, y, std_plot, levels=std_levels,
                           norm=LogNorm(vmin=std_min, vmax=std_max), cmap="Reds")

    cbar0 = fig.colorbar(im0, ax=axes[0])
    cbar1 = fig.colorbar(im1, ax=axes[1])
    
    cbar0.set_label("MF-GP prediction")
    cbar1.set_label("1σ uncertainty")
    
    _format_log_colorbar(cbar0, mean_min, mean_max)
    _format_log_colorbar(cbar1, std_min, std_max)

    axes[0].set_title("Predicted HF Mean")
    axes[1].set_title("Predicted HF Uncertainty")

    if hf_points is not None:
        if isinstance(hf_points, pd.DataFrame):
            points = hf_points[[x_col, y_col]].to_numpy(dtype=float)
        else:
            points = np.asarray(hf_points, dtype=float)

        for ax in axes:
            ax.scatter(points[:, 0], points[:, 1], facecolors="white",
                       edgecolors="black", s=35, label=f"HF training points (n={len(points)})")
            ax.legend()

    for ax in axes:
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_xlim(0, x.max())
        ax.set_ylim(0, y.max())
        ax.grid(alpha=0.2)

    fig.suptitle("MF-GP High-Fidelity Prediction")
    fig.tight_layout()

    if outpath is not None:
        outpath = Path(outpath)
        outpath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outpath, dpi=180)

    return fig, axes

def plot_mfgp_validation_points(
    result,
    x_cols=("detector_R", "detector_Z"),
    outpath=None,
):
    """Plot HF truth and MF-GP prediction with 1σ, 2σ, and 3σ ranges."""

    df = _mfgp_frame(result)
    required = {*x_cols, "y_true", "mf_prediction", "mf_std"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df.sort_values(list(x_cols)).reset_index(drop=True)

    truth = df["y_true"].to_numpy(dtype=float)
    pred = df["mf_prediction"].to_numpy(dtype=float)
    std = df["mf_std"].to_numpy(dtype=float)
    idx = np.arange(len(df))

    # Sigma intervals
    lower1, upper1 = pred - std, pred + std
    lower2, upper2 = pred - 2*std, pred + 2*std
    lower3, upper3 = pred - 3*std, pred + 3*std

    # Coverage
    inside1 = (truth >= lower1) & (truth <= upper1)
    inside2 = (truth >= lower2) & (truth <= upper2)
    inside3 = (truth >= lower3) & (truth <= upper3)

    n = len(truth)
    cov1, cov2, cov3 = inside1.sum(), inside2.sum(), inside3.sum()

    # Floor only for log-scale plotting
    eps = _positive_plot_floor(truth, pred, lower1, lower2, lower3)
    truth_plot = np.maximum(truth, eps)
    pred_plot = np.maximum(pred, eps)

    lower1_plot, upper1_plot = np.maximum(lower1, eps), np.maximum(upper1, pred_plot)
    lower2_plot, upper2_plot = np.maximum(lower2, eps), np.maximum(upper2, pred_plot)
    lower3_plot, upper3_plot = np.maximum(lower3, eps), np.maximum(upper3, pred_plot)

    fig_width = max(10, min(20, 0.48*len(df) + 5))
    fig, ax = plt.subplots(figsize=(fig_width, 6))

    # Residual: truth → prediction
    ax.vlines(idx, truth_plot, pred_plot, color="0.65", lw=1, alpha=0.7)

    # Nested uncertainty intervals: widest first
    ax.vlines(idx, lower3_plot, upper3_plot, color="tab:blue", lw=9, alpha=0.15)
    ax.vlines(idx, lower2_plot, upper2_plot, color="tab:blue", lw=6, alpha=0.25)
    ax.vlines(idx, lower1_plot, upper1_plot, color="tab:blue", lw=3, alpha=0.55)

    # Truth and predicted mean
    ax.scatter(idx, truth_plot, marker="x", s=55, color="black", linewidth=1.4, label="HF truth", zorder=5)
    ax.scatter(idx, pred_plot, marker="o", s=30, color="tab:blue", label="MF-GP prediction", zorder=6)

    labels = [
        "(" + ", ".join(f"{row[col]:g}" for col in x_cols) + ")"
        for _, row in df.iterrows()
    ]

    if len(df) <= 30:
        ax.set_xticks(idx)
        ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=8)
        ax.set_xlabel(f"HF geometry ({', '.join(x_cols)})")
    else:
        ax.set_xlabel("Validation geometry index")

    # Dummy artists make clean legend entries for sigma ranges
    ax.plot([], [], color="tab:blue", lw=3, alpha=0.55,
            label=f"1σ: {cov1}/{n} ({cov1/n:.0%})")
    ax.plot([], [], color="tab:blue", lw=6, alpha=0.25,
            label=f"2σ: {cov2}/{n} ({cov2/n:.0%})")
    ax.plot([], [], color="tab:blue", lw=9, alpha=0.15,
            label=f"3σ: {cov3}/{n} ({cov3/n:.0%})")

    ax.set_yscale("log")
    ax.set_ylabel("Containment probability")
    ax.set_title("MF-GP Validation: HF Truth vs Prediction")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best")

    fig.tight_layout()

    if outpath is not None:
        outpath = Path(outpath)
        outpath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outpath, dpi=180)

    return fig, ax

def plot_mfgp_validation_parity(result, outpath=None):
    """Plot MF-GP validation prediction against HF truth."""

    df = _mfgp_frame(result)

    required = {"y_true", "mf_prediction", "mf_std"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    truth = df["y_true"].to_numpy(dtype=float)
    pred = df["mf_prediction"].to_numpy(dtype=float)
    std = df["mf_std"].to_numpy(dtype=float)

    lower = pred - std
    upper = pred + std

    eps = _positive_plot_floor(truth, pred, lower, upper)

    truth_plot = np.maximum(truth, eps)
    pred_plot = np.maximum(pred, eps)
    lower_plot = np.maximum(lower, eps)
    upper_plot = np.maximum(upper, pred_plot)

    yerr = np.vstack((pred_plot - lower_plot, upper_plot - pred_plot))

    lo = min(truth_plot.min(), lower_plot.min())
    hi = max(truth_plot.max(), upper_plot.max())

    fig, ax = plt.subplots(figsize=(6.5, 6))

    ax.errorbar(truth_plot, pred_plot, yerr=yerr, fmt="o",
                ms=5, capsize=3, alpha=0.85)

    ax.plot([lo, hi], [lo, hi], "--", label="Perfect prediction")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)

    ax.set_xlabel("HF Truth")
    ax.set_ylabel("MF-GP Prediction")
    ax.set_title("MF-GP Validation Parity")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    fig.tight_layout()

    if outpath is not None:
        outpath = Path(outpath)
        outpath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outpath, dpi=180)

    return fig, ax