# This file is meant to house visualization suites for different types of data to make it easy to feed in any kind of data and visualize it

from typing import Optional
import os, sys, glob
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
import plotly.graph_objects as go

def find_repo_root(start: Path | None = None) -> Path:
    start = (start or Path.cwd()).resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "PROJECT_EXPERIMENT_GUIDE.md").exists() and (candidate / "README.md").exists():
            return candidate
    raise RuntimeError("Could not find the XLZD repo root from the current working directory.")

REPO_ROOT = find_repo_root()
os.chdir(REPO_ROOT)
if str(REPO_ROOT / "src" / "run_cnp") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src" / "run_cnp"))
from cnp_clean_pipeline import PredictResult

###====================================###

### Path Management

###====================================###

def resolve_input_paths(
    inpath: str| Path | list[str|Path],
) -> list[Path]:
    if isinstance(inpath, (str, Path)):
        inpath = [inpath]

    paths: list[Path] = []
    for item in inpath:
        item_str = str(item)
        matches = glob.glob(item_str)
        if matches:
            paths.extend(Path(m) for m in matches)
        else:
            paths.append(Path(item))

    return paths

###====================================###

### Shell Calculations

###====================================###

def load_df_with_rt(
    csv_files: list[Path]
) -> pd.DataFrame:
    if not csv_files:
        raise ValueError("No input files provided")
    
    dfs = []
    for f in csv_files:
        temp_df = pd.read_csv(f)
        temp_df['sr'] = np.sqrt(temp_df['sx']**2 + temp_df['sy']**2)
        temp_df['st'] = np.arctan2(temp_df['sy'], temp_df['sx'])
        temp_df['r'] = np.sqrt(temp_df['x']**2 + temp_df['y']**2)
        temp_df['t'] = np.arctan2(temp_df['y'], temp_df['x'])
        temp_df['filename'] = Path(f).stem

        dfs.append(temp_df)
    df = pd.concat(dfs, ignore_index=True)

    return df

def find_shells(
    nshells: int,
    R_max: float,
    Z_max: float,
    scale_power: Optional[float] = 1.0/3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Finds the shells given a certain number of shells and positional maximums

    Returns:
        List of Z shell limits
        List of R shell limits
    """
    # Calculate fractions of shellnum over total shells
    int_array = np.arange(nshells) + 1
    fracs = int_array/nshells

    # Find scale modifier
    scale = np.power(fracs, scale_power)

    # Calculate shells and return
    shell_z = Z_max * scale
    shell_r = R_max * scale

    return shell_z, shell_r

def find_shell_occupations(
    data: pd.DataFrame,
    z_shells: np.ndarray,
    r_shells: np.ndarray,
) -> pd.DataFrame:
    """
    Takes in a dataframe and calculates the shell occupation numbers for specific shell bins

    Data Input:
        Data input requires and r,z column that are used to calculate shells

    Returns: 
        Same dataframe, just with new columns that have shell index information
    """
    # Add in a zero to the data to count shell edges
    z_shells = np.insert(z_shells, 0, 0.0)
    r_shells = np.insert(r_shells, 0, 0.0)
    
    # Iterate through each shell and fill out the dataframe
    df = data.copy()
    df["shell"] = -1
    for shell_num, (z,r) in enumerate(zip(z_shells[1:], r_shells[1:]), start=1):
        # Ranges
        z_max = z
        z_min = z_shells[shell_num-1]
        r_max = r
        r_min = r_shells[shell_num-1]

        # Conditions
        inside_outer = (df['r'] <= r_max) & (df['z'] <= z_max)
        inside_inner = (df['r'] <= r_min) & (df['z'] <= z_min)
        condition = inside_outer & ~inside_inner

        # Log shell #
        df.loc[condition, "shell"] = shell_num

    return df

###====================================###

### MISC

###====================================###

def exponential(x, a, s):
    return a*np.exp(s*x)

def exponential_regression(
    x_data: np.ndarray,
    y_data: np.ndarray, 
    p0: Optional[tuple[float]] = (0.1, 0.2),
    maxfev: Optional[int] = 10000
):
    """
    Finds a regression fit for some occupation array

    Returns:
        Amplitude
        Exponent
    """ 
    popt, pcov = curve_fit(exponential, x_data, y_data, p0, maxfev=maxfev)
    return popt[0], popt[1]

def cylinder(
    r: float,
    h: float,
    a: float = 0.0,
    nt: int = 100,
    nv: int = 50,
):
    theta = np.linspace(0, 2 * np.pi, nt)
    v = np.linspace(a, a + h, nv)
    theta, v = np.meshgrid(theta, v)

    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = v

    return x, y, z

###====================================###

### Plotting

###====================================###

def plot_shell_histogram(
    data:np.ndarray,
    shell_end: int,
    title: str,
    shell_start: int = 0,
    ax = None
) -> None:
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))

    # Makes integer-centered bins
    bins = np.arange(shell_start - 0.5, shell_end + 1.5, 1)

    ax.hist(data, bins=bins, edgecolor="black")
    ax.set_xlim(shell_start-0.5, shell_end+0.5)
    ax.set_xlabel("Shell Number")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

def plot_cnp_pred_shell_occupancy(
    predict_result: PredictResult,
    outpath: str | Path | None = None,
) -> tuple[plt.Figure, list[plt.Axes]]:
    """
    Plot predicted vs true shell histograms from predict_cnp output

    Uses:
        predict_result.best_path
    """
    # Take the data we want
    data = pd.read_csv(predict_result.best_path)
    pred_shells = data["predicted_shell_index"].to_numpy(dtype=np.int16)
    true_shells = data["true_shell_index"].to_numpy(dtype=np.int16)

    # Filter if truth values exist. For emulated data it doesnt so just pass
    has_truth_vals = np.any(true_shells >= 0)
    if has_truth_vals:
        mask = true_shells >= 0
        pred_shells = pred_shells[mask]
        true_shells = true_shells[mask]
    else:
        true_shells = None

    # Plot - depending if truth values exist
    if true_shells is None:
        shell_end = pred_shells.max()

        fig, ax = plt.subplots(figsize=(12,4))
        plot_shell_histogram(
            pred_shells,
            shell_end=shell_end,
            title="Predicted Shell Distribution",
            ax=ax,
        )
        axes = [ax]
    else: 
        # Grab maximum shell between the two
        shell_end = max(
            pred_shells.max(),
            true_shells.max(),
        )

        fig, (ax1, ax2) = plt.subplots(2,1,figsize=(12,8),sharex=True)
        plot_shell_histogram(
            pred_shells,
            shell_end=shell_end,
            title="Predicted Shell Distribution",
            ax=ax1,
        )
        plot_shell_histogram(
            true_shells,
            shell_end=shell_end,
            title="True Shell Distribution",
            ax=ax2,
        )
    
        # Clean up
        fig.suptitle("CNP Predicted vs True Shell Distribution", fontsize=14)
        axes = [ax1, ax2]
    
    
    fig.tight_layout()
    if outpath is not None:
        fig.savefig(outpath, dpi=200)
    return fig, axes

def plot_input_shell_occupancy(
    nshells: int, 
    inpath: Path | str | list[str | Path],
    outpath: Path | str,
    scale_power: Optional[float] = 1.0/3.0,
    Z_max: Optional[float | None] = None,
    R_max: Optional[float | None] = None,
) -> None:
    """
    Takes a file that is an input to training and plots the shell distribution

    CSV columns: sx, sy, sz, x, y, z, E0, ETPC
    """
    csv_files = resolve_input_paths(inpath)
    
    # Load all files and calculate sr/st/r/t and log filename
    df = load_df_with_rt(csv_files)

    # Calculate maximums
    if Z_max is None:
        Z_max = np.ceil(df['z'].max()/2)
    if R_max is None:
        R_max = np.ceil(df['r'].max())

    # Set z to measure from center
    df["z"] = np.abs(df['z'] - Z_max)
    
    # Find shell distribution
    shell_z, shell_r = find_shells(nshells, Z_max, R_max, scale_power=scale_power)
    df = find_shell_occupations(df, shell_z, shell_r)
    shells = df["shell"]
    
    x_data = np.arange(nshells)+1
    occupancy = df["shell"].value_counts().reindex(range(1, nshells+1, fill_value=0).to_numpy())
    amplitude, exponent = exponential_regression(x_data, occupancy)
    regression = exponential(x_data, amplitude, exponent)
    
    # Plot
    plt.bar(x_data, occupancy)
    plt.plot(x_data, regression, color='red')
    plt.grid()
    plt.xlabel("Shell Number")
    plt.ylabel("Count")
    plt.title("Shell Occupation")
    plt.annotate(f"Regresion: {round(amplitude,4)}*e^({round(exponent,4)}x)", xy=(0.05, 0.90), xycoords='axes fraction')
    plt.tight_layout()
    plt.savefig(outpath)
    plt.clf()

def plot_input_points_3d(
    inpath: str | Path | list[str | Path],
    outpath: str | Path,
    max_points: Optional[int] = 50000,
    r_max: float | None = None,
    z_min: float | None = None,
    z_max: float | None = None,
    seed: int | None = None,
) -> go.Figure:
    """
    Plots source and endpoints from one or more input CSVs

    Input:
        CSV that has same format as training input files. Requires sx, sy, sz, x, y, z

    Results:
        Plot with endpoints x,y,z and source points sx, sy, sz
    """
    paths = resolve_input_paths(inpath)
    if not paths:
        raise FileNotFoundError(f"No files matched: {inpath}")

    # Load data and sample
    df = load_df_with_rt(paths)
    if max_points is not None and len(df) > max_points:
        plot_df = df.sample(n=max_points, random_state=seed).copy()
    else:
        plot_df = df.copy()

    # Grab max and mins
    if r_max is None:
        r_max = float(df["r"].max())
    if z_min is None:
        z_min = float(df["z"].min())
    if z_max is None:
        z_max = float(df["z"].max())

    # Plot
    fig = go.Figure()
    colors = [
        "#e41a1c",
        "#ff7f00",
        "#ffd92f",
        "#4daf4a",
        "#f781bf",
        "#984ea3",
        "#a65628",
        "#dede00",
        "#fb9a99",
        "#cab2d6",
    ]

    for i, (fname, group) in enumerate(plot_df.groupby("filename")):
        color = colors[i%len(colors)]
        fig.add_trace(
            go.Scatter3d(
                x=group["x"],
                y=group["y"],
                z=group["z"],
                mode="markers",
                legendgroup=fname,
                name=f"{fname} endpoints",
                marker=dict(size=2, color=color),
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=group["sx"],
                y=group["sy"],
                z=group["sz"],
                mode="markers",
                legendgroup=fname,
                name=f"{fname} sources",
                marker=dict(size=2, color=color, symbol="diamond"),
                hoverinfo="skip",
                showlegend=True,
            )
        )
    xs, ys, zs = cylinder(
        r=r_max,
        h=z_max - z_min,
        a=z_min,
    )
    fig.add_trace(
        go.Surface(
            x=xs,
            y=ys,
            z=zs,
            colorscale=[[0, "blue"], [1, "blue"]],
            showscale=False,
            opacity=0.2,
            name="TPC",
            hoverinfo="skip",
            showlegend=True,
        )
    )
    fig.update_layout(
        title="Input source and endpoint distribution",
        scene=dict(
            xaxis_title="x",
            yaxis_title="y",
            zaxis_title="z",
            aspectmode="data",
        ),
    )

    # Save
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(outpath)
    
    return fig
    



    
    
