# This file is used for functions that are helpful for emulating data and everything to do with that

from typing import Optional
import numpy as np
import pandas as pd
from dataclasses import replace
from pathlib import Path
import os, sys

def find_repo_root(start: Path | None = None) -> Path:
    start = (start or Path.cwd()).resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "PROJECT_EXPERIMENT_GUIDE.md").exists() and (candidate / "README.md").exists():
            return candidate
    raise RuntimeError("Could not find the XLZD repo root from the current working directory.")


REPO_ROOT = find_repo_root()
os.chdir(REPO_ROOT)
CNP_MFGP_ROOT = REPO_ROOT / "cnp_mfgp"
if str(CNP_MFGP_ROOT) not in sys.path:
    sys.path.insert(0, str(CNP_MFGP_ROOT))

from cnp_clean_pipeline import CNPRuntimeConfig, PredictResult, predict_cnp
from prepare_cnp_mfgp_data import ShellConfig, write_h5_single_block, TARGET_COLUMN
from common.geometry import shell_boundaries

###====================================###

### Geometry

###====================================###

def sample_cylinder_region(
    n: int,
    r_min: float,
    r_max: float,
    z_min: float,
    z_max: float,
    seed: int | None = None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(n)
    
    r = np.sqrt(rng.uniform(r_min**2, r_max**2, size=n))
    theta = rng.uniform(0.0, 2*np.pi, size=n)
    z = rng.uniform(z_min, z_max, size=n)

    x = r*np.cos(theta)
    y = r*np.sin(theta)

    return np.column_stack([x,y,z])

###====================================###

### Emulation

###====================================###

def emulate_detector_points(
    n_points: int,
    radius: float,
    height: float,
    width: float,
    E0: float = 2447.0,
    ETPC: float = 2447.0,
    seed: int | None = None,
) -> pd.DataFrame:
    """
    Generates a number of points within a certain "skin-width" of the defined detector
    Allocates points by comparative volume of each region

    Regions:
        Cylindrical Wall (radius <= r <= radius+width)
        Bottom Cap       (-width <= z <= 0) 
        Top Cap          (height <= z <= height+width)

    Returns:
        Dataframe with columns: sx, sy, sz, E0, ETPC, x, y, z
        This df mimics the input columns of regular data, but the x/y/z/E0/ETPC data is fake
    """
    rng = np.random.default_rng(seed)
    outer_radius = radius+width

    # Volumes of regions
    side_vol = np.pi * (outer_radius**2 - radius**2) * height
    cap_vol = np.pi * outer_radius**2 * width
    total_volume = side_vol + 2*cap_vol

    # Allocate points wrt volume
    n_side = int(round(n_points * side_vol / total_volume))
    n_top = int(round(n_points * cap_vol / total_volume))
    n_bot = n_points - n_side - n_top

    # Find points
    side_points = sample_cylinder_region(n_side, r_min=radius, r_max=outer_radius, z_min=0.0, z_max=height)
    top_points = sample_cylinder_region(n_top, r_min=0.0, r_max=outer_radius, z_min=height, z_max=height+width)
    bot_points = sample_cylinder_region(n_bot, r_min=0.0, r_max=outer_radius, z_min=-width, z_max=0.0)

    points = np.vstack([side_points, bot_points, top_points])
    rng.shuffle(points)

    df = pd.DataFrame({
        "sx": points[:,0],
        "sy": points[:,1],
        "sz": points[:,2],
        "E0": E0,
        "ETPC": ETPC,
        "x": 0.0,
        "y": 0.0,
        "z": 0.0,
    })

    return df
    
###====================================###

### Prediction

###====================================###

def prepare_emulated_csv_for_cnp_predict(
    emulated_csv: str | Path,
    shell_cfg: ShellConfig,
    h5_block_size: Optional[int] = 100000,
    phi_headers: list[str] | tuple[str, ...] = ("s_r", "s_z_from_center"),
) -> Path:
    """
    Convert an already-saved emulated CSV into one h5 file that predict_cnp can read

    This does not create true labels as those cant be created. It wites dummy taget values
    This is required to have the same structure as actual prediction files
    """
    emulated_csv = Path(emulated_csv)

    df = pd.read_csv(emulated_csv)
    df["s_r"] = np.sqrt(df["sx"] ** 2 + df["sy"] ** 2)
    if shell_cfg.z_center is None:
        z_center = 0.5 * (df["sz"].min() + df["sz"].max())
    else:
        z_center = float(shell_cfg.z_center)
    df["s_z_from_center"] = np.abs(df["sz"] - z_center)

    # Grab shell boundaries and make a fake shell to use as "truth"
    boundaries = shell_boundaries(shell_cfg)
    dummy_shell = boundaries.iloc[-1]

    if h5_block_size is None:
        h5_block_size = len(df)
    
    for block_index, start in enumerate(range(0, len(df), h5_block_size)):
        block_df = df.iloc[start:start+h5_block_size].copy()
        block_len = len(block_df)
        h5_path = emulated_csv.with_name(f"{emulated_csv.stem}_block{block_index:04d}.h5")

        # Get fake theta, labels and real phi
        theta = np.column_stack([
            np.full(block_len, float(dummy_shell["R_boundary"]), dtype=np.float32),
            np.full(block_len, float(dummy_shell["Z_boundary"]), dtype=np.float32),
        ])
        target = np.zeros((block_len, 1), dtype=np.float32)
        phi = block_df[list(phi_headers)].to_numpy(dtype=np.float32)
    
        meta = {
            "event_index": np.arange(start, start+block_len, dtype=np.int64),
            "shell_index": np.full(block_len, -1, dtype=np.int32),
            "pair_type": np.full(block_len, "unlabeled", dtype="S16"),
        }
        
        write_h5_single_block(
            output_path=h5_path,
            theta=theta,
            phi=phi,
            target=target,
            theta_headers=["R_shell", "Z_shell"],
            phi_headers=phi_headers,
            target_headers=[TARGET_COLUMN],
            meta=meta,
        )

    return emulated_csv.parent

def predict_cnp_from_emulated_csv(
    emulated_csv: str | Path,
    runtime: CNPRuntimeConfig,
    model_path: str | Path,
    output_dir: str | Path,
    shell_cfg: ShellConfig,
    h5_block_size: Optional[int] = 100000,
    mc_samples: Optional[int] = 30,
    chunk_size: Optional[int] = 20_000,
    device: Optional[str | None] = None,
    cleanup: Optional[bool] = True,
) -> PredictResult:
    """
    Run predict_cnp on an already-saved emulated CSV

    Steps:
        1) Convert emulated CSV -> prediction h5
        2) Point runtime.predict_dirs to that h5 folder
        3) Run predict_cnp
    """
    emulated_csv = Path(emulated_csv)
    if cleanup:
        for old_h5 in emulated_csv.parent.glob(f"{emulated_csv.stem}_block*.h5"):
            old_h5.unlink()
    
    h5_dir = prepare_emulated_csv_for_cnp_predict(
        emulated_csv=emulated_csv,
        shell_cfg=shell_cfg,
        h5_block_size=h5_block_size,
        phi_headers=runtime.phi_headers,
    )

    emulated_runtime = replace(
        runtime,
        predict_dirs=[h5_dir],
        predict_fidelities=[0.0],
        predict_iterations=[0.0],
        out_dir=Path(output_dir),
    )

    return predict_cnp(
        runtime=emulated_runtime,
        model_path=model_path,
        mc_samples=mc_samples,
        output_suffix=f"{emulated_csv.stem}_event_shell_distribution",
        chunk_size=chunk_size,
        device=device,
    )
        















    