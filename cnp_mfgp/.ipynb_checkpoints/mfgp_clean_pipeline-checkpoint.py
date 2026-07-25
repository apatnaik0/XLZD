#!/usr/bin/env python3
"""Clean two-level autoregressive MF-GP pipeline for CNP outputs.

The model uses two manifest-defined fidelities:
- fidelity=0: low-fidelity CNP prediction ``y_cnp``
- fidelity=1: high-fidelity simulation target ``y_raw``

Four target-transform experiments are supported independently:
``linear``, ``log_hf``, ``log_lf``, and ``log_both``.

The standard plotting contract is deliberately small and consistent. Every
experiment writes exactly three diagnostic figures:
1. positive-quadrant MF-GP mean and uncertainty maps,
2. one validation comparison point per unique HF theta with prediction error
   bars and a visible truth-to-prediction residual,
3. a validation parity plot with prediction uncertainty.

The pipeline refuses to fit when there are too few unique LF/HF theta points
for a meaningful two-dimensional GP instead of silently running a smoke test.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


FIDELITY_LF = 0
FIDELITY_HF = 1
SHELL_COLUMN = "shell_index"


@dataclass
class MFGPRuntimeConfig:
    config_path: Path
    version: str
    sim_type: str
    theta_headers: List[str]
    theta_min: List[float]
    theta_max: List[float]
    n_shells: int
    pca_components: int | float
    pca_epsilon: float
    distribution_mc_samples: int
    out_dir_cnp: Path
    out_dir_mfgp: Path


def _default_config_path() -> Path:
    here = Path(__file__).resolve().parent
    cwd = Path.cwd().resolve()
    candidates = [
        cwd / "src/xlzd/settings.yaml",
        cwd / "xlzd/settings.yaml",
        here / "../xlzd/settings.yaml",
    ]
    for c in candidates:
        c = c.resolve()
        if c.exists():
            return c
    return (here / "config" / "settings_shell_minibatch.yaml").resolve()


def _resolve_path(path_value: str | Path, base: Path) -> Path:
    p = Path(path_value)
    return p if p.is_absolute() else (base / p).resolve()


def load_runtime_config(config_path: str | Path) -> MFGPRuntimeConfig:
    cp = Path(config_path).resolve()
    raw = yaml.safe_load(cp.read_text())
    sim = raw.get("simulation_settings", {})
    paths = raw.get("path_settings", {})
    mfgp = raw.get("mfgp_settings", {})
    base = cp.parent

    return MFGPRuntimeConfig(
        config_path=cp,
        version=str(paths.get("version", "v_clean")),
        sim_type=str(sim.get("simulation_type")),
        theta_headers=list(sim.get("theta_headers", ["R_max", "Z_max"])),
        theta_min=[float(x) for x in sim.get("theta_min", [0.0, 0.0])],
        theta_max=[float(x) for x in sim.get("theta_max", [1.0, 1.0])],
        n_shells=int(sim.get("n_shells", 100)),
        pca_components=mfgp.get("pca_components", 0.995),
        pca_epsilon=float(mfgp.get("pca_epsilon", 1e-8)),
        distribution_mc_samples=int(mfgp.get("distribution_mc_samples", 500)),
        out_dir_cnp=_resolve_path(paths.get("path_out_cnp", "../../data/out/cnp"), base),
        out_dir_mfgp=_resolve_path(paths.get("path_out_mfgp", "../../data/out/mfgp"), base),
    )


def discover_cnp_output_csv(out_dir_cnp: Path, version: str, prefer_validation: bool = False) -> Path:
    """Find an aggregated CNP CSV compatible with the new shell structure."""
    out_dir_cnp = out_dir_cnp.resolve()
    if not out_dir_cnp.exists():
        raise FileNotFoundError(f"CNP output directory does not exist: {out_dir_cnp}")

    required = {"iteration", "fidelity", "shell_index", "y_cnp", "y_cnp_err", "y_raw"}
    candidates: list[Path] = []
    for path in out_dir_cnp.glob(f"cnp_{version}_*epochs.csv"):
        if not path.is_file():
            continue
        lowered = path.name.lower()
        if any(tag in lowered for tag in ("all_shells", "best_shell", "history")):
            continue
        try:
            columns = set(pd.read_csv(path, nrows=0).columns)
        except Exception:
            continue
        if required.issubset(columns):
            candidates.append(path)

    if not candidates:
        raise FileNotFoundError(
            f"No aggregated categorical CNP CSV found in {out_dir_cnp} for version={version}. "
            "Expected columns include iteration, fidelity, shell_index, y_cnp, y_cnp_err, and y_raw."
        )

    validation = [p for p in candidates if "validation" in p.name.lower()]
    regular = [p for p in candidates if "validation" not in p.name.lower()]
    ordered = (validation + regular) if prefer_validation else (regular + validation)
    ordered.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return ordered[0]

def _missing_csv_message(csv_path: Path, out_dir_cnp: Path, version: str) -> str:
    nearby = sorted(out_dir_cnp.glob(f"cnp_{version}_*epochs.csv")) if out_dir_cnp.exists() else []
    nearby_lines = "\n".join([f"  - {p}" for p in nearby[:20]]) if nearby else "  (none found)"
    return (
        f"CSV not found: {csv_path}\n"
        f"Searched under: {out_dir_cnp}\n"
        f"Available matching files:\n{nearby_lines}"
    )


def _aggregate_rows(df: pd.DataFrame, x_cols: Sequence[str]) -> pd.DataFrame:
    """Collapse duplicate CNP rows without collapsing shell classes.

    ``x_cols`` includes ``shell_index`` for categorical shell data.  When the
    CNP CSV already contains aggregated rows, ``n_samples`` is used as a weight
    so combining multiple CSV chunks remains statistically correct.
    """
    keys = [*x_cols, "fidelity", "iteration"]
    required = {*keys, "y_cnp", "y_cnp_err", "y_raw"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in CNP CSV: {missing}")

    work = df.copy()
    work["_weight"] = (
        pd.to_numeric(work["n_samples"], errors="coerce").fillna(1.0)
        if "n_samples" in work.columns
        else 1.0
    )
    if (work["_weight"] <= 0).any():
        raise ValueError("n_samples must be positive when present.")

    for column in [*x_cols, "fidelity", "iteration", "y_cnp", "y_cnp_err", "y_raw"]:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    if work[[*x_cols, "fidelity", "iteration", "y_cnp", "y_cnp_err", "y_raw"]].isna().any().any():
        raise ValueError("CNP CSV contains non-numeric or missing MF-GP values.")

    work["_y_cnp_weighted"] = work["y_cnp"] * work["_weight"]
    work["_y_raw_weighted"] = work["y_raw"] * work["_weight"]
    work["_y_cnp_err_sq_weighted"] = np.square(work["y_cnp_err"]) * work["_weight"]

    out = (
        work.groupby(keys, dropna=False, as_index=False)
        .agg(
            n_samples_agg=("_weight", "sum"),
            y_cnp_sum=("_y_cnp_weighted", "sum"),
            y_raw_sum=("_y_raw_weighted", "sum"),
            y_cnp_err_sq_sum=("_y_cnp_err_sq_weighted", "sum"),
        )
    )
    out["y_cnp"] = out["y_cnp_sum"] / out["n_samples_agg"]
    out["y_raw"] = out["y_raw_sum"] / out["n_samples_agg"]
    out["y_cnp_err"] = np.sqrt(out["y_cnp_err_sq_sum"] / out["n_samples_agg"])
    return out[[*keys, "y_cnp", "y_cnp_err", "y_raw", "n_samples_agg"]]


def _resolve_fidelity_pair(fidelities: Sequence[int]) -> tuple[int, int]:
    """Require the manifest-defined binary fidelity convention: 0=LF, 1=HF."""
    available = sorted({int(value) for value in fidelities})
    if available != [0, 1]:
        raise ValueError(
            "This two-level MF-GP requires both manifest fidelities 0 (LF) and 1 (HF). "
            f"Found {available}."
        )
    return 0, 1


def load_mfgp_training_data(
    csv_path: str | Path,
    x_cols: Sequence[str],
    iteration: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int, int]:
    df = pd.read_csv(csv_path)
    df = _aggregate_rows(df, x_cols)
    df = df[df["iteration"].astype(int) == int(iteration)].copy()
    if df.empty:
        raise ValueError(f"No rows found for iteration={iteration}")

    selected_lf, selected_hf = _resolve_fidelity_pair(
        df["fidelity"].astype(int).unique().tolist()
    )
    lf = df[df["fidelity"].astype(int) == selected_lf].copy()
    hf = df[df["fidelity"].astype(int) == selected_hf].copy()

    if lf.empty or hf.empty:
        raise ValueError(
            f"Need both selected fidelities at iteration={iteration}. "
            f"Found fidelity {selected_lf}: {len(lf)} rows; "
            f"fidelity {selected_hf}: {len(hf)} rows."
        )

    for column in x_cols:
        lf[column] = lf[column].astype(float)
        hf[column] = hf[column].astype(float)
    lf["y_cnp"] = lf["y_cnp"].astype(float)
    lf["y_cnp_err"] = lf["y_cnp_err"].astype(float)
    hf["y_raw"] = hf["y_raw"].astype(float)

    return df, lf, hf, selected_lf, selected_hf


class CleanAutoregressiveMFGP:
    """Two-level autoregressive MF-GP using sklearn GPs."""

    def __init__(self, random_state: int = 42, alpha_lf: float = 1e-8, alpha_hf: float = 1e-8) -> None:
        self.random_state = int(random_state)
        self.alpha_lf = float(alpha_lf)
        self.alpha_hf = float(alpha_hf)

        self.x_scaler: Optional[StandardScaler] = None
        self.y_lf_scaler: Optional[StandardScaler] = None
        self.y_d_scaler: Optional[StandardScaler] = None

        self.gp_lf: Optional[GaussianProcessRegressor] = None
        self.gp_d: Optional[GaussianProcessRegressor] = None
        self.rho: Optional[float] = None
        self.x_dim: Optional[int] = None

    def _kernel(self, input_dim: int) -> ConstantKernel:
        return (
            ConstantKernel(1.0, (1e-4, 1e4))
            * Matern(length_scale=np.ones(input_dim), length_scale_bounds=(1e-3, 1e3), nu=1.5)
            + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e1))
        )

    def fit(
        self,
        x_lf: np.ndarray,
        y_lf: np.ndarray,
        x_hf: np.ndarray,
        y_hf: np.ndarray,
        verbose: bool = False,
    ) -> "CleanAutoregressiveMFGP":
        x_lf = np.asarray(x_lf, dtype=float)
        y_lf = np.asarray(y_lf, dtype=float).reshape(-1, 1)
        x_hf = np.asarray(x_hf, dtype=float)
        y_hf = np.asarray(y_hf, dtype=float).reshape(-1, 1)

        if x_lf.ndim != 2 or x_hf.ndim != 2:
            raise ValueError("x_lf and x_hf must be 2D")
        if x_lf.shape[1] != x_hf.shape[1]:
            raise ValueError("x_lf and x_hf must have same feature dimension")
        if len(x_lf) < 1 or len(x_hf) < 1:
            raise ValueError("Need at least one LF and one HF detector geometry")

        self.x_dim = x_lf.shape[1]
        small_data = min(len(x_lf), len(x_hf)) < 3
        optimizer = None if small_data else "fmin_l_bfgs_b"
        n_restarts = 0 if small_data else 2

        self.x_scaler = StandardScaler().fit(np.vstack([x_lf, x_hf]))
        x_lf_s = self.x_scaler.transform(x_lf)
        x_hf_s = self.x_scaler.transform(x_hf)

        self.y_lf_scaler = StandardScaler().fit(y_lf)
        y_lf_s = self.y_lf_scaler.transform(y_lf).ravel()

        self.gp_lf = GaussianProcessRegressor(
            kernel=self._kernel(self.x_dim),
            alpha=self.alpha_lf,
            normalize_y=False,
            optimizer=optimizer,
            n_restarts_optimizer=n_restarts,
            random_state=self.random_state,
        )
        if verbose:
            print("[fit] Training LF GP...")
        self.gp_lf.fit(x_lf_s, y_lf_s)

        mu_lf_hf_s, _ = self.gp_lf.predict(x_hf_s, return_std=True)
        mu_lf_hf = self.y_lf_scaler.inverse_transform(mu_lf_hf_s.reshape(-1, 1)).ravel()

        y_hf_vec = y_hf.ravel()
        denom = float(np.dot(mu_lf_hf, mu_lf_hf)) + 1e-12
        self.rho = float(np.dot(mu_lf_hf, y_hf_vec) / denom)
        if verbose:
            print(f"[fit] Estimated rho={self.rho:.6f}")

        y_d = y_hf_vec - self.rho * mu_lf_hf
        self.y_d_scaler = StandardScaler().fit(y_d.reshape(-1, 1))
        y_d_s = self.y_d_scaler.transform(y_d.reshape(-1, 1)).ravel()

        self.gp_d = GaussianProcessRegressor(
            kernel=self._kernel(self.x_dim),
            alpha=self.alpha_hf,
            normalize_y=False,
            optimizer=optimizer,
            n_restarts_optimizer=n_restarts,
            random_state=self.random_state + 1000,
        )
        if verbose:
            print("[fit] Training HF discrepancy GP...")
        self.gp_d.fit(x_hf_s, y_d_s)
        if verbose:
            print("[fit] GP training complete.")

        return self

    def predict(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.gp_lf is None or self.gp_d is None or self.x_scaler is None:
            raise RuntimeError("Model not fitted")
        if self.rho is None or self.y_lf_scaler is None or self.y_d_scaler is None:
            raise RuntimeError("Model not fitted")

        x = np.asarray(x, dtype=float)
        x_s = self.x_scaler.transform(x)

        mu_lf_s, std_lf_s = self.gp_lf.predict(x_s, return_std=True)
        mu_d_s, std_d_s = self.gp_d.predict(x_s, return_std=True)

        y_lf_scale = float(self.y_lf_scaler.scale_[0])
        y_d_scale = float(self.y_d_scaler.scale_[0])

        mu_lf = self.y_lf_scaler.inverse_transform(mu_lf_s.reshape(-1, 1)).ravel()
        mu_d = self.y_d_scaler.inverse_transform(mu_d_s.reshape(-1, 1)).ravel()

        std_lf = np.maximum(std_lf_s * y_lf_scale, 1e-12)
        std_d = np.maximum(std_d_s * y_d_scale, 1e-12)

        mu_hf = self.rho * mu_lf + mu_d
        var_hf = (self.rho ** 2) * (std_lf ** 2) + (std_d ** 2)
        std_hf = np.sqrt(np.maximum(var_hf, 1e-12))

        return mu_hf, std_hf, mu_lf, std_lf


@dataclass
class GeometryDistributionData:
    """One row per detector geometry and fidelity with shell-probability lists."""

    frame: pd.DataFrame
    theta: np.ndarray
    cnp_distributions: np.ndarray
    raw_distributions: np.ndarray
    cnp_uncertainties: np.ndarray


def _normalize_distributions(values: np.ndarray, epsilon: float) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D shell-distribution matrix, got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError("Shell distributions contain non-finite values")
    if (arr < -1e-12).any():
        raise ValueError("Shell distributions contain negative probabilities")
    arr = np.maximum(arr, 0.0) + float(epsilon)
    totals = arr.sum(axis=1, keepdims=True)
    if (totals <= 0).any():
        raise ValueError("A shell distribution has zero total probability")
    return arr / totals


def _validate_complete_shell_groups(
    df: pd.DataFrame,
    theta_headers: Sequence[str],
    n_shells: int,
) -> None:
    expected = set(range(1, int(n_shells) + 1))
    group_columns = [*theta_headers, "iteration", "fidelity"]
    problems: list[str] = []
    for key, group in df.groupby(group_columns, dropna=False):
        shell_values = group[SHELL_COLUMN].astype(int)
        actual = set(shell_values.tolist())
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        duplicates = int(shell_values.duplicated().sum())
        if missing or unexpected or duplicates:
            problems.append(
                f"group={key}: missing={missing[:10]}, unexpected={unexpected[:10]}, "
                f"duplicates={duplicates}"
            )
    if problems:
        raise ValueError(
            "Each detector geometry/fidelity must contain exactly one aggregate row "
            f"for every shell 1..{n_shells}.\n" + "\n".join(problems[:10])
        )


def load_geometry_distribution_data(
    csv_path: str | Path,
    theta_headers: Sequence[str],
    n_shells: int,
    iteration: int = 0,
    *,
    require_both_fidelities: bool = True,
    epsilon: float = 1e-8,
) -> GeometryDistributionData:
    """Load the existing long-form CNP CSV and pivot it internally.

    The CNP file format is unchanged. Internally, every returned dataframe row
    represents one detector geometry and one manifest-defined fidelity. The
    list-valued columns contain the ordered shell values for shells 1..N.
    """

    long_df = _aggregate_rows(pd.read_csv(csv_path), [*theta_headers, SHELL_COLUMN])
    long_df = long_df[long_df["iteration"].astype(int) == int(iteration)].copy()
    if long_df.empty:
        raise ValueError(f"No rows found for iteration={iteration}")

    fidelities = sorted(long_df["fidelity"].astype(int).unique().tolist())
    allowed = {FIDELITY_LF, FIDELITY_HF}
    if not set(fidelities).issubset(allowed):
        raise ValueError(
            f"fidelity must be 0 (LF) or 1 (HF); found {fidelities}"
        )
    if require_both_fidelities and fidelities != [FIDELITY_LF, FIDELITY_HF]:
        raise ValueError(
            "Training CNP CSV must contain both fidelity=0 and fidelity=1; "
            f"found {fidelities}"
        )

    long_df[SHELL_COLUMN] = long_df[SHELL_COLUMN].astype(int)
    _validate_complete_shell_groups(long_df, theta_headers, n_shells)

    rows: list[dict[str, object]] = []
    group_columns = [*theta_headers, "iteration", "fidelity"]
    for key, group in long_df.groupby(group_columns, dropna=False, sort=True):
        group = group.sort_values(SHELL_COLUMN)
        if not isinstance(key, tuple):
            key = (key,)
        row: dict[str, object] = dict(zip(group_columns, key))
        row["shell_indices"] = group[SHELL_COLUMN].astype(int).tolist()
        row["cnp_shell_probabilities"] = group["y_cnp"].astype(float).tolist()
        row["raw_shell_probabilities"] = group["y_raw"].astype(float).tolist()
        row["cnp_shell_uncertainties"] = group["y_cnp_err"].astype(float).tolist()
        row["n_samples_by_shell"] = group["n_samples_agg"].astype(float).tolist()
        rows.append(row)

    frame = pd.DataFrame(rows).sort_values(group_columns).reset_index(drop=True)
    theta = frame[list(theta_headers)].to_numpy(dtype=float)
    cnp = _normalize_distributions(
        np.vstack(frame["cnp_shell_probabilities"].map(np.asarray)), epsilon
    )
    raw = _normalize_distributions(
        np.vstack(frame["raw_shell_probabilities"].map(np.asarray)), epsilon
    )
    cnp_uncertainty = np.vstack(
        frame["cnp_shell_uncertainties"].map(np.asarray)
    ).astype(float)
    return GeometryDistributionData(
        frame=frame,
        theta=theta,
        cnp_distributions=cnp,
        raw_distributions=raw,
        cnp_uncertainties=cnp_uncertainty,
    )


class CLRPrincipalComponents:
    """Compress complete shell distributions in centered-log-ratio space."""

    def __init__(self, n_components: int | float = 0.995, epsilon: float = 1e-8) -> None:
        self.n_components = n_components
        self.epsilon = float(epsilon)
        self.pca: Optional[PCA] = None

    def _clr(self, distributions: np.ndarray) -> np.ndarray:
        simplex = _normalize_distributions(distributions, self.epsilon)
        logs = np.log(simplex)
        return logs - logs.mean(axis=1, keepdims=True)

    def fit(self, distributions: np.ndarray) -> "CLRPrincipalComponents":
        clr = self._clr(distributions)
        max_components = min(clr.shape[0], clr.shape[1])
        requested = self.n_components
        if isinstance(requested, int):
            requested = max(1, min(int(requested), max_components))
        else:
            requested = float(requested)
            if not 0.0 < requested <= 1.0:
                raise ValueError("pca_components must be an integer or a fraction in (0, 1]")
        self.pca = PCA(n_components=requested, svd_solver="full")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            self.pca.fit(clr)
        return self

    def transform(self, distributions: np.ndarray) -> np.ndarray:
        if self.pca is None:
            raise RuntimeError("PCA compressor has not been fitted")
        return self.pca.transform(self._clr(distributions))

    def inverse_transform(self, coefficients: np.ndarray) -> np.ndarray:
        if self.pca is None:
            raise RuntimeError("PCA compressor has not been fitted")
        clr = self.pca.inverse_transform(np.asarray(coefficients, dtype=float))
        clr = clr - clr.max(axis=1, keepdims=True)
        values = np.exp(clr)
        return values / values.sum(axis=1, keepdims=True)


class DistributionAutoregressiveMFGP:
    """One autoregressive MF-GP per latent distribution coefficient."""

    def __init__(
        self,
        pca_components: int | float = 0.995,
        pca_epsilon: float = 1e-8,
        random_state: int = 42,
        alpha_lf: float = 1e-8,
        alpha_hf: float = 1e-8,
    ) -> None:
        self.compressor = CLRPrincipalComponents(pca_components, pca_epsilon)
        self.random_state = int(random_state)
        self.alpha_lf = float(alpha_lf)
        self.alpha_hf = float(alpha_hf)
        self.models: list[CleanAutoregressiveMFGP] = []

    def fit(
        self,
        x_lf: np.ndarray,
        distributions_lf: np.ndarray,
        x_hf: np.ndarray,
        distributions_hf: np.ndarray,
        *,
        verbose: bool = False,
    ) -> "DistributionAutoregressiveMFGP":
        self.compressor.fit(np.vstack([distributions_lf, distributions_hf]))
        lf_coefficients = self.compressor.transform(distributions_lf)
        hf_coefficients = self.compressor.transform(distributions_hf)
        self.models = []
        for component in range(lf_coefficients.shape[1]):
            if verbose:
                print(
                    f"[fit] Latent component {component + 1}/"
                    f"{lf_coefficients.shape[1]}"
                )
            scalar_model = CleanAutoregressiveMFGP(
                random_state=self.random_state + component,
                alpha_lf=self.alpha_lf,
                alpha_hf=self.alpha_hf,
            )
            scalar_model.fit(
                x_lf=x_lf,
                y_lf=lf_coefficients[:, component],
                x_hf=x_hf,
                y_hf=hf_coefficients[:, component],
                verbose=False,
            )
            self.models.append(scalar_model)
        return self

    def predict_latent(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not self.models:
            raise RuntimeError("Distribution MF-GP has not been fitted")
        means: list[np.ndarray] = []
        stds: list[np.ndarray] = []
        for scalar_model in self.models:
            mean, std, _, _ = scalar_model.predict(x)
            means.append(mean)
            stds.append(std)
        return np.column_stack(means), np.column_stack(stds)

    def predict_distribution(
        self,
        x: np.ndarray,
        *,
        mc_samples: int = 500,
        random_state: Optional[int] = None,
    ) -> dict[str, np.ndarray]:
        latent_mean, latent_std = self.predict_latent(x)
        point = self.compressor.inverse_transform(latent_mean)
        if mc_samples <= 0:
            return {
                "mean": point,
                "std": np.zeros_like(point),
                "q025": point,
                "q975": point,
                "latent_mean": latent_mean,
                "latent_std": latent_std,
            }
        rng = np.random.default_rng(
            self.random_state if random_state is None else int(random_state)
        )
        draws = []
        for _ in range(int(mc_samples)):
            sample = rng.normal(latent_mean, np.maximum(latent_std, 1e-12))
            draws.append(self.compressor.inverse_transform(sample))
        stack = np.stack(draws, axis=0)
        return {
            "mean": stack.mean(axis=0),
            "std": stack.std(axis=0, ddof=0),
            "q025": np.quantile(stack, 0.025, axis=0),
            "q975": np.quantile(stack, 0.975, axis=0),
            "latent_mean": latent_mean,
            "latent_std": latent_std,
        }


def _distribution_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    truth = _normalize_distributions(truth, 1e-12)
    prediction = _normalize_distributions(prediction, 1e-12)
    midpoint = 0.5 * (truth + prediction)
    js = 0.5 * (
        np.sum(truth * np.log(truth / midpoint), axis=1)
        + np.sum(prediction * np.log(prediction / midpoint), axis=1)
    )
    tv = 0.5 * np.sum(np.abs(truth - prediction), axis=1)
    rmse = np.sqrt(np.mean(np.square(truth - prediction), axis=1))
    mae = np.mean(np.abs(truth - prediction), axis=1)
    return {
        "mean_shell_rmse": float(rmse.mean()),
        "mean_shell_mae": float(mae.mean()),
        "mean_total_variation": float(tv.mean()),
        "mean_jensen_shannon": float(js.mean()),
        "max_sum_error": float(np.max(np.abs(prediction.sum(axis=1) - 1.0))),
    }


def _json_list(values: np.ndarray | Sequence[float]) -> str:
    return json.dumps(np.asarray(values, dtype=float).tolist())


def _prediction_rows(
    geometry_frame: pd.DataFrame,
    prediction: dict[str, np.ndarray],
    *,
    iteration: int,
    fidelity: int,
    truth: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index, geometry in geometry_frame.reset_index(drop=True).iterrows():
        row = geometry.to_dict()
        row["iteration"] = int(iteration)
        row["fidelity"] = int(fidelity)
        row["predicted_shell_probabilities"] = _json_list(prediction["mean"][index])
        row["predicted_shell_std"] = _json_list(prediction["std"][index])
        row["predicted_shell_q025"] = _json_list(prediction["q025"][index])
        row["predicted_shell_q975"] = _json_list(prediction["q975"][index])
        if truth is not None:
            row["true_shell_probabilities"] = _json_list(truth[index])
        rows.append(row)
    return pd.DataFrame(rows)


def _geometry_grid_frame(
    runtime: MFGPRuntimeConfig,
    lf_theta: np.ndarray,
    hf_theta: np.ndarray,
    points_per_axis: int,
) -> pd.DataFrame:
    combined = np.vstack([lf_theta, hf_theta])
    if len(runtime.theta_headers) != 2 or points_per_axis <= 0:
        return pd.DataFrame(combined, columns=runtime.theta_headers).drop_duplicates()
    if len(runtime.theta_min) == 2 and len(runtime.theta_max) == 2:
        lower = np.asarray(runtime.theta_min, dtype=float)
        upper = np.asarray(runtime.theta_max, dtype=float)
    else:
        lower = combined.min(axis=0)
        upper = combined.max(axis=0)
    if np.allclose(lower, upper):
        return pd.DataFrame([lower], columns=runtime.theta_headers)
    axis0 = np.linspace(lower[0], upper[0], int(points_per_axis))
    axis1 = np.linspace(lower[1], upper[1], int(points_per_axis))
    mesh0, mesh1 = np.meshgrid(axis0, axis1, indexing="xy")
    return pd.DataFrame(
        np.column_stack([mesh0.ravel(), mesh1.ravel()]),
        columns=runtime.theta_headers,
    )


def _plot_distribution_observations(
    geometry_frame: pd.DataFrame,
    truth: np.ndarray,
    out_path: Path,
    title: str,
    max_geometries: int = 8,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    shells = np.arange(1, truth.shape[1] + 1)
    for index in range(min(len(geometry_frame), max_geometries)):
        label = ", ".join(
            f"{column}={geometry_frame.iloc[index][column]:g}"
            for column in geometry_frame.columns
        )
        ax.plot(shells, truth[index], label=label)
    ax.set_xlabel("Shell index")
    ax.set_ylabel("Probability")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    if len(geometry_frame):
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def _plot_distribution_predictions(
    geometry_frame: pd.DataFrame,
    prediction: dict[str, np.ndarray],
    out_path: Path,
    title: str,
    truth: Optional[np.ndarray] = None,
    max_geometries: int = 6,
) -> None:
    count = min(len(geometry_frame), max_geometries)
    fig, axes = plt.subplots(count, 1, figsize=(11, max(4, 3.3 * count)), squeeze=False)
    shells = np.arange(1, prediction["mean"].shape[1] + 1)
    for index in range(count):
        ax = axes[index, 0]
        mean = prediction["mean"][index]
        ax.plot(shells, mean, label="MF-GP mean")
        ax.fill_between(
            shells,
            prediction["q025"][index],
            prediction["q975"][index],
            alpha=0.25,
            label="95% interval",
        )
        if truth is not None:
            ax.plot(shells, truth[index], linestyle="--", label="HF truth")
        label = ", ".join(
            f"{column}={geometry_frame.iloc[index][column]:g}"
            for column in geometry_frame.columns
        )
        ax.set_title(label)
        ax.set_xlabel("Shell index")
        ax.set_ylabel("Probability")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


@dataclass
class MFGPResult:
    cnp_csv: Path
    model_json: Path
    metrics_json: Path
    prediction_csv: Path
    grid_csv: Path
    data_plot: Optional[Path]
    parity_plot: Optional[Path]
    mean_std_plot: Path
    mean_std_plot_3d_html: Optional[Path]
    residual_plot: Optional[Path]
    theta_group_plot_dir: Optional[Path]
    across_theta_plot: Optional[Path]
    across_theta_zoom_plot: Optional[Path]
    coverage_plot: Optional[Path]
    validation_parity_plot: Optional[Path]
    train_across_theta_linear_plot: Optional[Path]
    train_across_theta_log_linear_sigma_plot: Optional[Path]
    train_across_theta_log_log_sigma_plot: Optional[Path]
    validation_across_theta_linear_plot: Optional[Path]
    validation_across_theta_log_linear_sigma_plot: Optional[Path]
    validation_across_theta_log_log_sigma_plot: Optional[Path]
    train_parity_linear_plot: Optional[Path]
    train_parity_log_plot: Optional[Path]
    validation_parity_linear_plot: Optional[Path]
    validation_parity_log_plot: Optional[Path]


def _grid_from_bounds(theta_min: Sequence[float], theta_max: Sequence[float], n: int = 120) -> np.ndarray:
    x = np.linspace(float(theta_min[0]), float(theta_max[0]), n)
    y = np.linspace(float(theta_min[1]), float(theta_max[1]), n)
    gx, gy = np.meshgrid(x, y, indexing="xy")
    return np.column_stack([gx.ravel(), gy.ravel()])



def _shell_grid_from_data(
    lf: pd.DataFrame,
    hf: pd.DataFrame,
    theta_headers: Sequence[str],
    shell_column: str = SHELL_COLUMN,
) -> pd.DataFrame:
    """Build every observed detector geometry crossed with every shell class."""
    geometry = (
        pd.concat([lf[list(theta_headers)], hf[list(theta_headers)]], ignore_index=True)
        .drop_duplicates()
        .sort_values(list(theta_headers), kind="mergesort")
        .reset_index(drop=True)
    )
    shells = np.sort(
        pd.concat([lf[shell_column], hf[shell_column]], ignore_index=True)
        .astype(int)
        .unique()
    )
    if geometry.empty or len(shells) == 0:
        raise ValueError("Cannot build shell prediction grid from empty geometry or shell values.")

    geometry = geometry.copy()
    geometry["_cross"] = 1
    shell_df = pd.DataFrame({shell_column: shells, "_cross": 1})
    return geometry.merge(shell_df, on="_cross", how="inner").drop(columns="_cross")


def _geometry_label(row: pd.Series, theta_headers: Sequence[str]) -> str:
    return ", ".join(f"{name}={float(row[name]):.5g}" for name in theta_headers)


def _plot_shell_observations(
    hf: pd.DataFrame,
    theta_headers: Sequence[str],
    out_path: Path,
    shell_column: str = SHELL_COLUMN,
    max_geometries: int = 12,
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    groups = list(hf.groupby(list(theta_headers), dropna=False, sort=True))
    for _, group in groups[:max_geometries]:
        group = group.sort_values(shell_column)
        ax.plot(
            group[shell_column],
            group["y_raw"],
            marker="o",
            ms=2.5,
            lw=1.1,
            alpha=0.8,
            label=_geometry_label(group.iloc[0], theta_headers),
        )
    ax.set_xlabel("Shell index")
    ax.set_ylabel("High-fidelity shell occupation")
    ax.set_title("High-fidelity shell distributions")
    ax.grid(True, alpha=0.25)
    if groups:
        ax.legend(loc="best", fontsize=7, frameon=True)
    if len(groups) > max_geometries:
        ax.text(
            0.01,
            0.01,
            f"Showing {max_geometries}/{len(groups)} detector geometries",
            transform=ax.transAxes,
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def _plot_shell_mean_std(
    grid_df: pd.DataFrame,
    theta_headers: Sequence[str],
    out_path: Path,
    shell_column: str = SHELL_COLUMN,
    max_geometries: int = 12,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    groups = list(grid_df.groupby(list(theta_headers), dropna=False, sort=True))
    for _, group in groups[:max_geometries]:
        group = group.sort_values(shell_column)
        label = _geometry_label(group.iloc[0], theta_headers)
        axes[0].plot(group[shell_column], group["mf_mean"], lw=1.2, label=label)
        axes[1].plot(group[shell_column], group["mf_std"], lw=1.2, label=label)

    axes[0].set_title("MF-GP high-fidelity mean by shell")
    axes[0].set_ylabel("Predicted shell occupation")
    axes[1].set_title("MF-GP high-fidelity uncertainty by shell")
    axes[1].set_ylabel("Prediction standard deviation")
    for axis in axes:
        axis.set_xlabel("Shell index")
        axis.grid(True, alpha=0.25)
        if groups:
            axis.legend(loc="best", fontsize=7, frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def _plot_shell_validation_comparison(
    validation_df: pd.DataFrame,
    theta_headers: Sequence[str],
    out_path: Path,
    shell_column: str = SHELL_COLUMN,
    max_geometries: int = 8,
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(11, 6))
    groups = list(validation_df.groupby(list(theta_headers), dropna=False, sort=True))
    for index, (_, group) in enumerate(groups[:max_geometries]):
        group = group.sort_values(shell_column)
        label = _geometry_label(group.iloc[0], theta_headers)
        ax.plot(
            group[shell_column],
            group["y_raw"],
            marker="o",
            ms=2.5,
            lw=1.0,
            alpha=0.75,
            label=f"truth: {label}",
        )
        ax.plot(
            group[shell_column],
            group["mf_mean"],
            ls="--",
            lw=1.2,
            alpha=0.9,
            label=f"MF-GP: {label}",
        )
    ax.set_xlabel("Shell index")
    ax.set_ylabel("High-fidelity shell occupation")
    ax.set_title("Validation shell distributions: truth vs MF-GP")
    ax.grid(True, alpha=0.25)
    if groups:
        ax.legend(loc="best", fontsize=6.5, ncol=2, frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def _scatter_grid(df: pd.DataFrame, x_col: str, y_col: str, z_col: str, n: int = 120) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = df[x_col].to_numpy(dtype=float)
    y = df[y_col].to_numpy(dtype=float)
    z = df[z_col].to_numpy(dtype=float)
    gx = np.linspace(x.min(), x.max(), n)
    gy = np.linspace(y.min(), y.max(), n)
    X, Y = np.meshgrid(gx, gy, indexing="xy")
    try:
        from scipy.interpolate import griddata

        Z = griddata((x, y), z, (X, Y), method="cubic")
        if Z is None or np.isnan(Z).all():
            Z = griddata((x, y), z, (X, Y), method="linear")
    except Exception:
        Z = None

    if Z is None:
        Z = np.full_like(X, np.nan, dtype=float)
    return X, Y, Z


def _plot_hf_observation_map(hf: pd.DataFrame, x_cols: Sequence[str], out_path: Path) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    s = ax.scatter(hf[x_cols[0]], hf[x_cols[1]], c=hf["y_raw"], cmap="viridis", s=40, edgecolor="none")
    ax.set_title("HF observations (y_raw)")
    ax.set_xlabel(x_cols[0])
    ax.set_ylabel(x_cols[1])
    plt.colorbar(s, ax=ax, label="y_raw")
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def _choose_log_plot_eps(*arrays: np.ndarray) -> float:
    positives: List[np.ndarray] = []
    for arr in arrays:
        a = np.asarray(arr, dtype=float)
        if a.size == 0:
            continue
        pos = a[np.isfinite(a) & (a > 0)]
        if pos.size:
            positives.append(pos)
    if positives:
        min_pos = float(min(np.min(pos) for pos in positives))
        return max(min_pos * 0.5, 1e-12)
    return 1e-12


def _safe_log10(values: np.ndarray, eps: float) -> np.ndarray:
    return np.log10(np.maximum(np.asarray(values, dtype=float), float(eps)))


def _log_sigma_from_linear(mu: np.ndarray, sigma: np.ndarray, eps: float) -> np.ndarray:
    mu_safe = np.maximum(np.asarray(mu, dtype=float), float(eps))
    sigma_arr = np.maximum(np.asarray(sigma, dtype=float), 1e-12)
    return sigma_arr / (mu_safe * np.log(10.0))


def _target_transform_suffix(target_transform: str) -> str:
    mode = str(target_transform).strip().lower()
    if mode not in {"linear", "log_hf", "log_lf", "log_both"}:
        raise ValueError(
            f"Unsupported target_transform={target_transform!r}. "
            "Expected one of: linear, log_hf, log_lf, log_both."
        )
    return mode


def _transform_mfgp_targets(
    y_lf: np.ndarray,
    y_hf: np.ndarray,
    target_transform: str,
    eps: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, float, bool, bool]:
    mode = _target_transform_suffix(target_transform)
    y_lf = np.asarray(y_lf, dtype=float)
    y_hf = np.asarray(y_hf, dtype=float)
    use_log_lf = mode in {"log_lf", "log_both"}
    use_log_hf = mode in {"log_hf", "log_both"}
    eps_val = float(eps) if eps is not None else _choose_log_plot_eps(y_lf, y_hf)

    out_lf = _safe_log10(y_lf, eps_val) if use_log_lf else y_lf.copy()
    out_hf = _safe_log10(y_hf, eps_val) if use_log_hf else y_hf.copy()
    return out_lf, out_hf, eps_val, use_log_lf, use_log_hf


def _transform_series_for_mode(values: np.ndarray, use_log: bool, eps: float) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return _safe_log10(arr, eps) if use_log else arr


def _plot_parity(y_true: np.ndarray, y_pred: np.ndarray, y_std: np.ndarray, out_path: Path) -> None:
    eps = _choose_log_plot_eps(y_true, y_pred, y_pred - y_std, y_pred + y_std)
    y_true_log = _safe_log10(y_true, eps)
    y_pred_log = _safe_log10(y_pred, eps)
    y_lower_log = _safe_log10(y_pred - y_std, eps)
    y_upper_log = _safe_log10(y_pred + y_std, eps)
    yerr_log = np.vstack(
        [
            np.maximum(y_pred_log - y_lower_log, 0.0),
            np.maximum(y_upper_log - y_pred_log, 0.0),
        ]
    )

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.errorbar(y_true_log, y_pred_log, yerr=yerr_log, fmt="o", ms=4, alpha=0.6, capsize=2, color="tab:blue")
    lo = float(min(np.min(y_true_log), np.min(y_pred_log)))
    hi = float(max(np.max(y_true_log), np.max(y_pred_log)))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlabel(f"log10(HF truth y_raw, floor={eps:.2e})")
    ax.set_ylabel("log10(MF-GP prediction)")
    ax.set_title("HF Parity Plot (Log Scale)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def _plot_parity_linear(y_true: np.ndarray, y_pred: np.ndarray, y_std: np.ndarray, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.errorbar(y_true, y_pred, yerr=y_std, fmt="o", ms=4, alpha=0.6, capsize=2, color="tab:blue")
    lo = float(min(np.min(y_true), np.min(y_pred)))
    hi = float(max(np.max(y_true), np.max(y_pred)))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlabel("HF truth (y_raw)")
    ax.set_ylabel("MF-GP prediction")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def _plot_parity_log(y_true: np.ndarray, y_pred: np.ndarray, y_std: np.ndarray, out_path: Path, title: str) -> None:
    eps = _choose_log_plot_eps(y_true, y_pred, y_pred - y_std, y_pred + y_std)
    y_true_log = _safe_log10(y_true, eps)
    y_pred_log = _safe_log10(y_pred, eps)
    y_lower_log = _safe_log10(y_pred - y_std, eps)
    y_upper_log = _safe_log10(y_pred + y_std, eps)
    yerr_log = np.vstack(
        [
            np.maximum(y_pred_log - y_lower_log, 0.0),
            np.maximum(y_upper_log - y_pred_log, 0.0),
        ]
    )

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.errorbar(y_true_log, y_pred_log, yerr=yerr_log, fmt="o", ms=4, alpha=0.6, capsize=2, color="tab:blue")
    lo = float(min(np.min(y_true_log), np.min(y_pred_log)))
    hi = float(max(np.max(y_true_log), np.max(y_pred_log)))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlabel(f"log10(HF truth y_raw, floor={eps:.2e})")
    ax.set_ylabel("log10(MF-GP prediction)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)

def _plot_mean_std_linear(
    grid_xy: np.ndarray,
    mean_hf: np.ndarray,
    std_hf: np.ndarray,
    x_cols: Sequence[str],
    hf_points: Optional[np.ndarray],
    out_path: Path) -> None:

    # Plots the mean and std in a line plot instead of 2D
    # Useful for shell distributions for ease of recognition
    gx = np.unique(grid_xy[:, 0])
    gy = np.unique(grid_xy[:, 1])
    nx, ny = len(gx), len(gy)
    Zm = mean_hf.reshape(ny, nx)
    Zs = std_hf.reshape(ny, nx)

    # Take only the diagonal points
    mean_points = Zm.diagonal()
    std_points = Zs.diagonal()

    # Scale the plotting on x axis to proportion of detector
    max_x, max_y = max(gx), max(gy)
    scaled_axis = gx/max_x
    scaled_hf = hf_points / np.array([max_x, max_y])

    fig, ax = plt.subplots(1, 2, figsize=(18, 8))
    # Mean Plot
    im1 = ax[0].plot(scaled_axis, mean_points)
    ax[0].set_title("MF-GP Mean (HF)")
    ax[0].set_ylabel("Prediction Mean")
    
    # STD Plot
    im2 = ax[1].plot(scaled_axis, std_points)
    ax[1].set_title("MF-GP STD (HF)")
    ax[1].set_ylabel("Prediction STD")

    for a in ax:
        a.set_xlabel(r"Normalized detector position, $R/R_{\max}=Z/Z_{\max}$")
        a.grid(alpha=0.25)
    
        if hf_points is not None and len(hf_points):
            scaled_hf = hf_points[:,0] / max_x
    
            ymin, ymax = a.get_ylim()
            rug_height = 0.05 * (ymax - ymin)
    
            a.vlines(
                scaled_hf,
                ymin=ymin,
                ymax=ymin + rug_height,
                color="k",
                linewidth=1.2,
                alpha=0.75,
                label="HF validation points"
            )
    
        a.legend()
    
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)

def _plot_mean_std_heatmaps(
    grid_xy: np.ndarray,
    mean_hf: np.ndarray,
    std_hf: np.ndarray,
    x_cols: Sequence[str],
    hf_points: Optional[np.ndarray],
    out_path: Path,
    *,
    title: str = "MF-GP high-fidelity prediction",
) -> None:
    """Plot only the physical R>=0, Z>=0 quadrant.

    ``std_hf`` is the one-sigma half-width in the displayed target space. Each
    overlaid marker is one unique HF theta point; no mirrored copies are made.
    """
    grid_xy = np.asarray(grid_xy, dtype=float)
    mean_hf = np.asarray(mean_hf, dtype=float).reshape(-1)
    std_hf = np.asarray(std_hf, dtype=float).reshape(-1)
    if grid_xy.ndim != 2 or grid_xy.shape[1] != 2:
        raise ValueError(f"Expected a (n, 2) prediction grid, got {grid_xy.shape}")
    if len(grid_xy) != len(mean_hf) or len(grid_xy) != len(std_hf):
        raise ValueError("Grid, mean, and uncertainty arrays must have equal length")

    positive = (grid_xy[:, 0] >= 0.0) & (grid_xy[:, 1] >= 0.0)
    grid_xy = grid_xy[positive]
    mean_hf = mean_hf[positive]
    std_hf = std_hf[positive]
    if not len(grid_xy):
        raise ValueError("No R>=0, Z>=0 grid points are available for plotting")

    gx = np.unique(grid_xy[:, 0])
    gy = np.unique(grid_xy[:, 1])
    nx, ny = len(gx), len(gy)
    if nx * ny != len(grid_xy):
        raise ValueError("Prediction grid is not a complete rectangular R-Z mesh")

    order = np.lexsort((grid_xy[:, 0], grid_xy[:, 1]))
    Zm = mean_hf[order].reshape(ny, nx)
    Zs = std_hf[order].reshape(ny, nx)

    points = None
    if hf_points is not None and len(hf_points):
        points = np.asarray(hf_points, dtype=float)
        points = points[(points[:, 0] >= 0.0) & (points[:, 1] >= 0.0)]
        if len(points):
            points = np.unique(points, axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.8), constrained_layout=True)

    mean_levels = 24 if np.nanmax(Zm) > np.nanmin(Zm) else 2
    std_levels = 24 if np.nanmax(Zs) > np.nanmin(Zs) else 2
    im_mean = axes[0].contourf(gx, gy, Zm, levels=mean_levels, cmap="viridis")
    im_std = axes[1].contourf(gx, gy, Zs, levels=std_levels, cmap="Reds")

    for axis in axes:
        if points is not None and len(points):
            axis.scatter(
                points[:, 0],
                points[:, 1],
                s=38,
                facecolor="white",
                edgecolor="black",
                linewidth=0.8,
                alpha=0.95,
                label=f"HF points (n={len(points)})",
                zorder=4,
            )
            axis.legend(loc="best", fontsize=8, frameon=True)
        axis.set_xlabel(x_cols[0])
        axis.set_ylabel(x_cols[1])
        axis.set_xlim(left=max(0.0, float(gx.min())))
        axis.set_ylim(bottom=max(0.0, float(gy.min())))
        axis.grid(alpha=0.16)

    axes[0].set_title("Predicted HF mean")
    axes[1].set_title("Predicted HF 1σ uncertainty")
    fig.colorbar(im_mean, ax=axes[0], label="prediction")
    fig.colorbar(im_std, ax=axes[1], label="1σ half-width")
    fig.suptitle(title, fontsize=14)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

def _write_interactive_mean_std_surfaces_3d_html(
    grid_xy: np.ndarray,
    mean_hf: np.ndarray,
    std_hf: np.ndarray,
    x_cols: Sequence[str],
    hf_points: Optional[np.ndarray],
    hf_mean_values: Optional[np.ndarray],
    hf_std_values: Optional[np.ndarray],
    out_path: Path,
) -> bool:
    """Optional positive-quadrant interactive surface retained for compatibility."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception:
        return False

    grid_xy = np.asarray(grid_xy, dtype=float)
    positive = (grid_xy[:, 0] >= 0.0) & (grid_xy[:, 1] >= 0.0)
    grid_xy = grid_xy[positive]
    mean_hf = np.asarray(mean_hf, dtype=float).reshape(-1)[positive]
    std_hf = np.asarray(std_hf, dtype=float).reshape(-1)[positive]

    gx = np.unique(grid_xy[:, 0])
    gy = np.unique(grid_xy[:, 1])
    nx, ny = len(gx), len(gy)
    order = np.lexsort((grid_xy[:, 0], grid_xy[:, 1]))
    Zm = mean_hf[order].reshape(ny, nx)
    Zs = std_hf[order].reshape(ny, nx)
    X, Y = np.meshgrid(gx, gy, indexing="xy")

    points = None
    if hf_points is not None and len(hf_points):
        points = np.asarray(hf_points, dtype=float)
        points = points[(points[:, 0] >= 0.0) & (points[:, 1] >= 0.0)]
        if len(points):
            points = np.unique(points, axis=0)

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=("MF-GP mean", "MF-GP 1σ uncertainty"),
        horizontal_spacing=0.04,
    )
    fig.add_trace(
        go.Surface(x=X, y=Y, z=Zm, colorscale="Viridis", opacity=0.88,
                   name="Mean surface", showscale=True,
                   colorbar={"title": "prediction", "x": 0.46}),
        row=1, col=1,
    )
    fig.add_trace(
        go.Surface(x=X, y=Y, z=Zs, colorscale="Reds", opacity=0.88,
                   name="Uncertainty surface", showscale=True,
                   colorbar={"title": "1σ", "x": 1.02}),
        row=1, col=2,
    )

    if points is not None and len(points):
        z_mean = np.zeros(len(points)) if hf_mean_values is None else np.asarray(hf_mean_values, dtype=float)[:len(points)]
        z_std = np.zeros(len(points)) if hf_std_values is None else np.asarray(hf_std_values, dtype=float)[:len(points)]
        style = dict(mode="markers", marker={"size": 4, "color": "white", "line": {"color": "black", "width": 1}}, showlegend=False)
        fig.add_trace(go.Scatter3d(x=points[:, 0], y=points[:, 1], z=z_mean, **style), row=1, col=1)
        fig.add_trace(go.Scatter3d(x=points[:, 0], y=points[:, 1], z=z_std, **style), row=1, col=2)

    scene_common = dict(xaxis_title=x_cols[0], yaxis_title=x_cols[1], aspectmode="cube")
    fig.update_layout(
        template="plotly_white",
        height=850,
        width=1600,
        margin=dict(l=10, r=10, t=60, b=10),
        scene={**scene_common, "zaxis_title": "prediction"},
        scene2={**scene_common, "zaxis_title": "1σ half-width"},
        title="MF-GP Mean / Uncertainty Surfaces (R≥0, Z≥0)",
    )
    fig.write_html(out_path, include_plotlyjs="cdn")
    return True

def _plot_residual_heatmap(
    df_hf_pred: pd.DataFrame,
    x_cols: Sequence[str],
    out_path: Path,
) -> None:
    tmp = df_hf_pred.copy()
    tmp["residual"] = tmp["mf_mean"] - tmp["y_raw"]
    X, Y, Z = _scatter_grid(tmp, x_cols[0], x_cols[1], "residual", n=120)

    vmax = float(np.nanmax(np.abs(tmp["residual"].to_numpy()))) if len(tmp) else 1.0
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    if np.isfinite(Z).any():
        im = ax.contourf(X, Y, Z, levels=24, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        plt.colorbar(im, ax=ax, label="mf_mean - y_raw")
    sc = ax.scatter(
        tmp[x_cols[0]],
        tmp[x_cols[1]],
        c=tmp["residual"],
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        s=25,
        edgecolor="white",
        linewidth=0.4,
    )
    ax.set_title("HF Residual Map")
    ax.set_xlabel(x_cols[0])
    ax.set_ylabel(x_cols[1])
    if not np.isfinite(Z).any():
        plt.colorbar(sc, ax=ax, label="mf_mean - y_raw")

    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def _resolve_optional_validation_csv(runtime: MFGPRuntimeConfig) -> Optional[Path]:
    try:
        raw = yaml.safe_load(runtime.config_path.read_text())
        p = raw.get("path_settings", {}).get("path_to_files_validation")
        if not p:
            return None
        cand = Path(p)
        if not cand.is_absolute():
            cand = (runtime.config_path.parent / cand).resolve()
        return cand if cand.exists() else None
    except Exception:
        return None


def _plot_theta_group_uncertainty_bands(
    df_val: pd.DataFrame,
    theta_headers: Sequence[str],
    model: "CleanAutoregressiveMFGP",
    out_dir: Path,
    band_mode: str = "linear_sigma",
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    thx, thy = theta_headers[0], theta_headers[1]
    groups = list(df_val.groupby([thx, thy], dropna=False))
    count = 0

    for (xv, yv), g in groups:
        y_true = g["y_raw"].to_numpy(dtype=float)
        if len(y_true) == 0:
            continue

        x_pred = np.array([[float(xv), float(yv)]], dtype=float)
        mu, std, _, _ = model.predict(x_pred)
        mu = float(mu[0])
        std = float(std[0])
        if not np.isfinite(mu):
            mu = 0.0
        if not np.isfinite(std) or std < 0:
            std = 0.0
        idx = np.arange(len(y_true), dtype=int)
        eps = _choose_log_plot_eps(y_true, np.array([mu]), np.array([mu - 3 * std]), np.array([mu + 3 * std]))
        y_true_log = _safe_log10(y_true, eps)
        mu_log = float(_safe_log10(np.array([mu]), eps)[0])

        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        ax.plot(idx, y_true_log, "o", ms=3.5, alpha=0.7, label="validation log10(y_raw)")
        ax.axhline(mu_log, color="tab:red", lw=1.5, label="MF-GP mean (log10)")
        # Use true model std for both statistics and rendered bands.
        std_true = max(std, 1e-12)
        err = np.abs(y_true - mu)
        n = int(len(y_true))
        c1 = int(np.sum(err <= 1.0 * std_true))
        c2 = int(np.sum(err <= 2.0 * std_true))
        c3 = int(np.sum(err <= 3.0 * std_true))
        y_raw_mean = float(np.mean(y_true))
        y_raw_std = float(np.std(y_true))

        # Render bands using axhspan to guarantee drawing even for tiny/degenerate x ranges.
        sigma_log = float(_log_sigma_from_linear(np.array([mu]), np.array([std_true]), eps)[0])
        if band_mode == "log_sigma":
            mode_title = "Log-σ Bands"
            mean_for_lines = mu_log
            band_label_suffix = "log-σ"
        else:
            mode_title = "Linear-σ Bands"
            mean_for_lines = mu_log
            band_label_suffix = "linear-σ"
        for k, color, alpha, label in [
            (3, "tab:purple", 0.10, f"3σ band ({band_label_suffix})"),
            (2, "tab:orange", 0.18, f"2σ band ({band_label_suffix})"),
            (1, "tab:red", 0.30, f"1σ band ({band_label_suffix})"),
        ]:
            if band_mode == "log_sigma":
                lo = mean_for_lines - k * sigma_log
                hi = mean_for_lines + k * sigma_log
            else:
                lo = float(_safe_log10(np.array([mu - k * std_true]), eps)[0])
                hi = float(_safe_log10(np.array([mu + k * std_true]), eps)[0])
            ax.axhspan(lo, hi, color=color, alpha=alpha, label=label)
        # Draw boundary lines explicitly so band edges are always visible.
        for k, color in [(1, "tab:red"), (2, "tab:orange"), (3, "tab:purple")]:
            if band_mode == "log_sigma":
                hi = mean_for_lines + k * sigma_log
                lo = mean_for_lines - k * sigma_log
            else:
                hi = float(_safe_log10(np.array([mu + k * std_true]), eps)[0])
                lo = float(_safe_log10(np.array([mu - k * std_true]), eps)[0])
            ax.axhline(hi, color=color, lw=0.8, alpha=0.9)
            ax.axhline(lo, color=color, lw=0.8, alpha=0.9)
        ax.set_title(f"Theta ({thx}={xv}, {thy}={yv}) | Log Scale | {mode_title}")
        ax.set_xlabel("Sample index")
        ax.set_ylabel(f"log10(y_raw, floor={eps:.2e})")
        # Mild zoom-in around the main mass while keeping uncertainty region in view.
        if n >= 5:
            q_lo, q_hi = np.quantile(y_true_log, [0.02, 0.98])
            if band_mode == "log_sigma":
                y_lo = float(min(q_lo, mean_for_lines - 3 * sigma_log))
                y_hi = float(max(q_hi, mean_for_lines + 3 * sigma_log))
            else:
                y_lo = float(min(q_lo, _safe_log10(np.array([mu - 3 * std_true]), eps)[0]))
                y_hi = float(max(q_hi, _safe_log10(np.array([mu + 3 * std_true]), eps)[0]))
        else:
            if band_mode == "log_sigma":
                y_lo = float(min(np.min(y_true_log), mean_for_lines - 3 * sigma_log))
                y_hi = float(max(np.max(y_true_log), mean_for_lines + 3 * sigma_log))
            else:
                y_lo = float(min(np.min(y_true_log), _safe_log10(np.array([mu - 3 * std_true]), eps)[0]))
                y_hi = float(max(np.max(y_true_log), _safe_log10(np.array([mu + 3 * std_true]), eps)[0]))
        y_span = max(y_hi - y_lo, 1e-9)
        ax.set_ylim(y_lo - 0.10 * y_span, y_hi + 0.10 * y_span)
        ax.grid(True, alpha=0.3)
        # Add numeric diagnostics directly in the legend.
        extra = [
            Line2D([0], [0], color="none", label=f"n points: {n}"),
            Line2D([0], [0], color="none", label=f"y_raw mean: {y_raw_mean:.6g}"),
            Line2D([0], [0], color="none", label=f"y_raw std: {y_raw_std:.6g}"),
            Line2D([0], [0], color="none", label=f"model mean: {mu:.6g}"),
            Line2D([0], [0], color="none", label=f"model std: {std_true:.6g}"),
            Line2D([0], [0], color="none", label=f"log std approx: {sigma_log:.6g}"),
            Line2D([0], [0], color="none", label=f"log floor: {eps:.2e}"),
            Line2D([0], [0], color="none", label=f"within 1σ: {c1}/{n}"),
            Line2D([0], [0], color="none", label=f"within 2σ: {c2}/{n}"),
            Line2D([0], [0], color="none", label=f"within 3σ: {c3}/{n}"),
        ]
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles + extra, labels + [h.get_label() for h in extra], loc="best", fontsize=8, frameon=True)
        fig.tight_layout()
        mode_suffix = "log_sigma" if band_mode == "log_sigma" else "linear_sigma"
        f = out_dir / f"uncertainty_bands_theta_{int(round(float(xv)))}_{int(round(float(yv)))}_{mode_suffix}.png"
        fig.savefig(f, dpi=170)
        plt.close(fig)
        count += 1
    return count


def _plot_across_thetas(
    df_val: pd.DataFrame,
    theta_headers: Sequence[str],
    model: "CleanAutoregressiveMFGP",
    out_path: Path,
    band_mode: str = "linear_sigma",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    thx, thy = theta_headers[0], theta_headers[1]
    agg = (
        df_val.groupby([thx, thy], as_index=False)
        .agg(y_raw_mean=("y_raw", "mean"))
        .sort_values([thx, thy], kind="mergesort")
        .reset_index(drop=True)
    )
    x_pred = agg[[thx, thy]].to_numpy(dtype=float)
    mu, std, _, _ = model.predict(x_pred)

    idx = np.arange(len(agg), dtype=int)
    y_true = agg["y_raw_mean"].to_numpy(dtype=float)
    y_pred = mu.astype(float)
    y_std = std.astype(float)
    y_std_true = np.maximum(y_std, 1e-12)
    err = np.abs(y_true - y_pred)
    eps = _choose_log_plot_eps(y_true, y_pred, y_pred - 3 * y_std_true, y_pred + 3 * y_std_true)
    y_true_log = _safe_log10(y_true, eps)
    y_pred_log = _safe_log10(y_pred, eps)
    y_std_log = _log_sigma_from_linear(y_pred, y_std_true, eps)
    n = int(len(y_true))
    c1 = int(np.sum(err <= 1.0 * y_std_true))
    c2 = int(np.sum(err <= 2.0 * y_std_true))
    c3 = int(np.sum(err <= 3.0 * y_std_true))

    fig, ax = plt.subplots(1, 1, figsize=(max(10, 0.18 * len(idx)), 5))
    ax.plot(idx, y_true_log, "o", ms=4, color="black", alpha=0.75, label="Validation log10(y_raw mean)")
    ax.plot(idx, y_pred_log, "-", lw=1.5, color="tab:blue", label="MF-GP mean (log10)")
    for k, color, alpha, label in [
        (1, "tab:blue", 0.24, "1σ band"),
        (2, "tab:orange", 0.16, "2σ band"),
        (3, "tab:purple", 0.10, "3σ band"),
    ]:
        if band_mode == "log_sigma":
            lo = y_pred_log - k * y_std_log
            hi = y_pred_log + k * y_std_log
        else:
            lo = _safe_log10(y_pred - k * y_std_true, eps)
            hi = _safe_log10(y_pred + k * y_std_true, eps)
        ax.fill_between(idx, lo, hi, color=color, alpha=alpha, label=label)
    ax.set_xlabel(f"Theta index (sorted by {thx}, {thy})")
    ax.set_ylabel(f"log10(y, floor={eps:.2e})")
    title_suffix = "Log-σ Bands" if band_mode == "log_sigma" else "Linear-σ Bands"
    ax.set_title(f"Validation Thetas: Mean y_raw vs MF-GP Prediction (Log Scale, {title_suffix})")
    ax.grid(True, alpha=0.3)
    extra = [
        Line2D([0], [0], color="none", label=f"n points: {n}"),
        Line2D([0], [0], color="none", label=f"band mode: {band_mode}"),
        Line2D([0], [0], color="none", label=f"log floor: {eps:.2e}"),
        Line2D([0], [0], color="none", label=f"within 1σ: {c1}/{n}"),
        Line2D([0], [0], color="none", label=f"within 2σ: {c2}/{n}"),
        Line2D([0], [0], color="none", label=f"within 3σ: {c3}/{n}"),
    ]
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + extra, labels + [h.get_label() for h in extra], loc="best", fontsize=8, frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

    return y_true, y_pred, y_std


def _plot_coverage_summary(y_true: np.ndarray, y_pred: np.ndarray, y_std: np.ndarray, out_path: Path) -> None:
    if len(y_true) == 0:
        return
    err = np.abs(y_true - y_pred)
    s = np.maximum(y_std, 1e-12)
    cov1 = float(np.mean(err <= 1.0 * s))
    cov2 = float(np.mean(err <= 2.0 * s))
    cov3 = float(np.mean(err <= 3.0 * s))
    mae = float(np.mean(err))

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    labels = ["1σ coverage", "2σ coverage", "3σ coverage"]
    vals = [cov1, cov2, cov3]
    ax.bar(labels, vals, color=["tab:blue", "tab:green", "tab:purple"], alpha=0.85)
    ax.axhline(0.68, ls="--", lw=1, color="tab:blue", alpha=0.8, label="Ideal 1σ ~ 0.68")
    ax.axhline(0.95, ls="--", lw=1, color="tab:green", alpha=0.8, label="Ideal 2σ ~ 0.95")
    ax.axhline(0.997, ls="--", lw=1, color="tab:purple", alpha=0.8, label="Ideal 3σ ~ 0.997")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Fraction")
    ax.set_title(f"Coverage Summary (MAE={mae:.4g})")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def _aggregate_theta_predictions_from_df(
    df: pd.DataFrame,
    theta_headers: Sequence[str],
) -> pd.DataFrame:
    thx, thy = theta_headers[0], theta_headers[1]
    need = {thx, thy, "y_raw", "mf_mean", "mf_std"}
    missing = sorted(need - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns for theta aggregation: {missing}")
    return (
        df.groupby([thx, thy], as_index=False)
        .agg(
            y_true=("y_raw", "mean"),
            y_pred=("mf_mean", "mean"),
            y_std=("mf_std", "mean"),
        )
        .sort_values([thx, thy], kind="mergesort")
        .reset_index(drop=True)
    )


def _aggregate_theta_predictions_from_model(
    df: pd.DataFrame,
    theta_headers: Sequence[str],
    model: "CleanAutoregressiveMFGP",
) -> pd.DataFrame:
    thx, thy = theta_headers[0], theta_headers[1]
    agg = (
        df.groupby([thx, thy], as_index=False)
        .agg(y_true=("y_raw", "mean"))
        .sort_values([thx, thy], kind="mergesort")
        .reset_index(drop=True)
    )
    x_pred = agg[[thx, thy]].to_numpy(dtype=float)
    mu, std, _, _ = model.predict(x_pred)
    agg["y_pred"] = mu.astype(float)
    agg["y_std"] = std.astype(float)
    return agg


def _plot_across_theta_series(
    agg: pd.DataFrame,
    out_path: Path,
    title: str,
    *,
    y_mode: str = "linear",
    band_mode: str = "linear_sigma",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_true = agg["y_true"].to_numpy(dtype=float)
    y_pred = agg["y_pred"].to_numpy(dtype=float)
    y_std = np.maximum(agg["y_std"].to_numpy(dtype=float), 1e-12)
    idx = np.arange(len(agg), dtype=int)
    err = np.abs(y_true - y_pred)
    n = int(len(y_true))
    c1 = int(np.sum(err <= 1.0 * y_std))
    c2 = int(np.sum(err <= 2.0 * y_std))
    c3 = int(np.sum(err <= 3.0 * y_std))

    fig, ax = plt.subplots(1, 1, figsize=(max(10, 0.25 * max(1, len(idx))), 5))
    if y_mode == "linear":
        ax.plot(idx, y_true, "o", ms=4, color="black", alpha=0.75, label="HF y_raw mean")
        ax.plot(idx, y_pred, "-", lw=1.5, color="tab:blue", label="MF-GP mean")
        for k, color, alpha, label in [
            (1, "tab:blue", 0.24, "1σ band"),
            (2, "tab:orange", 0.16, "2σ band"),
            (3, "tab:purple", 0.10, "3σ band"),
        ]:
            ax.fill_between(idx, y_pred - k * y_std, y_pred + k * y_std, color=color, alpha=alpha, label=label)
        ax.set_ylabel("y")
        extra = [
            Line2D([0], [0], color="none", label=f"n points: {n}"),
            Line2D([0], [0], color="none", label=f"within 1σ: {c1}/{n}"),
            Line2D([0], [0], color="none", label=f"within 2σ: {c2}/{n}"),
            Line2D([0], [0], color="none", label=f"within 3σ: {c3}/{n}"),
        ]
    else:
        eps = _choose_log_plot_eps(y_true, y_pred, y_pred - 3 * y_std, y_pred + 3 * y_std)
        y_true_plot = _safe_log10(y_true, eps)
        y_pred_plot = _safe_log10(y_pred, eps)
        y_std_log = _log_sigma_from_linear(y_pred, y_std, eps)
        ax.plot(idx, y_true_plot, "o", ms=4, color="black", alpha=0.75, label="HF log10(y_raw mean)")
        ax.plot(idx, y_pred_plot, "-", lw=1.5, color="tab:blue", label="MF-GP mean (log10)")
        for k, color, alpha, label in [
            (1, "tab:blue", 0.24, "1σ band"),
            (2, "tab:orange", 0.16, "2σ band"),
            (3, "tab:purple", 0.10, "3σ band"),
        ]:
            if band_mode == "log_sigma":
                lo = y_pred_plot - k * y_std_log
                hi = y_pred_plot + k * y_std_log
            else:
                lo = _safe_log10(y_pred - k * y_std, eps)
                hi = _safe_log10(y_pred + k * y_std, eps)
            ax.fill_between(idx, lo, hi, color=color, alpha=alpha, label=label)
        ax.set_ylabel(f"log10(y, floor={eps:.2e})")
        extra = [
            Line2D([0], [0], color="none", label=f"n points: {n}"),
            Line2D([0], [0], color="none", label=f"band mode: {band_mode}"),
            Line2D([0], [0], color="none", label=f"log floor: {eps:.2e}"),
            Line2D([0], [0], color="none", label=f"within 1σ: {c1}/{n}"),
            Line2D([0], [0], color="none", label=f"within 2σ: {c2}/{n}"),
            Line2D([0], [0], color="none", label=f"within 3σ: {c3}/{n}"),
        ]

    ax.set_xlabel("Theta index (sorted)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + extra, labels + [h.get_label() for h in extra], loc="best", fontsize=8, frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return y_true, y_pred, y_std



_TRANSFORM_TITLES = {
    "linear": "Normal MF-GP",
    "log_hf": "Log HF: emulate log10(y_raw)",
    "log_lf": "Log LF: use log10(y_cnp)",
    "log_both": "Log HF + Log LF: use log10(y_raw) and log10(y_cnp)",
}


def _predict_in_chunks(
    model: CleanAutoregressiveMFGP,
    x: np.ndarray,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    chunk_size = max(1, int(chunk_size))
    means: list[np.ndarray] = []
    stds: list[np.ndarray] = []
    for start in range(0, len(x), chunk_size):
        mean, std, _, _ = model.predict(x[start:start + chunk_size])
        means.append(np.asarray(mean, dtype=float))
        stds.append(np.asarray(std, dtype=float))
    if not means:
        return np.empty(0, dtype=float), np.empty(0, dtype=float)
    return np.concatenate(means), np.concatenate(stds)


def _pow10_stable(values: np.ndarray) -> np.ndarray:
    return np.power(10.0, np.clip(np.asarray(values, dtype=float), -300.0, 300.0))


def _prediction_interval_in_output_space(
    mean_model: np.ndarray,
    std_model: np.ndarray,
    *,
    use_log_hf: bool,
    sigma: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean_model = np.asarray(mean_model, dtype=float)
    std_model = np.maximum(np.asarray(std_model, dtype=float), 1e-12)
    k = float(sigma)
    if use_log_hf:
        center = _pow10_stable(mean_model)
        lower = _pow10_stable(mean_model - k * std_model)
        upper = _pow10_stable(mean_model + k * std_model)
    else:
        center = mean_model.copy()
        lower = mean_model - k * std_model
        upper = mean_model + k * std_model
    return center, lower, upper


def _positive_prediction_grid(
    runtime: MFGPRuntimeConfig,
    x_lf: np.ndarray,
    x_hf: np.ndarray,
    points_per_axis: int,
) -> np.ndarray:
    if len(runtime.theta_headers) != 2:
        raise ValueError(
            "The requested R-Z map requires exactly two theta_headers; "
            f"found {runtime.theta_headers}"
        )
    combined = np.vstack([x_lf, x_hf]).astype(float)
    observed_lower = np.nanmin(combined, axis=0)
    observed_upper = np.nanmax(combined, axis=0)

    configured_lower = np.asarray(runtime.theta_min, dtype=float) if len(runtime.theta_min) == 2 else observed_lower
    configured_upper = np.asarray(runtime.theta_max, dtype=float) if len(runtime.theta_max) == 2 else observed_upper
    configured_valid = (
        np.all(np.isfinite(configured_lower))
        and np.all(np.isfinite(configured_upper))
        and np.all(configured_upper > configured_lower)
        and np.all(observed_lower >= configured_lower - 1e-9)
        and np.all(observed_upper <= configured_upper + 1e-9)
    )
    lower = configured_lower if configured_valid else observed_lower
    upper = configured_upper if configured_valid else observed_upper
    lower = np.maximum(lower, 0.0)
    if np.any(upper <= lower):
        raise ValueError(
            "Cannot construct a positive R-Z grid. Need non-degenerate positive "
            f"bounds, got lower={lower.tolist()}, upper={upper.tolist()}."
        )
    n = max(2, int(points_per_axis))
    axis0 = np.linspace(lower[0], upper[0], n)
    axis1 = np.linspace(lower[1], upper[1], n)
    gx, gy = np.meshgrid(axis0, axis1, indexing="xy")
    return np.column_stack([gx.ravel(), gy.ravel()])


def _minimum_point_count(input_dim: int) -> int:
    # For two inputs this is four unique points: the smallest non-smoke-test
    # design that can span both axes and leave at least one degree of freedom.
    return max(4, int(input_dim) + 2)


def _validate_unique_point_count(
    frame: pd.DataFrame,
    x_cols: Sequence[str],
    *,
    label: str,
    minimum: int,
    require_axis_span: bool,
) -> None:
    unique = frame[list(x_cols)].drop_duplicates()
    n_unique = len(unique)
    if n_unique < int(minimum):
        raise ValueError(
            f"MF-GP was not run: {label} has only {n_unique} unique theta points; "
            f"at least {int(minimum)} are required. Add more data instead of "
            "using the previous smoke-test path."
        )
    if require_axis_span:
        missing_span = [column for column in x_cols if unique[column].nunique(dropna=False) < 2]
        if missing_span:
            raise ValueError(
                f"MF-GP was not run: {label} does not span both theta axes. "
                f"Need at least two unique values in {missing_span}."
            )


def _load_validation_hf_points(
    validation_csv: str | Path,
    x_cols: Sequence[str],
    iteration: int,
) -> pd.DataFrame:
    path = Path(validation_csv).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Validation CNP CSV not found: {path}")
    frame = _aggregate_rows(pd.read_csv(path), x_cols)
    frame = frame[frame["iteration"].astype(int) == int(iteration)].copy()
    if frame.empty:
        raise ValueError(f"No validation rows found for iteration={iteration}: {path}")
    fidelities = sorted(frame["fidelity"].astype(int).unique().tolist())
    if FIDELITY_HF not in fidelities:
        raise ValueError(
            "Validation CSV must contain fidelity=1 HF rows; "
            f"found fidelities {fidelities}."
        )
    return (
        frame[frame["fidelity"].astype(int) == FIDELITY_HF]
        .sort_values(list(x_cols), kind="mergesort")
        .reset_index(drop=True)
    )


def _regression_metrics_with_coverage(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mean_model: np.ndarray,
    std_model: np.ndarray,
    *,
    use_log_hf: bool,
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residual = y_pred - y_true
    result = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else float("nan"),
        "mean_residual": float(np.mean(residual)),
        "max_abs_residual": float(np.max(np.abs(residual))),
    }
    for k in (1, 2, 3):
        _, lower, upper = _prediction_interval_in_output_space(
            mean_model, std_model, use_log_hf=use_log_hf, sigma=float(k)
        )
        result[f"coverage_{k}sigma"] = float(np.mean((y_true >= lower) & (y_true <= upper)))
    return result


def _plot_validation_hf_points(
    validation: pd.DataFrame,
    x_cols: Sequence[str],
    mean_model: np.ndarray,
    std_model: np.ndarray,
    out_path: Path,
    *,
    use_log_hf: bool,
    title: str,
) -> None:
    """One visible truth/prediction comparison for each unique validation HF theta."""
    y_true = validation["y_raw"].to_numpy(dtype=float)
    y_pred, lower, upper = _prediction_interval_in_output_space(
        mean_model, std_model, use_log_hf=use_log_hf, sigma=1.0
    )
    yerr = np.vstack([
        np.maximum(y_pred - lower, 0.0),
        np.maximum(upper - y_pred, 0.0),
    ])
    idx = np.arange(len(validation), dtype=int)

    fig_width = max(10.0, min(20.0, 0.48 * len(validation) + 5.0))
    fig, ax = plt.subplots(figsize=(fig_width, 6.2))

    # The connector is the residual; the error bar is the model uncertainty.
    ax.vlines(idx, y_true, y_pred, color="0.60", lw=1.1, alpha=0.8, label="truth-to-prediction residual")
    ax.scatter(idx, y_true, marker="x", s=48, color="black", linewidth=1.3, label="HF truth")
    ax.errorbar(
        idx,
        y_pred,
        yerr=yerr,
        fmt="o",
        ms=5,
        capsize=3,
        elinewidth=1.1,
        color="tab:blue",
        ecolor="tab:blue",
        alpha=0.9,
        label="MF-GP prediction ±1σ",
    )

    labels = [
        "(" + ", ".join(f"{float(row[column]):g}" for column in x_cols) + ")"
        for _, row in validation.iterrows()
    ]
    if len(labels) <= 30:
        ax.set_xticks(idx)
        ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=8)
        ax.set_xlabel(f"HF theta point ({', '.join(x_cols)})")
    else:
        ax.set_xlabel(f"HF point index (sorted by {', '.join(x_cols)})")

    inside = (y_true >= lower) & (y_true <= upper)
    ax.set_ylabel("HF target / MF-GP prediction")
    ax.set_title(f"{title}\nEach marker is one unique validation HF theta point")
    ax.grid(True, alpha=0.25)
    handles, legend_labels = ax.get_legend_handles_labels()
    extra = Line2D([0], [0], color="none", label=f"within 1σ: {int(inside.sum())}/{len(inside)}")
    ax.legend(handles + [extra], legend_labels + [extra.get_label()], loc="best", fontsize=8, frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_validation_parity_pointwise(
    y_true: np.ndarray,
    mean_model: np.ndarray,
    std_model: np.ndarray,
    out_path: Path,
    *,
    use_log_hf: bool,
    title: str,
) -> None:
    y_true = np.asarray(y_true, dtype=float)
    y_pred, lower, upper = _prediction_interval_in_output_space(
        mean_model, std_model, use_log_hf=use_log_hf, sigma=1.0
    )
    yerr = np.vstack([
        np.maximum(y_pred - lower, 0.0),
        np.maximum(upper - y_pred, 0.0),
    ])

    fig, ax = plt.subplots(figsize=(6.8, 6.4))
    ax.errorbar(
        y_true,
        y_pred,
        yerr=yerr,
        fmt="o",
        ms=5,
        capsize=3,
        elinewidth=1.0,
        color="tab:blue",
        ecolor="tab:blue",
        alpha=0.82,
    )
    lo = float(min(np.min(y_true), np.min(lower)))
    hi = float(max(np.max(y_true), np.max(upper)))
    if np.isclose(lo, hi):
        pad = max(abs(lo) * 0.05, 1e-6)
        lo, hi = lo - pad, hi + pad
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.0, label="perfect prediction")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("HF truth (y_raw)")
    ax.set_ylabel("MF-GP prediction")
    ax.set_title(f"{title}\nValidation parity with ±1σ prediction uncertainty")
    ax.grid(True, alpha=0.28)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

def run_clean_mfgp(
    config_path: str | Path,
    cnp_csv: Optional[str | Path] = None,
    validation_csv: Optional[str | Path] = None,
    iteration: int = 0,
    target_transform: str = "linear",
    log_epsilon: Optional[float] = None,
    grid_points_per_axis: int = 120,
    random_state: int = 42,
    prefer_validation_csv: bool = False,
    predict_chunk_size: int = 20000,
    verbose: bool = True,
    min_lf_points: Optional[int] = None,
    min_hf_points: Optional[int] = None,
    min_validation_points: int = 3,
    require_validation: bool = True,
) -> MFGPResult:
    """Fit one scalar two-level MF-GP target-transform experiment.

    The fit is rejected before GP optimization when there are too few unique
    theta points. The three standard figures are a positive-quadrant mean/std
    map, a pointwise validation comparison, and a validation parity plot.
    """
    t0 = time.time()
    runtime = load_runtime_config(config_path)
    runtime.out_dir_mfgp.mkdir(parents=True, exist_ok=True)
    transform_mode = _target_transform_suffix(target_transform)
    experiment_title = _TRANSFORM_TITLES[transform_mode]
    artifact_tag = f"{runtime.version}_{transform_mode}"
    x_cols = list(runtime.theta_headers)
    if len(x_cols) != 2:
        raise ValueError(
            "This plotting workflow requires exactly two theta_headers for the R-Z maps; "
            f"found {x_cols}."
        )

    cnp_csv_path = (
        Path(cnp_csv).expanduser().resolve()
        if cnp_csv
        else discover_cnp_output_csv(
            runtime.out_dir_cnp,
            runtime.version,
            prefer_validation=prefer_validation_csv,
        )
    )
    if not cnp_csv_path.exists():
        raise FileNotFoundError(
            _missing_csv_message(cnp_csv_path, runtime.out_dir_cnp, runtime.version)
        )

    if verbose:
        print(f"=== {experiment_title} ===")
        print(f"[stage] Training CNP CSV: {cnp_csv_path}")

    _, lf, hf, selected_lf, selected_hf = load_mfgp_training_data(
        cnp_csv_path,
        x_cols=x_cols,
        iteration=iteration,
    )

    default_minimum = _minimum_point_count(len(x_cols))
    minimum_lf = default_minimum if min_lf_points is None else int(min_lf_points)
    minimum_hf = default_minimum if min_hf_points is None else int(min_hf_points)
    _validate_unique_point_count(
        lf, x_cols, label="LF training data", minimum=minimum_lf, require_axis_span=True
    )
    _validate_unique_point_count(
        hf, x_cols, label="HF training data", minimum=minimum_hf, require_axis_span=True
    )

    val_csv_path: Optional[Path]
    if validation_csv is not None:
        val_csv_path = Path(validation_csv).expanduser().resolve()
    else:
        val_csv_path = _resolve_optional_validation_csv(runtime)
    if require_validation and val_csv_path is None:
        raise ValueError(
            "MF-GP was not run: a validation CSV is required to produce the three "
            "requested diagnostic plots. Pass validation_csv=validation_prediction.mfgp_path."
        )

    validation_hf: Optional[pd.DataFrame] = None
    if val_csv_path is not None:
        validation_hf = _load_validation_hf_points(val_csv_path, x_cols, iteration)
        _validate_unique_point_count(
            validation_hf,
            x_cols,
            label="HF validation data",
            minimum=max(1, int(min_validation_points)),
            require_axis_span=False,
        )
        if verbose:
            print(f"[stage] Validation CNP CSV: {val_csv_path}")

    x_lf = lf[x_cols].to_numpy(dtype=float)
    x_hf = hf[x_cols].to_numpy(dtype=float)
    y_lf_raw = lf["y_cnp"].to_numpy(dtype=float)
    y_hf_raw = hf["y_raw"].to_numpy(dtype=float)
    y_lf, y_hf, eps, use_log_lf, use_log_hf = _transform_mfgp_targets(
        y_lf_raw,
        y_hf_raw,
        transform_mode,
        eps=log_epsilon,
    )

    lf_sigma = np.maximum(lf["y_cnp_err"].to_numpy(dtype=float), 1e-12)
    if use_log_lf:
        lf_sigma = _log_sigma_from_linear(y_lf_raw, lf_sigma, eps)
    alpha_lf = max(float(np.mean(np.square(lf_sigma))), 1e-10)
    alpha_hf = 1e-10

    model = CleanAutoregressiveMFGP(
        random_state=random_state,
        alpha_lf=alpha_lf,
        alpha_hf=alpha_hf,
    )
    model.fit(
        x_lf=x_lf,
        y_lf=y_lf,
        x_hf=x_hf,
        y_hf=y_hf,
        verbose=verbose,
    )

    hf_mean_model, hf_std_model = _predict_in_chunks(model, x_hf, predict_chunk_size)
    hf_pred, _, _ = _prediction_interval_in_output_space(
        hf_mean_model, hf_std_model, use_log_hf=use_log_hf
    )
    training_metrics = _regression_metrics_with_coverage(
        y_hf_raw,
        hf_pred,
        hf_mean_model,
        hf_std_model,
        use_log_hf=use_log_hf,
    )

    grid_xy = _positive_prediction_grid(runtime, x_lf, x_hf, grid_points_per_axis)
    grid_mean_model, grid_std_model = _predict_in_chunks(model, grid_xy, predict_chunk_size)
    grid_pred, grid_lower, grid_upper = _prediction_interval_in_output_space(
        grid_mean_model, grid_std_model, use_log_hf=use_log_hf
    )
    grid_uncertainty = 0.5 * np.maximum(grid_upper - grid_lower, 0.0)

    pred_csv = runtime.out_dir_mfgp / f"mfgp_{artifact_tag}_hf_training_predictions_iter{iteration}.csv"
    grid_csv = runtime.out_dir_mfgp / f"mfgp_{artifact_tag}_positive_grid_iter{iteration}.csv"
    hf_output = hf[x_cols + ["y_raw", "y_cnp", "y_cnp_err"]].copy()
    hf_output["mf_mean_model_space"] = hf_mean_model
    hf_output["mf_std_model_space"] = hf_std_model
    hf_output["mf_prediction"] = hf_pred
    _, hf_lower, hf_upper = _prediction_interval_in_output_space(
        hf_mean_model, hf_std_model, use_log_hf=use_log_hf
    )
    hf_output["mf_lower_1sigma"] = hf_lower
    hf_output["mf_upper_1sigma"] = hf_upper
    hf_output["residual"] = hf_pred - y_hf_raw
    hf_output.to_csv(pred_csv, index=False)

    grid_output = pd.DataFrame(grid_xy, columns=x_cols)
    grid_output["mf_mean_model_space"] = grid_mean_model
    grid_output["mf_std_model_space"] = grid_std_model
    grid_output["mf_prediction"] = grid_pred
    grid_output["mf_lower_1sigma"] = grid_lower
    grid_output["mf_upper_1sigma"] = grid_upper
    grid_output["mf_uncertainty_1sigma"] = grid_uncertainty
    grid_output.to_csv(grid_csv, index=False)

    mean_std_plot = runtime.out_dir_mfgp / f"mfgp_{artifact_tag}_mean_uncertainty_positive_iter{iteration}.png"
    _plot_mean_std_heatmaps(
        grid_xy,
        grid_pred,
        grid_uncertainty,
        x_cols,
        x_hf,
        mean_std_plot,
        title=f"{experiment_title}: MF-GP prediction over R≥0, Z≥0",
    )

    validation_point_plot: Optional[Path] = None
    validation_parity_plot: Optional[Path] = None
    validation_metrics: Optional[dict[str, float]] = None
    validation_prediction_csv: Optional[Path] = None
    if validation_hf is not None:
        x_val = validation_hf[x_cols].to_numpy(dtype=float)
        val_mean_model, val_std_model = _predict_in_chunks(model, x_val, predict_chunk_size)
        y_val = validation_hf["y_raw"].to_numpy(dtype=float)
        val_pred, val_lower, val_upper = _prediction_interval_in_output_space(
            val_mean_model, val_std_model, use_log_hf=use_log_hf
        )
        validation_metrics = _regression_metrics_with_coverage(
            y_val,
            val_pred,
            val_mean_model,
            val_std_model,
            use_log_hf=use_log_hf,
        )

        validation_prediction_csv = runtime.out_dir_mfgp / f"mfgp_{artifact_tag}_validation_predictions_iter{iteration}.csv"
        val_output = validation_hf[x_cols + ["y_raw"]].copy()
        val_output["mf_mean_model_space"] = val_mean_model
        val_output["mf_std_model_space"] = val_std_model
        val_output["mf_prediction"] = val_pred
        val_output["mf_lower_1sigma"] = val_lower
        val_output["mf_upper_1sigma"] = val_upper
        val_output["residual"] = val_pred - y_val
        val_output["absolute_residual"] = np.abs(val_output["residual"])
        val_output["within_1sigma"] = (y_val >= val_lower) & (y_val <= val_upper)
        val_output.to_csv(validation_prediction_csv, index=False)

        validation_point_plot = runtime.out_dir_mfgp / f"mfgp_{artifact_tag}_validation_hf_points_iter{iteration}.png"
        validation_parity_plot = runtime.out_dir_mfgp / f"mfgp_{artifact_tag}_validation_parity_iter{iteration}.png"
        _plot_validation_hf_points(
            validation_hf,
            x_cols,
            val_mean_model,
            val_std_model,
            validation_point_plot,
            use_log_hf=use_log_hf,
            title=f"{experiment_title}: validation HF truth vs prediction",
        )
        _plot_validation_parity_pointwise(
            y_val,
            val_mean_model,
            val_std_model,
            validation_parity_plot,
            use_log_hf=use_log_hf,
            title=experiment_title,
        )

    model_json = runtime.out_dir_mfgp / f"mfgp_{artifact_tag}_model_iter{iteration}.json"
    model_json.write_text(json.dumps({
        "version": runtime.version,
        "config_path": str(runtime.config_path),
        "cnp_csv": str(cnp_csv_path),
        "validation_csv": str(val_csv_path) if val_csv_path is not None else None,
        "iteration": int(iteration),
        "target_transform": transform_mode,
        "use_log_lf": bool(use_log_lf),
        "use_log_hf": bool(use_log_hf),
        "log_epsilon": float(eps),
        "theta_headers": x_cols,
        "fidelity_definition": {"0": "low fidelity y_cnp", "1": "high fidelity y_raw"},
        "selected_lf": int(selected_lf),
        "selected_hf": int(selected_hf),
        "n_lf_points": int(len(lf)),
        "n_hf_points": int(len(hf)),
        "n_validation_hf_points": int(len(validation_hf)) if validation_hf is not None else 0,
        "minimum_lf_points": int(minimum_lf),
        "minimum_hf_points": int(minimum_hf),
        "minimum_validation_points": int(min_validation_points),
        "rho": float(model.rho),
        "alpha_lf": float(alpha_lf),
        "alpha_hf": float(alpha_hf),
        "lf_kernel": str(model.gp_lf.kernel_) if model.gp_lf is not None else None,
        "hf_discrepancy_kernel": str(model.gp_d.kernel_) if model.gp_d is not None else None,
    }, indent=2))

    metrics_json = runtime.out_dir_mfgp / f"mfgp_{artifact_tag}_metrics_iter{iteration}.json"
    metrics_json.write_text(json.dumps({
        "transform": transform_mode,
        "training": training_metrics,
        "validation": validation_metrics,
        "n_lf_points": int(len(lf)),
        "n_hf_points": int(len(hf)),
        "n_validation_hf_points": int(len(validation_hf)) if validation_hf is not None else 0,
        "rho": float(model.rho),
        "prediction_csv": str(pred_csv),
        "validation_prediction_csv": str(validation_prediction_csv) if validation_prediction_csv is not None else None,
    }, indent=2))

    elapsed = time.time() - t0
    if verbose:
        print(
            f"[done] {experiment_title} | LF points={len(lf)} | HF points={len(hf)} | "
            f"validation HF points={0 if validation_hf is None else len(validation_hf)}"
        )
        if validation_metrics is not None:
            print(
                "[done] Validation: "
                f"RMSE={validation_metrics['rmse']:.6g}, "
                f"MAE={validation_metrics['mae']:.6g}, "
                f"1σ coverage={validation_metrics['coverage_1sigma']:.3f}"
            )
        print(f"[done] Three plots -> {mean_std_plot}, {validation_point_plot}, {validation_parity_plot}")
        print(f"[done] Elapsed: {elapsed:.1f}s")

    return MFGPResult(
        cnp_csv=cnp_csv_path,
        model_json=model_json,
        metrics_json=metrics_json,
        prediction_csv=pred_csv,
        grid_csv=grid_csv,
        data_plot=None,
        parity_plot=validation_parity_plot,
        mean_std_plot=mean_std_plot,
        mean_std_plot_3d_html=None,
        residual_plot=None,
        theta_group_plot_dir=None,
        across_theta_plot=validation_point_plot,
        across_theta_zoom_plot=None,
        coverage_plot=None,
        validation_parity_plot=validation_parity_plot,
        train_across_theta_linear_plot=None,
        train_across_theta_log_linear_sigma_plot=None,
        train_across_theta_log_log_sigma_plot=None,
        validation_across_theta_linear_plot=validation_point_plot,
        validation_across_theta_log_linear_sigma_plot=None,
        validation_across_theta_log_log_sigma_plot=None,
        train_parity_linear_plot=None,
        train_parity_log_plot=None,
        validation_parity_linear_plot=validation_parity_plot,
        validation_parity_log_plot=None,
    )

def run_mfgp_transform_suite(
    config_path: str | Path,
    cnp_csv: Optional[str | Path] = None,
    validation_csv: Optional[str | Path] = None,
    transforms: Sequence[str] = ("linear", "log_hf", "log_lf", "log_both"),
    iteration: int = 0,
    log_epsilon: Optional[float] = None,
    grid_points_per_axis: int = 120,
    random_state: int = 42,
    predict_chunk_size: int = 20000,
    verbose: bool = True,
    min_lf_points: Optional[int] = None,
    min_hf_points: Optional[int] = None,
    min_validation_points: int = 3,
) -> Dict[str, MFGPResult]:
    """Run each requested transform as an independent MF-GP experiment."""
    normalized: list[str] = []
    for transform in transforms:
        mode = _target_transform_suffix(transform)
        if mode not in normalized:
            normalized.append(mode)
    if not normalized:
        raise ValueError("At least one target transform must be requested")

    results: Dict[str, MFGPResult] = {}
    for mode in normalized:
        results[mode] = run_clean_mfgp(
            config_path=config_path,
            cnp_csv=cnp_csv,
            validation_csv=validation_csv,
            iteration=iteration,
            target_transform=mode,
            log_epsilon=log_epsilon,
            grid_points_per_axis=grid_points_per_axis,
            random_state=int(random_state),
            predict_chunk_size=predict_chunk_size,
            verbose=verbose,
            min_lf_points=min_lf_points,
            min_hf_points=min_hf_points,
            min_validation_points=min_validation_points,
            require_validation=True,
        )
    return results

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Two-level MF-GP transform suite for CNP low-fidelity and HF simulation targets")
    p.add_argument("--config", type=str, default=str(_default_config_path()), help="Path to XLZD settings.yaml")
    p.add_argument("--cnp-csv", type=str, default=None, help="Optional explicit CNP output CSV")
    p.add_argument("--validation-csv", type=str, default=None, help="Optional explicit validation CSV for extra plots")
    p.add_argument("--iteration", type=int, default=0, help="Iteration value to filter")
    p.add_argument(
        "--target-transform",
        type=str,
        default="linear",
        choices=["linear", "log_hf", "log_lf", "log_both"],
        help="Target transform used for this MF-GP experiment.",
    )
    p.add_argument("--log-epsilon", type=float, default=None, help="Optional explicit epsilon for log target transforms")
    p.add_argument("--all-transforms", action="store_true", help="Run linear, log_hf, log_lf, and log_both as four independent experiments")
    p.add_argument("--grid-points", type=int, default=120, help="Grid points per axis")
    p.add_argument("--predict-chunk-size", type=int, default=20000, help="Chunk size for prediction progress")
    p.add_argument("--random-state", type=int, default=42, help="Random seed")
    p.add_argument(
        "--prefer-validation-csv",
        action="store_true",
        help="When auto-discovering CNP CSV, prefer output_validation files first",
    )
    p.add_argument("--quiet", action="store_true", help="Reduce progress logs")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.all_transforms:
        run_mfgp_transform_suite(
            config_path=args.config,
            cnp_csv=args.cnp_csv,
            validation_csv=args.validation_csv,
            iteration=args.iteration,
            log_epsilon=args.log_epsilon,
            grid_points_per_axis=args.grid_points,
            random_state=args.random_state,
            predict_chunk_size=args.predict_chunk_size,
            verbose=not args.quiet,
        )
        return

    run_clean_mfgp(
        config_path=args.config,
        cnp_csv=args.cnp_csv,
        validation_csv=args.validation_csv,
        iteration=args.iteration,
        target_transform=args.target_transform,
        log_epsilon=args.log_epsilon,
        grid_points_per_axis=args.grid_points,
        random_state=args.random_state,
        prefer_validation_csv=args.prefer_validation_csv,
        predict_chunk_size=args.predict_chunk_size,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
