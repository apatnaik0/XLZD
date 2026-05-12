from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


THETA_HEADERS = ["R_max", "Z_max"]
REQUIRED_PER_SIGNAL_COLS = {
    "iteration",
    "fidelity",
    "source_file",
    "event_index",
    "n_samples_file",
    *THETA_HEADERS,
    "y_raw",
    "y_cnp",
    "y_cnp_err",
}
REQUIRED_CNP_AGG_COLS = {
    "iteration",
    "fidelity",
    "n_samples",
    *THETA_HEADERS,
    "y_cnp",
    "y_cnp_err",
    "y_raw",
}

THETA_KEY_DECIMALS = 6
THETA_SNAP_TOLERANCE = 1e-2


@dataclass(slots=True)
class LFPaths:
    repo_root: Path
    config_path: Path
    per_signal_csv: Path
    training_cnp_csv: Path
    validation_cnp_csv: Path
    artifact_dir: Path


def find_repo_root(start: Optional[Path] = None) -> Path:
    start = (start or Path.cwd()).resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "prepare_resum_data.py").exists() and (candidate / "README.md").exists():
            return candidate
    raise RuntimeError("Could not find the XLZD repo root.")


def default_paths(config_path: str | Path | None = None) -> LFPaths:
    repo_root = find_repo_root()
    config_path = Path(config_path) if config_path else repo_root / "src" / "xlzd" / "settings_minibatch.yaml"
    config_path = config_path.resolve()
    raw = yaml.safe_load(config_path.read_text())
    version = str(raw.get("path_settings", {}).get("version", "xlzd_v1"))
    epochs = int(raw.get("cnp_settings", {}).get("training_epochs", 15))
    out_dir_cnp = (config_path.parent / raw.get("path_settings", {}).get("path_out_cnp", "../../data/out/cnp")).resolve()
    artifact_dir = repo_root / "lf_augmentations" / "artifacts"
    return LFPaths(
        repo_root=repo_root,
        config_path=config_path,
        per_signal_csv=out_dir_cnp / f"cnp_{version}_output_per_signal_{epochs}epochs.csv",
        training_cnp_csv=out_dir_cnp / f"cnp_{version}_output_{epochs}epochs.csv",
        validation_cnp_csv=out_dir_cnp / f"cnp_{version}_output_validation_{epochs}epochs.csv",
        artifact_dir=artifact_dir,
    )


def load_per_signal_predictions(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = sorted(REQUIRED_PER_SIGNAL_COLS - set(df.columns))
    if missing:
        raise ValueError(f"Per-signal CSV is missing required columns: {missing}")
    out = df.copy()
    out["theta_key"] = _theta_key_series(out["R_max"], out["Z_max"])
    return out


def load_training_cnp_csv(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = sorted(REQUIRED_CNP_AGG_COLS - set(df.columns))
    if missing:
        raise ValueError(f"Aggregated CNP CSV is missing required columns: {missing}")
    return df.copy()


def _theta_key_series(r_vals: pd.Series, z_vals: pd.Series, decimals: int = THETA_KEY_DECIMALS) -> pd.Series:
    fmt = f"{{:.{decimals}f}}"
    return r_vals.astype(float).map(fmt.format) + "|" + z_vals.astype(float).map(fmt.format)


def _canonicalize_augmented_theta(
    base_training_df: pd.DataFrame,
    new_lf_trials_df: pd.DataFrame,
    *,
    decimals: int = THETA_KEY_DECIMALS,
    snap_tolerance: float = THETA_SNAP_TOLERANCE,
) -> pd.DataFrame:
    """Snap augmented LF theta values to canonical theta pairs in the base LF CSV."""
    lf_base = base_training_df[base_training_df["fidelity"].astype(int) == 0].copy()
    if lf_base.empty or new_lf_trials_df.empty:
        return new_lf_trials_df.copy()

    lf_base["theta_key"] = _theta_key_series(lf_base["R_max"], lf_base["Z_max"], decimals=decimals)
    canonical = (
        lf_base[["theta_key", "R_max", "Z_max"]]
        .drop_duplicates(subset=["theta_key"])
        .set_index("theta_key")
    )

    out = new_lf_trials_df.copy()
    out["theta_key"] = _theta_key_series(out["R_max"], out["Z_max"], decimals=decimals)
    exact_mask = out["theta_key"].isin(canonical.index)
    if exact_mask.any():
        out.loc[exact_mask, "R_max"] = out.loc[exact_mask, "theta_key"].map(canonical["R_max"]).astype(float)
        out.loc[exact_mask, "Z_max"] = out.loc[exact_mask, "theta_key"].map(canonical["Z_max"]).astype(float)

    if (~exact_mask).any():
        canonical_points = canonical[["R_max", "Z_max"]].to_numpy(dtype=float)
        miss_idx = np.flatnonzero(~exact_mask.to_numpy())
        for idx in miss_idx:
            r_val = float(out.at[idx, "R_max"])
            z_val = float(out.at[idx, "Z_max"])
            diff = canonical_points - np.array([r_val, z_val], dtype=float)
            dist = np.sqrt(np.sum(np.square(diff), axis=1))
            best = int(np.argmin(dist))
            best_dist = float(dist[best])
            if not np.isfinite(best_dist) or best_dist > float(snap_tolerance):
                raise ValueError(
                    "Augmented LF rows contain theta values that do not match canonical base LF theta keys. "
                    f"Nearest match for ({r_val:.6f}, {z_val:.6f}) is "
                    f"({canonical_points[best, 0]:.6f}, {canonical_points[best, 1]:.6f}) "
                    f"with distance {best_dist:.6g}, above tolerance {snap_tolerance:.6g}."
                )
            out.at[idx, "R_max"] = float(canonical_points[best, 0])
            out.at[idx, "Z_max"] = float(canonical_points[best, 1])

    out["theta_key"] = _theta_key_series(out["R_max"], out["Z_max"], decimals=decimals)
    return out


def _aggregate_trial_error_from_events(probs: np.ndarray, errs: np.ndarray, trial_size: int) -> float:
    probs = np.asarray(probs, dtype=float)
    errs = np.asarray(errs, dtype=float)
    event_var = np.clip(probs * (1.0 - probs) + np.square(errs), 1e-10, None)
    return float(np.sqrt(np.mean(event_var) / max(int(trial_size), 1)))


def build_file_level_trials(per_signal_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["iteration", "fidelity", "source_file", *THETA_HEADERS]
    for _, part in per_signal_df.groupby(keys, dropna=False, sort=True):
        probs = part["y_cnp"].to_numpy(dtype=float)
        errs = part["y_cnp_err"].to_numpy(dtype=float)
        raw = part["y_raw"].to_numpy(dtype=float)
        n = int(len(part))
        rows.append(
            {
                "iteration": int(part["iteration"].iloc[0]),
                "fidelity": int(part["fidelity"].iloc[0]),
                "source_file": str(part["source_file"].iloc[0]),
                "R_max": float(part["R_max"].iloc[0]),
                "Z_max": float(part["Z_max"].iloc[0]),
                "n_samples": n,
                "y_cnp": float(np.mean(probs)),
                "y_cnp_err": _aggregate_trial_error_from_events(probs, errs, n),
                "y_raw": float(np.mean(raw)),
                "log_prop": np.nan,
                "bce": np.nan,
                "theta_key": str(part["theta_key"].iloc[0]),
            }
        )
    return pd.DataFrame(rows)


def theta_support_summary(file_trials_df: pd.DataFrame, *, fidelity: int = 0) -> pd.DataFrame:
    df = file_trials_df[file_trials_df["fidelity"].astype(int) == int(fidelity)].copy()
    summary = (
        df.groupby(["iteration", *THETA_HEADERS, "theta_key"], dropna=False)
        .agg(
            n_trials=("source_file", "size"),
            total_events=("n_samples", "sum"),
            mean_trial_size=("n_samples", "mean"),
            mean_y_cnp=("y_cnp", "mean"),
            trial_y_cnp_std=("y_cnp", "std"),
            mean_y_cnp_err=("y_cnp_err", "mean"),
        )
        .reset_index()
        .sort_values([THETA_HEADERS[0], THETA_HEADERS[1]])
    )
    return summary


def augment_lf_trials_bootstrap(
    per_signal_df: pd.DataFrame,
    *,
    n_augmented_trials: int = 3,
    trial_size: Optional[int] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    lf = per_signal_df[per_signal_df["fidelity"].astype(int) == 0].copy()
    rows = []
    for (_, iteration, r_max, z_max), part in lf.groupby(["theta_key", "iteration", "R_max", "Z_max"], sort=True):
        file_sizes = part.groupby("source_file", dropna=False).size().to_numpy(dtype=int)
        target_n = int(trial_size or np.median(file_sizes))
        target_n = max(target_n, 2)
        probs_full = part["y_cnp"].to_numpy(dtype=float)
        errs_full = part["y_cnp_err"].to_numpy(dtype=float)
        raw_full = part["y_raw"].to_numpy(dtype=float)
        for trial_idx in range(int(n_augmented_trials)):
            choice = rng.integers(0, len(part), size=target_n)
            probs = probs_full[choice]
            errs = errs_full[choice]
            raw = raw_full[choice]
            rows.append(
                {
                    "iteration": int(iteration),
                    "fidelity": 0,
                    "source_file": f"bootstrap_theta_{r_max:.1f}_{z_max:.1f}_trial_{trial_idx:03d}",
                    "R_max": float(r_max),
                    "Z_max": float(z_max),
                    "n_samples": int(target_n),
                    "y_cnp": float(np.mean(probs)),
                    "y_cnp_err": _aggregate_trial_error_from_events(probs, errs, target_n),
                    "y_raw": float(np.mean(raw)),
                    "log_prop": np.nan,
                    "bce": np.nan,
                    "theta_key": str(part["theta_key"].iloc[0]),
                    "augmentation_method": "bootstrap",
                }
            )
    return pd.DataFrame(rows)


def _aggregate_merged_trial(part: pd.DataFrame) -> dict[str, float | int | str]:
    weights = part["n_samples"].to_numpy(dtype=float)
    weights = weights / np.sum(weights)
    y_cnp = float(np.sum(weights * part["y_cnp"].to_numpy(dtype=float)))
    y_raw = float(np.sum(weights * part["y_raw"].to_numpy(dtype=float)))
    y_cnp_err = float(np.sqrt(np.sum(np.square(weights) * np.square(part["y_cnp_err"].to_numpy(dtype=float)))))
    return {
        "y_cnp": y_cnp,
        "y_raw": y_raw,
        "y_cnp_err": y_cnp_err,
        "n_samples": int(part["n_samples"].sum()),
    }


def augment_lf_trials_merged(
    file_level_trials: pd.DataFrame,
    *,
    n_augmented_trials: int = 3,
    merge_size: int = 4,
    random_state: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    lf = file_level_trials[file_level_trials["fidelity"].astype(int) == 0].copy()
    rows = []
    for (_, iteration, r_max, z_max), part in lf.groupby(["theta_key", "iteration", "R_max", "Z_max"], sort=True):
        base = part.reset_index(drop=True)
        replace = len(base) < int(merge_size)
        for trial_idx in range(int(n_augmented_trials)):
            chosen = rng.choice(len(base), size=int(merge_size), replace=replace)
            merged = base.iloc[chosen].copy()
            agg = _aggregate_merged_trial(merged)
            rows.append(
                {
                    "iteration": int(iteration),
                    "fidelity": 0,
                    "source_file": f"merged_theta_{r_max:.1f}_{z_max:.1f}_trial_{trial_idx:03d}",
                    "R_max": float(r_max),
                    "Z_max": float(z_max),
                    "n_samples": agg["n_samples"],
                    "y_cnp": agg["y_cnp"],
                    "y_cnp_err": agg["y_cnp_err"],
                    "y_raw": agg["y_raw"],
                    "log_prop": np.nan,
                    "bce": np.nan,
                    "theta_key": str(part["theta_key"].iloc[0]),
                    "augmentation_method": "merged_blocks",
                }
            )
    return pd.DataFrame(rows)


def replace_lf_rows_in_training_csv(
    base_training_df: pd.DataFrame,
    new_lf_trials_df: pd.DataFrame,
    *,
    keep_original_lf: bool = True,
) -> pd.DataFrame:
    hf = base_training_df[base_training_df["fidelity"].astype(int) == 1].copy()
    lf_original = base_training_df[base_training_df["fidelity"].astype(int) == 0].copy()
    lf_new = _canonicalize_augmented_theta(base_training_df, new_lf_trials_df)

    if keep_original_lf:
        combined_lf = pd.concat([lf_original, lf_new], ignore_index=True, sort=False)
    else:
        combined_lf = lf_new

    expected_cols = list(base_training_df.columns)
    for col in expected_cols:
        if col not in combined_lf.columns:
            combined_lf[col] = np.nan
        if col not in hf.columns:
            hf[col] = np.nan
    combined = pd.concat([combined_lf[expected_cols], hf[expected_cols]], ignore_index=True, sort=False)
    combined["fidelity"] = combined["fidelity"].astype(int)
    combined["iteration"] = combined["iteration"].astype(int)
    return combined


def write_augmented_training_csv(
    output_path: str | Path,
    base_training_df: pd.DataFrame,
    new_lf_trials_df: pd.DataFrame,
    *,
    keep_original_lf: bool = True,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined = replace_lf_rows_in_training_csv(
        base_training_df,
        new_lf_trials_df,
        keep_original_lf=keep_original_lf,
    )
    combined.to_csv(output_path, index=False)
    return output_path


def write_mfgp_config_variant(
    base_config_path: str | Path,
    *,
    output_path: str | Path,
    version_suffix: str,
    out_dir_mfgp: str | Path,
) -> Path:
    base_config_path = Path(base_config_path)
    output_path = Path(output_path)
    raw = yaml.safe_load(base_config_path.read_text())
    raw.setdefault("path_settings", {})
    raw["path_settings"]["version"] = f"{raw['path_settings'].get('version', 'xlzd_v1')}_{version_suffix}"
    raw["path_settings"]["path_out_mfgp"] = str(out_dir_mfgp)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(raw, sort_keys=False))
    return output_path


def plot_theta_trial_count_comparison(original_trials: pd.DataFrame, augmented_trials: pd.DataFrame, *, title: str) -> plt.Figure:
    orig = theta_support_summary(original_trials)
    aug = theta_support_summary(augmented_trials)
    merged = orig.merge(
        aug[["theta_key", "n_trials", "total_events"]],
        on="theta_key",
        how="outer",
        suffixes=("_orig", "_aug"),
    ).fillna(0)
    labels = [f"R{r:.0f}\nZ{z:.0f}" for r, z in zip(merged["R_max"], merged["Z_max"])]
    x = np.arange(len(merged))
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8))
    axes[0].bar(x - 0.2, merged["n_trials_orig"], width=0.4, label="original")
    axes[0].bar(x + 0.2, merged["n_trials_aug"], width=0.4, label="augmented")
    axes[0].set_title("LF trial count per theta")
    axes[1].bar(x - 0.2, merged["total_events_orig"], width=0.4, label="original")
    axes[1].bar(x + 0.2, merged["total_events_aug"], width=0.4, label="augmented")
    axes[1].set_title("LF event support per theta")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_trial_value_distributions(original_trials: pd.DataFrame, augmented_trials: pd.DataFrame, *, title: str) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    axes[0].hist(original_trials["y_cnp"], bins=35, alpha=0.6, label="original")
    axes[0].hist(augmented_trials["y_cnp"], bins=35, alpha=0.6, label="augmented")
    axes[0].set_title("LF trial y_cnp distribution")
    axes[0].set_xlabel("y_cnp")
    axes[1].hist(original_trials["y_cnp_err"], bins=35, alpha=0.6, label="original")
    axes[1].hist(augmented_trials["y_cnp_err"], bins=35, alpha=0.6, label="augmented")
    axes[1].set_title("LF trial y_cnp_err distribution")
    axes[1].set_xlabel("y_cnp_err")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_theta_mean_shift(original_trials: pd.DataFrame, augmented_trials: pd.DataFrame, *, title: str) -> plt.Figure:
    orig = theta_support_summary(original_trials)[["theta_key", "R_max", "Z_max", "mean_y_cnp", "mean_y_cnp_err"]]
    aug = theta_support_summary(augmented_trials)[["theta_key", "mean_y_cnp", "mean_y_cnp_err"]]
    merged = orig.merge(aug, on="theta_key", suffixes=("_orig", "_aug"))
    labels = [f"R{r:.0f}\nZ{z:.0f}" for r, z in zip(merged["R_max"], merged["Z_max"])]
    x = np.arange(len(merged))
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8))
    axes[0].plot(x, merged["mean_y_cnp_orig"], "o-", label="original")
    axes[0].plot(x, merged["mean_y_cnp_aug"], "o-", label="augmented")
    axes[0].set_title("Theta-level mean y_cnp")
    axes[1].plot(x, merged["mean_y_cnp_err_orig"], "o-", label="original")
    axes[1].plot(x, merged["mean_y_cnp_err_aug"], "o-", label="augmented")
    axes[1].set_title("Theta-level mean y_cnp_err")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.grid(True, alpha=0.25)
        ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_tail_fraction_summary(
    original_trials: pd.DataFrame,
    augmented_trials: pd.DataFrame,
    *,
    quantiles: Iterable[float] = (0.05, 0.10, 0.20, 0.50),
    title: str,
) -> plt.Figure:
    qs = list(quantiles)
    orig_vals = original_trials["y_cnp"].to_numpy(dtype=float)
    aug_vals = augmented_trials["y_cnp"].to_numpy(dtype=float)
    orig_cutoffs = {q: float(np.quantile(orig_vals, q)) for q in qs}
    aug_cutoffs = {q: float(np.quantile(aug_vals, q)) for q in qs}
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    x = np.arange(len(qs))
    ax.bar(x - 0.2, [float(np.mean(orig_vals <= orig_cutoffs[q])) for q in qs], width=0.4, label="original")
    ax.bar(x + 0.2, [float(np.mean(aug_vals <= aug_cutoffs[q])) for q in qs], width=0.4, label="augmented")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(q*100)}%" for q in qs])
    ax.set_ylabel("fraction")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig


def prepare_lf_augmentation_outputs(
    *,
    config_path: str | Path | None = None,
    n_augmented_trials: int = 3,
    merge_size: int = 4,
    random_state: int = 42,
) -> dict[str, Path]:
    paths = default_paths(config_path)
    paths.artifact_dir.mkdir(parents=True, exist_ok=True)
    per_signal_df = load_per_signal_predictions(paths.per_signal_csv)
    training_df = load_training_cnp_csv(paths.training_cnp_csv)
    original_trials = build_file_level_trials(per_signal_df)
    bootstrap_trials = augment_lf_trials_bootstrap(
        per_signal_df,
        n_augmented_trials=n_augmented_trials,
        random_state=random_state,
    )
    merged_trials = augment_lf_trials_merged(
        original_trials,
        n_augmented_trials=n_augmented_trials,
        merge_size=merge_size,
        random_state=random_state,
    )

    out = {
        "original_trials_csv": paths.artifact_dir / "original_lf_file_trials.csv",
        "bootstrap_trials_csv": paths.artifact_dir / "augmented_lf_bootstrap_trials.csv",
        "merged_trials_csv": paths.artifact_dir / "augmented_lf_merged_trials.csv",
        "bootstrap_training_csv": paths.artifact_dir / "cnp_augmented_bootstrap_training.csv",
        "merged_training_csv": paths.artifact_dir / "cnp_augmented_merged_training.csv",
    }
    original_trials.to_csv(out["original_trials_csv"], index=False)
    bootstrap_trials.to_csv(out["bootstrap_trials_csv"], index=False)
    merged_trials.to_csv(out["merged_trials_csv"], index=False)
    write_augmented_training_csv(out["bootstrap_training_csv"], training_df, bootstrap_trials)
    write_augmented_training_csv(out["merged_training_csv"], training_df, merged_trials)
    return out
