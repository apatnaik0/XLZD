#!/usr/bin/env python3
"""Self-contained CNP pipeline for XLZD HDF5 data.

This module intentionally avoids any dependency on the `resum` package.
It provides:
- HDF5 event sampling
- Deterministic Conditional Neural Process model
- Training loop with history/plots/checkpoint
- Prediction/export pipeline compatible with downstream MFGP CSV usage
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from tqdm.auto import tqdm

# -----------------------------
# Configuration and utilities
# -----------------------------

LOW_FIDELITY = 0
HIGH_FIDELITY = 1
VALID_FIDELITIES = {LOW_FIDELITY, HIGH_FIDELITY}


def _validate_binary_fidelity_array(
    values: object,
    *,
    n_events: int,
    context: str,
) -> np.ndarray:
    """Validate per-event fidelity metadata as exactly 0 (LF) or 1 (HF)."""
    arr = np.asarray(values).reshape(-1)
    if arr.size == 1:
        arr = np.full(n_events, arr.item())
    elif len(arr) != n_events:
        raise ValueError(
            f"{context}: fidelity has {len(arr)} values for {n_events} events. "
            "Expected one value per event, or one scalar for a homogeneous block."
        )

    numeric = pd.to_numeric(pd.Series(arr), errors="coerce")
    numeric_values = numeric.to_numpy(dtype=float)
    invalid = (
        numeric.isna().to_numpy()
        | ~np.isfinite(numeric_values)
        | ~np.isclose(numeric_values, np.rint(numeric_values))
        | ~numeric.isin(VALID_FIDELITIES).to_numpy()
    )
    if invalid.any():
        bad_values = arr[invalid].tolist()
        raise ValueError(
            f"{context}: invalid fidelity values {bad_values}. "
            "Fidelity must be exactly 0 (low fidelity) or 1 (high fidelity), "
            "as defined in file_manifest.csv."
        )

    return numeric_values.astype(np.int32)


@dataclass
class CNPRuntimeConfig:
    config_path: Path
    version: str
    train_dir: Path
    predict_dirs: List[Path]
    predict_iterations: List[int]
    out_dir: Path
    n_shells: int
    theta_headers: List[str]
    phi_headers: List[str]
    target_headers: List[str]
    context_ratio: float
    context_mode: str
    training_mode: str
    epochs: int
    steps_per_epoch: int
    batch_size_train: int
    files_per_batch_train: int
    ratio_testing_vs_training: float
    plot_after: int
    seed: int


def _as_float_fraction(value: object, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if "/" in text:
            a, b = text.split("/", 1)
            return float(a) / float(b)
        return float(text)
    return default


def _resolve_path(path_value: str | Path, base: Path) -> Path:
    p = Path(path_value)
    return p if p.is_absolute() else (base / p).resolve()
    
def load_runtime_config(
    config_path: str | Path,
    seed: int = 42
) -> CNPRuntimeConfig:
    config_path = Path(config_path).resolve()
    raw = yaml.safe_load(config_path.read_text())

    cnp = raw.get("cnp_settings", {})
    sim = raw.get("simulation_settings", {})
    paths = raw.get("path_settings", {})
    base = config_path.parent

    predict_dirs = [_resolve_path(p, base) for p in paths.get("path_to_files_predict", [])]
    predict_iterations = [int(x) for x in paths.get("iteration", [0] * len(predict_dirs))]

    if len(predict_iterations) < len(predict_dirs):
        predict_iterations.extend([0] * (len(predict_dirs) - len(predict_iterations)))
    
    return CNPRuntimeConfig(
        config_path=config_path,
        version=str(paths.get("version", "v_clean")),
        train_dir=_resolve_path(paths["path_to_files_train"], base),
        predict_dirs=predict_dirs,
        predict_iterations=predict_iterations,
        out_dir=_resolve_path(paths.get("path_out_cnp", "../../data/out/cnp"), base),
        n_shells=int(sim.get("n_shells", 100)),
        theta_headers=list(sim.get("theta_headers", ["detector_R", "detector_Z"])),
        phi_headers=list(sim.get("phi_labels", ["s_r", "s_z_from_center"])),
        target_headers=list(sim.get("target_headers", ["target_shell"])),
        context_ratio=float(cnp.get("context_ratio", 1 / 3)),
        context_mode=str(cnp.get("context_mode", "random")).strip().lower(),
        training_mode=str(cnp.get("training_mode", "minibatch")).strip().lower(),
        epochs=int(cnp.get("training_epochs", 15)),
        steps_per_epoch=int(cnp.get("steps_per_epoch", 5000)),
        batch_size_train=int(cnp.get("batch_size_train", 4096)),
        files_per_batch_train=int(cnp.get("files_per_batch_train", 32)),
        ratio_testing_vs_training=_as_float_fraction(cnp.get("ratio_testing_vs_training", "1/40"), default=1 / 40),
        plot_after=int(cnp.get("plot_after", 1000)),
        seed=seed,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------------
# HDF5 data access
# -----------------------------


@dataclass
class EventBatch:
    x: torch.Tensor # concat(theta, phi), float32, shape (N, theta_dim + phi_dim)
    y: torch.Tensor # target_shell, int64, shape (N,)


class H5EventPool:
    """Event-level sampler for categorical shell classification.

    Each H5 file contains:
        theta:        (N, theta_dim)
        phi:          (N, phi_dim)
        target_shell: (N,) int64, zero-based shell class labels 0..n_shells-1

    The model I/O is:
        x = concat(theta, phi)
        y = target_shell
    """

    def __init__(
        self,
        directory: str | Path,
        theta_headers: Sequence[str],
        phi_headers: Sequence[str],
        target_headers: Sequence[str],
        n_shells: int,
        seed: int = 42,
        cache_files: bool = True,
    ) -> None:
        self.directory = Path(directory)
        self.theta_headers = list(theta_headers)
        self.phi_headers = list(phi_headers)
        self.target_headers = list(target_headers)
        self.n_shells = int(n_shells)
        self.rng = np.random.default_rng(seed)
        self.cache_files = cache_files

        self._cache: Dict[Path, Tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]] = {}
        self._row_count_cache: Dict[Path, int] = {}

        if not self.directory.exists():
            raise FileNotFoundError(f"H5 directory does not exist: {self.directory}")

        self.files = sorted([p for p in self.directory.rglob("*.h5") if p.is_file()])
        if not self.files:
            raise FileNotFoundError(f"No .h5 files found in {self.directory}")

    def _decode_labels(self, arr: np.ndarray) -> List[str]:
        return [
            item.decode("utf-8") if isinstance(item, (bytes, np.bytes_)) else str(item)
            for item in arr
        ]

    def _read_meta(self, f: h5py.File) -> dict[str, np.ndarray]:
        meta: dict[str, np.ndarray] = {}
        if "meta" not in f:
            return meta

        for key in f["meta"].keys():
            meta[key] = np.asarray(f["meta"][key])

        return meta

    def _load_one(self, file_path: Path) -> Tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        if self.cache_files and file_path in self._cache:
            return self._cache[file_path]

        with h5py.File(file_path, "r") as f:
            if "theta" not in f:
                raise ValueError(f"{file_path.name}: missing dataset 'theta'")
                
            if "phi" not in f:
                raise ValueError(f"{file_path.name}: missing dataset 'phi'")

            if "target_shell" not in f:
                raise ValueError(f"{file_path.name}: missing dataset 'target_shell'")

            theta = np.asarray(f["theta"], dtype=np.float32)
            phi = np.asarray(f["phi"], dtype=np.float32)
            target_shell = np.asarray(f["target_shell"], dtype=np.int64).reshape(-1)
            meta = self._read_meta(f)

            if "fidelity" not in meta:
                raise ValueError(
                    f"{file_path.name}: missing required meta/fidelity. "
                    "Fidelity must originate from file_manifest.csv during data preparation."
                )
            meta["fidelity"] = _validate_binary_fidelity_array(
                meta["fidelity"],
                n_events=len(target_shell),
                context=f"{file_path.name} meta/fidelity",
            )

            if theta.ndim != 2:
                raise ValueError(f"{file_path.name}: expected theta shape (N, theta_dim), got {theta.shape}")
            
            if phi.ndim != 2:
                raise ValueError(f"{file_path.name}: expected phi shape (N, phi_dim), got {phi.shape}")

            if len(theta) != len(phi) or len(phi) != len(target_shell):
                raise ValueError(
                    f"{file_path.name}: row count mismatch: "
                    f"theta={len(theta)}, phi={len(phi)}, target_shell={len(target_shell)}"
                )
    
            if theta.shape[1] != len(self.theta_headers):
                raise ValueError(
                    f"{file_path.name}: theta dim/header mismatch. "
                    f"Expected {len(self.theta_headers)} columns {self.theta_headers}, "
                    f"got shape {theta.shape}"
                )

            if phi.shape[1] != len(self.phi_headers):
                raise ValueError(
                    f"{file_path.name}: phi dim/header mismatch. "
                    f"Expected {len(self.phi_headers)} columns {self.phi_headers}, "
                    f"got shape {phi.shape}"
                )
            
            if len(target_shell) > 0:
                y_min = int(target_shell.min())
                y_max = int(target_shell.max())
                if y_min < 0 or y_max >= self.n_shells:
                    raise ValueError(
                        f"{file_path.name}: target_shell must be in [0, {self.n_shells - 1}], "
                        f"got min={y_min}, max={y_max}"
                    )
    
            if "theta_labels" in f:
                labels = self._decode_labels(np.asarray(f["theta_labels"]))
                if labels[: len(self.theta_headers)] != self.theta_headers:
                    raise ValueError(
                        f"Theta labels mismatch in {file_path.name}. "
                        f"Expected {self.theta_headers}, got {labels}"
                    )
                
            if "phi_labels" in f:
                labels = self._decode_labels(np.asarray(f["phi_labels"]))
                if labels[: len(self.phi_headers)] != self.phi_headers:
                    raise ValueError(
                        f"Phi labels mismatch in {file_path.name}. "
                        f"Expected {self.phi_headers}, got {labels}"
                    )

            if "target_headers" in f:
                labels = self._decode_labels(np.asarray(f["target_headers"]))
                if labels[: len(self.target_headers)] != self.target_headers:
                    raise ValueError(
                        f"Target headers mismatch in {file_path.name}. "
                        f"Expected {self.target_headers}, got {labels}"
                    )

            x = np.concatenate([theta, phi], axis=1).astype(np.float32)
        
        if self.cache_files:
            self._cache[file_path] = (x, target_shell, meta)

        return x, target_shell, meta

    def _count_rows_one(self, file_path: Path) -> int:
        if file_path in self._row_count_cache:
            return self._row_count_cache[file_path]

        with h5py.File(file_path, "r") as f:
            n_rows = int(f["target_shell"].shape[0])

        self._row_count_cache[file_path] = n_rows
        return n_rows

    def _choose_files(self, files_per_batch: int) -> List[Path]:
        k = min(files_per_batch, len(self.files))

        if k == len(self.files):
            return self.files

        idx = self.rng.choice(len(self.files), size=k, replace=False)
        return [self.files[i] for i in idx]

    def sample_batch(self, batch_size: int, files_per_batch: int) -> EventBatch:
        chosen = self._choose_files(files_per_batch)
        per_file = max(1, batch_size // len(chosen))

        xs: List[np.ndarray] = []
        ys: List[np.ndarray] = []

        for f in chosen:
            x, target_shell, _meta = self._load_one(f)
            n = len(target_shell)

            if n == 0:
                continue

            idx = self.rng.integers(0, n, size=per_file)

            xs.append(x[idx].astype(np.float32))
            ys.append(target_shell[idx].astype(np.int64))

        if not xs:
            raise RuntimeError("Could not sample non-empty batch from H5 files")

        x_arr = np.vstack(xs).astype(np.float32)
        y_arr = np.concatenate(ys).astype(np.int64)

        return EventBatch(
            x=torch.from_numpy(x_arr),
            y=torch.from_numpy(y_arr),
        )

    def iter_epoch_batches(
        self,
        batch_size: int,
        files_per_batch: int,
        *,
        shuffle: bool = True,
        drop_last: bool = False,
    ) -> Iterable[EventBatch]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        if files_per_batch <= 0:
            raise ValueError("files_per_batch must be positive.")

        file_order = list(self.files)

        if shuffle and len(file_order) > 1:
            perm = self.rng.permutation(len(file_order))
            file_order = [file_order[i] for i in perm]

        x_buffer: List[np.ndarray] = []
        y_buffer: List[np.ndarray] = []
        buffer_rows = 0

        def flush_batches(final: bool = False) -> Iterable[EventBatch]:
            nonlocal x_buffer, y_buffer, buffer_rows

            if buffer_rows == 0:
                return

            x_arr = np.vstack(x_buffer).astype(np.float32)
            y_arr = np.concatenate(y_buffer).astype(np.int64)

            if shuffle and len(x_arr) > 1:
                perm = self.rng.permutation(len(x_arr))
                x_arr = x_arr[perm]
                y_arr = y_arr[perm]

            n_full = len(x_arr) // batch_size

            for batch_idx in range(n_full):
                start = batch_idx * batch_size
                end = start + batch_size

                yield EventBatch(
                    x=torch.from_numpy(x_arr[start:end]),
                    y=torch.from_numpy(y_arr[start:end]),
                )

            used = n_full * batch_size
            remainder_x = x_arr[used:]
            remainder_y = y_arr[used:]

            if final and len(remainder_x) and not drop_last:
                yield EventBatch(
                    x=torch.from_numpy(remainder_x),
                    y=torch.from_numpy(remainder_y),
                )
                remainder_x = np.empty((0, x_arr.shape[1]), dtype=np.float32)
                remainder_y = np.empty((0,), dtype=np.int64)

            x_buffer = [remainder_x] if len(remainder_x) else []
            y_buffer = [remainder_y] if len(remainder_y) else []
            buffer_rows = int(len(remainder_x))

        for start in range(0, len(file_order), files_per_batch):
            file_group = file_order[start : start + files_per_batch]

            for f in file_group:
                x, target_shell, _meta = self._load_one(f)
                x = x.astype(np.float32)
                n = len(target_shell)

                if n == 0:
                    continue

                y = target_shell.astype(np.int64)

                if shuffle and n > 1:
                    perm = self.rng.permutation(n)
                    x = x[perm]
                    y = y[perm]

                x_buffer.append(x)
                y_buffer.append(y)
                buffer_rows += n

            yield from flush_batches(final=False)

        yield from flush_batches(final=True)

    def total_events(self) -> int:
        total = 0
        for f in self.files:
            total += self._count_rows_one(f)
        return total

    def iter_file_data(self) -> Iterable[Tuple[Path, np.ndarray, np.ndarray, dict[str, np.ndarray]]]:
        for f in self.files:
            x, target_shell, meta = self._load_one(f)
            yield f, x.astype(np.float32), target_shell.astype(np.int64), meta

# -----------------------------
# CNP model
# -----------------------------


class MLP(nn.Module):
    def __init__(self, sizes: Sequence[int], dropout: float = 0.0) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2:
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DeterministicCNP(nn.Module):
    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        repr_dim: int = 32,
        hidden: int = 128,
        dropout: float = 0.1,
        encoder_sizes: Optional[Sequence[int]] = None,
        decoder_sizes: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim

        # Legacy-compatible deep architecture (same shape style as old notebooks):
        # encoder: [d_in, 32, 64, 128, 128, 128, 64, 48, representation_size]
        # decoder: [representation_size + d_x, 32, 64, 128, 128, 128, 64, 48, d_out]
        # where d_out = y_dim * 2 (prediction mean/logit + uncertainty head)
        if encoder_sizes is None:
            encoder_sizes = [x_dim + y_dim, 32, 64, 128, 128, 128, 64, 48, repr_dim]
        if decoder_sizes is None:
            decoder_sizes = [x_dim + repr_dim, 32, 64, 128, 128, 128, 64, 48, y_dim * 2]

        self.encoder = MLP(encoder_sizes, dropout=dropout)
        self.decoder = MLP(decoder_sizes, dropout=dropout)

    def encode(self, context_x: torch.Tensor, context_y: torch.Tensor) -> torch.Tensor:
        h = torch.cat([context_x, context_y], dim=-1)
        r_i = self.encoder(h)
        return r_i.mean(dim=0, keepdim=True)  # [1, repr_dim]

    def forward(self, context_x: torch.Tensor, context_y: torch.Tensor, target_x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        r = self.encode(context_x, context_y)
        r_rep = r.expand(target_x.shape[0], -1)
        out = self.decoder(torch.cat([target_x, r_rep], dim=-1))
        logits = out[:, : self.y_dim]
        raw_sigma = out[:, self.y_dim :]
        sigma = F.softplus(raw_sigma) + 1e-6
        return logits, sigma

    @torch.no_grad()
    def predict_proba_mc(
        self,
        context_x: torch.Tensor,
        context_y: torch.Tensor,
        target_x: torch.Tensor,
        mc_samples: int = 30,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        was_training = self.training
        self.train()  # enable dropout for MC uncertainty
    
        preds = []
    
        for _ in range(mc_samples):
            logits, _sigma = self.forward(context_x, context_y, target_x)
            probs = F.softmax(logits, dim=-1)
            preds.append(probs)
    
        pred_stack = torch.stack(preds, dim=0)
    
        mean = pred_stack.mean(dim=0)
        std = pred_stack.std(dim=0, unbiased=False)
    
        if not was_training:
            self.eval()
    
        return mean, std


# -----------------------------
# Training and prediction
# -----------------------------


@dataclass
class TrainResult:
    model_path: Path
    history_csv: Path
    history_plot: Path
    sample_plot: Path

@dataclass
class PredictResult:
    all_path: Path
    best_path: Path
    mfgp_path: Path

def split_context_target_class(
    x: torch.Tensor,
    y: torch.Tensor,
    n_classes: int,
    context_ratio: float,
    rng: np.random.Generator,
    context_mode: str = "random",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    x: float tensor, shape (N, theta_dim + phi_dim)
    y: long tensor, shape (N,), class labels 0..n_classes-1

    Returns:
        context_x:       (N_context, theta_dim + phi_dim)
        context_y:       (N_context, n_classes)
        target_x:        (N, theta_dim + phi_dim)
        target_y:        (N,)
    """
    n = x.shape[0]
    if n < 4:
        raise ValueError("Batch too small; need at least 4 samples for context-target split")

    y=y.long()
    
    min_context = max(2, int(0.1 * n))
    max_context = max(min_context + 1, int(context_ratio * n))
    max_context = min(max_context, n - 1)

    if context_mode == "fixed":
        num_context = max(min_context, max_context)
    elif context_mode == "random":
        num_context = int(rng.integers(min_context, max_context + 1))
    else:
        raise ValueError(f"Unsupported context_mode={context_mode!r}. Expected 'random' or 'fixed'.")

    perm = rng.permutation(n)
    context_idx = torch.as_tensor(perm[:num_context], dtype=torch.long, device=x.device)

    context_x = x[context_idx]
    context_y_idx = y[context_idx]
    context_y = F.one_hot(context_y_idx, num_classes=n_classes).float()

    target_x = x
    target_y = y
    return context_x, context_y, target_x, target_y


def compute_shell_class_weights(
    pool: H5EventPool,
    n_shells: int,
    beta: float = 0.5,
    max_weight: float | None = 20.0,
) -> torch.Tensor:
    counts = np.zeros(n_shells, dtype=np.float64)

    for file_path in pool.files:
        with h5py.File(file_path, "r") as f:
            target_shell = np.asarray(f["target_shell"], dtype=np.int64).reshape(-1)
        counts += np.bincount(target_shell, minlength=n_shells)

    if np.any(counts == 0):
        missing = np.where(counts == 0)[0]
        print(f"[warn] Shell classes with zero training examples: {missing.tolist()}")

    safe_counts = np.maximum(counts, 1.0)

    weights = safe_counts ** (-beta)

    present = counts > 0

    if np.any(present):
        weights[present] = weights[present] / weights[present].mean()

    weights[~present] = 0.0

    if max_weight is not None:
        weights = np.clip(weights, 0.0, max_weight)

    print("\n[class weights]")
    print(f"min count: {counts[present].min() if np.any(present) else 0:.0f}")
    print(f"max count: {counts[present].max() if np.any(present) else 0:.0f}")
    print(f"min weight: {weights[present].min() if np.any(present) else 0:.4f}")
    print(f"max weight: {weights[present].max() if np.any(present) else 0:.4f}")

    return torch.tensor(weights, dtype=torch.float32)


def _class_probability_diagnostics(
    logits: torch.Tensor,
    true_shell: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert categorical logits into useful diagnostic quantities.

    Returns:
        p_true:       probability assigned to the true shell
        p_best_wrong: highest probability assigned to any wrong shell
        pred_shell:   predicted zero-based shell index
        true_rank:    rank of the true shell, where 1 means the true shell was top prediction
    """
    probs = F.softmax(logits.detach(), dim=-1).cpu().numpy()
    true_shell_np = true_shell.detach().cpu().numpy().astype(np.int64)

    rows = np.arange(len(true_shell_np))

    p_true = probs[rows, true_shell_np]

    wrong_probs = probs.copy()
    wrong_probs[rows, true_shell_np] = -np.inf
    p_best_wrong = np.max(wrong_probs, axis=1)

    pred_shell = np.argmax(probs, axis=1)

    # Rank 1 means true shell has the highest predicted probability.
    # Rank 2 means one shell had higher probability than the true shell, etc.
    true_rank = 1 + np.sum(probs > p_true[:, None], axis=1)

    return p_true, p_best_wrong, pred_shell, true_rank


def _plot_train_val_shell_probability_snapshot(
    train_logits: torch.Tensor,
    train_true_shell: torch.Tensor,
    val_logits: torch.Tensor,
    val_true_shell: torch.Tensor,
    out_path: Path,
    step: int,
    train_loss: float,
    val_loss: float,
    n_shells: int,
    max_wrong_samples: int = 200_000,
) -> None:
    """
    Categorical replacement for the old BCE inside/outside monitor plot.

    For each event:
        true shell probability      = softmax probability at the true class
        wrong shell probabilities   = softmax probabilities at all non-true classes
    """
    train_probs = F.softmax(train_logits.detach(), dim=-1).cpu().numpy()
    val_probs = F.softmax(val_logits.detach(), dim=-1).cpu().numpy()

    train_true = train_true_shell.detach().cpu().numpy().astype(np.int64)
    val_true = val_true_shell.detach().cpu().numpy().astype(np.int64)

    rng = np.random.default_rng(12345)

    def split_true_wrong(
        probs: np.ndarray,
        true_shell: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        n_events, n_classes = probs.shape
        rows = np.arange(n_events)

        true_scores = probs[rows, true_shell]

        wrong_mask = np.ones_like(probs, dtype=bool)
        wrong_mask[rows, true_shell] = False
        wrong_scores = probs[wrong_mask]

        # There are 99 wrong probabilities per event, so optionally downsample
        # them to keep the plot readable and fast.
        if len(wrong_scores) > max_wrong_samples:
            idx = rng.choice(len(wrong_scores), size=max_wrong_samples, replace=False)
            wrong_scores = wrong_scores[idx]

        return true_scores, wrong_scores

    train_true_scores, train_wrong_scores = split_true_wrong(train_probs, train_true)
    val_true_scores, val_wrong_scores = split_true_wrong(val_probs, val_true)

    train_pred = np.argmax(train_probs, axis=1)
    val_pred = np.argmax(val_probs, axis=1)

    train_acc = np.mean(train_pred == train_true)
    val_acc = np.mean(val_pred == val_true)

    train_mae = np.mean(np.abs(train_pred - train_true))
    val_mae = np.mean(np.abs(val_pred - val_true))

    random_prob = 1.0 / float(n_shells)

    bins = np.linspace(0.0, 1.0, 101)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle(f"Training Iteration {step}")

    # Training panel
    axes[0].hist(
        np.zeros_like(train_wrong_scores),
        bins=bins,
        alpha=0.9,
        label="true label (wrong shell)",
    )
    axes[0].hist(
        np.ones_like(train_true_scores),
        bins=bins,
        alpha=0.9,
        label="true label (true shell)",
    )
    axes[0].hist(
        train_wrong_scores,
        bins=bins,
        alpha=0.75,
        label="network score (wrong shell)",
    )
    axes[0].hist(
        train_true_scores,
        bins=bins,
        alpha=0.75,
        label="network score (true shell)",
    )
    axes[0].axvline(
        random_prob,
        linestyle="--",
        linewidth=1.0,
        label=f"random = {random_prob:.3f}",
    )
    axes[0].set_yscale("log")
    axes[0].set_xlabel(r"$P(\mathrm{shell})$")
    axes[0].set_ylabel("Count")
    axes[0].set_title(
        f"Training: Shell Probability Monitor "
        f"(CE {train_loss:.4f}, acc {train_acc:.3f}, MAE {train_mae:.2f})",
        fontsize=9,
    )
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    # Validation panel
    axes[1].hist(
        np.zeros_like(val_wrong_scores),
        bins=bins,
        alpha=0.9,
        label="true label (wrong shell)",
    )
    axes[1].hist(
        np.ones_like(val_true_scores),
        bins=bins,
        alpha=0.9,
        label="true label (true shell)",
    )
    axes[1].hist(
        val_wrong_scores,
        bins=bins,
        alpha=0.75,
        label="network score (wrong shell)",
    )
    axes[1].hist(
        val_true_scores,
        bins=bins,
        alpha=0.75,
        label="network score (true shell)",
    )
    axes[1].axvline(
        random_prob,
        linestyle="--",
        linewidth=1.0,
        label=f"random = {random_prob:.3f}",
    )
    axes[1].set_yscale("log")
    axes[1].set_xlabel(r"$P(\mathrm{shell})$")
    axes[1].set_ylabel("Count")
    axes[1].set_title(
        f"Validation: Shell Probability Monitor "
        f"(CE {val_loss:.4f}, acc {val_acc:.3f}, MAE {val_mae:.2f})",
        fontsize=9,
    )
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

def _plot_training_history(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    ax.plot(df["step"], df["train_loss"], label="train CE")
    ax.plot(df["step"], df["val_loss"], label="val CE")
    ax.set_xlabel("step")
    ax.set_ylabel("Cross entropy")
    ax.set_title("CNP categorical training history")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_sample_shell_predictions(
    pred_shell: np.ndarray,
    true_shell: np.ndarray,
    out_path: Path,
) -> None:
    pred_shell = np.asarray(pred_shell).reshape(-1)
    true_shell = np.asarray(true_shell).reshape(-1)

    fig, ax = plt.subplots(1, 1, figsize=(7, 6))

    ax.scatter(true_shell + 1, pred_shell + 1, s=8, alpha=0.4)

    lo = 1
    hi = max(int(true_shell.max()) + 1, int(pred_shell.max()) + 1)
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.0)

    ax.set_xlabel("true shell index")
    ax.set_ylabel("predicted shell index")
    ax.set_title("Sample Batch: Predicted vs True Shell")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def loss_function(
    logits: torch.Tensor,
    target: torch.Tensor,
    sigma: float=1.25,
    hard_fraction: float=0.8,
    class_weights: Optional[torch.Tensor]=None
) -> torch.Tensor:
    """
    Function that defines what the loss of the model is going to be. Made a separate function so its easy to change and shift.

    Current structure:
        One-Hot targetting added to Gaussian Blurred near-shell measurement

        sigma: width of the gaussian in units of shell index
        hard_fraction: how much of the total loss should be weighted toward one hot targetting? 
            1 -> normal hard cross entropy
            0 -> entirely gaussian smoothed
    """
    if sigma <= 0:
        raise ValueError("sigma must be greater than zero.")

    if not 0.0 <= hard_fraction <= 1.0:
        raise ValueError("hard_fraction must be between 0 and 1.")

    target = target.long().reshape(-1)
    n_shells = logits.shape[-1]
    
    shell_indices = torch.arange(n_shells, device=logits.device, dtype=logits.dtype)
    distances = shell_indices.unsqueeze(0) - target.to(logits.dtype).unsqueeze(1)
    gaussian_target = torch.exp(-0.5 * torch.square(distances/sigma))
    
    # Normalize the events gaussian sum to 1
    gaussian_target = gaussian_target/gaussian_target.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    hard_target = F.one_hot(target, num_classes=n_shells).to(dtype=logits.dtype)
    soft_target = hard_fraction * hard_target + (1.0-hard_fraction) * gaussian_target

    log_probabilities = F.log_softmax(logits, dim=-1)
    per_event_loss = -torch.sum(soft_target*log_probabilities, dim=-1)

    if class_weights is not None:
        # Weight each event based on its actual true shell
        sample_weights = class_weights[target]
        return torch.sum(sample_weights*per_event_loss) / sample_weights.sum().clamp_min(1e-12)

    return per_event_loss.mean()
    

def train_cnp(
    runtime: CNPRuntimeConfig,
    steps_per_epoch: Optional[int] = None,
    lr: float = 1e-4,
    weight_decay: float = 0.0,
    repr_dim: int = 32,
    hidden: int = 128,
    dropout: float = 0.1,
    monitor_every: Optional[int] = None,
    show_monitor_plots: bool = False,
    device: Optional[str] = None,
) -> TrainResult:
    runtime.out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(runtime.seed)

    pool = H5EventPool(
        runtime.train_dir,
        theta_headers=runtime.theta_headers,
        phi_headers=runtime.phi_headers,
        target_headers=runtime.target_headers,
        n_shells=runtime.n_shells,
        seed=runtime.seed,
        cache_files=True,
    )

    x_dim = len(runtime.theta_headers) + len(runtime.phi_headers)
    y_dim = runtime.n_shells

    model = DeterministicCNP(x_dim=x_dim, y_dim=y_dim, repr_dim=repr_dim, hidden=hidden, dropout=dropout)

    dev = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(dev)

    class_weights = compute_shell_class_weights(pool, n_shells=runtime.n_shells, beta=0.5, max_weight=20.0).to(dev)
    loss_kwargs = {
        "sigma": 1.25,
        "hard_fraction": 0.5,
        #"class_weights": class_weights
    }
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    rng = np.random.default_rng(runtime.seed)

    val_batch_size = max(128, int(runtime.batch_size_train * runtime.ratio_testing_vs_training))
    monitor_every = int(monitor_every if monitor_every is not None else runtime.plot_after)
    history_rows: List[Dict[str, float]] = []
    global_step = 0
    training_mode = runtime.training_mode
    effective_steps_per_epoch = int(steps_per_epoch if steps_per_epoch is not None else runtime.steps_per_epoch)

    if training_mode not in {"minibatch", "full_pass"}:
        raise ValueError(f"Unsupported training_mode={training_mode!r}. Expected 'minibatch' or 'full_pass'.")

    total_train_events = None
    approx_batches_per_epoch = None
    if training_mode == "full_pass":
        total_train_events = pool.total_events()
        approx_batches_per_epoch = int(np.ceil(total_train_events / runtime.batch_size_train))
        print(
            "\n[info] Full-pass dataloader mode is enabled. "
            f"Each epoch sees all {total_train_events} training events in ≈{approx_batches_per_epoch} batches."
        )
    else:
        print(
            "\n[info] Random mini-batch mode is enabled. "
            f"Each epoch uses {effective_steps_per_epoch} sampled steps."
        )
    
    for epoch in range(runtime.epochs):
        model.train()
        epoch_steps = 0
        
        if training_mode == "full_pass":
            batch_iter: Iterable[EventBatch] = pool.iter_epoch_batches(
                runtime.batch_size_train,
                runtime.files_per_batch_train,
                shuffle=True,
                drop_last=False,
            )
            epoch_total = approx_batches_per_epoch
        else:
            batch_iter = (
                pool.sample_batch(runtime.batch_size_train, runtime.files_per_batch_train)
                for _ in range(effective_steps_per_epoch)
            )
            epoch_total = effective_steps_per_epoch

        epoch_pbar = tqdm(
            batch_iter,
            total=epoch_total,
            desc=f"Epoch {epoch+1}/{runtime.epochs}",
            unit="batch",
            leave=True,
        )

        running_train_loss: list[float] = []
        running_val_loss: list[float] = []
        running_train_acc: list[float] = []
        running_val_acc: list[float] = []
        running_train_mae: list[float] = []
        running_val_mae: list[float] = []

        for batch in epoch_pbar:
            if batch.x.shape[0] < 4:
                continue
            x = batch.x.to(dev)
            y = batch.y.to(dev).long()

            cx, cy, tx, ty = split_context_target_class(
                x, y, 
                n_classes=runtime.n_shells, 
                context_ratio=runtime.context_ratio, 
                rng=rng, 
                context_mode=runtime.context_mode,
            )
            logits, _sigma = model(cx, cy, tx)
            train_loss = loss_function(logits, ty, **loss_kwargs)

            optimizer.zero_grad()
            train_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Validation step from an independent sampled mini-batch.
            with torch.no_grad():
                val_batch = pool.sample_batch(
                    val_batch_size,
                    max(1, runtime.files_per_batch_train // 2),
                )
            
                vx = val_batch.x.to(dev)
                vy = val_batch.y.to(dev).long()
            
                vcx, vcy, vtx, vty = split_context_target_class(
                    vx,
                    vy,
                    n_classes=runtime.n_shells,
                    context_ratio=runtime.context_ratio,
                    rng=rng,
                    context_mode=runtime.context_mode,
                )
            
                vlogits, _vsigma = model(vcx, vcy, vtx)
                val_loss = loss_function(vlogits, vty, **loss_kwargs)
            
                pred_shell = torch.argmax(logits, dim=-1)
                val_pred_shell = torch.argmax(vlogits, dim=-1)
            
                train_acc = (pred_shell == ty).float().mean()
                val_acc = (val_pred_shell == vty).float().mean()
            
                train_mae = (pred_shell.float() - ty.float()).abs().mean()
                val_mae = (val_pred_shell.float() - vty.float()).abs().mean()
            
            history_rows.append(
                {
                    "epoch": float(epoch),
                    "step": float(global_step),
                    "train_loss": float(train_loss.item()),
                    "val_loss": float(val_loss.item()),
                    "train_acc": float(train_acc.item()),
                    "val_acc": float(val_acc.item()),
                    "train_mae_shell": float(train_mae.item()),
                    "val_mae_shell": float(val_mae.item()),
                }
            )

            # Log data for epoch progress bar
            running_train_loss.append(float(train_loss.item()))
            running_val_loss.append(float(val_loss.item()))
            running_train_acc.append(float(train_acc.item()))
            running_val_acc.append(float(val_acc.item()))
            running_train_mae.append(float(train_mae.item()))
            running_val_mae.append(float(val_mae.item()))
            
            # Keep only recent values so the displayed average reacts during training.
            window = 100
            running_train_loss = running_train_loss[-window:]
            running_val_loss = running_val_loss[-window:]
            running_train_acc = running_train_acc[-window:]
            running_val_acc = running_val_acc[-window:]
            running_train_mae = running_train_mae[-window:]
            running_val_mae = running_val_mae[-window:]
            
            epoch_pbar.set_postfix(
                {
                    "train_loss": f"{np.mean(running_train_loss):.4f}",
                    "val_loss": f"{np.mean(running_val_loss):.4f}",
                    "train_acc": f"{np.mean(running_train_acc):.3f}",
                    "val_acc": f"{np.mean(running_val_acc):.3f}",
                    "train_mae": f"{np.mean(running_train_mae):.2f}",
                    "val_mae": f"{np.mean(running_val_mae):.2f}",
                }
            )

            if monitor_every > 0 and global_step % monitor_every == 0:
                ts = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            
                hist_df_live = pd.DataFrame(history_rows)
                history_csv_live = runtime.out_dir / f"cnp_{runtime.version}_history_{runtime.epochs}epochs.csv"
                hist_df_live.to_csv(history_csv_live, index=False)
            
                history_plot_live = runtime.out_dir / f"cnp_{runtime.version}_training_curve_{runtime.epochs}epochs.png"
                _plot_training_history(hist_df_live, history_plot_live)
                
                latest_plot = runtime.out_dir / f"cnp_{runtime.version}_class_monitor_latest.png"
                _plot_train_val_shell_probability_snapshot(
                    train_logits=logits,
                    train_true_shell=ty,
                    val_logits=vlogits,
                    val_true_shell=vty,
                    out_path=latest_plot,
                    step=global_step,
                    train_loss=float(train_loss.item()),
                    val_loss=float(val_loss.item()),
                    n_shells=runtime.n_shells,
                )

                if show_monitor_plots:
                    try: 
                        from IPython.display import Image, display
                        display(Image(filename=str(latest_plot)))
                    except Exception as e:
                        print(f"[warn] Could not display monitor plot inline: {e}")
                        print(f"[info] Monitor plot saved to {latest_plot}")
                        
            global_step += 1
            epoch_steps += 1

    # Save model and artifacts.
    model_path = runtime.out_dir / f"cnp_{runtime.version}_model_{runtime.epochs}epochs.pth"
    torch.save({
            "state_dict": model.state_dict(),
            "x_dim": x_dim,
            "y_dim": y_dim,
            "repr_dim": repr_dim,
            "hidden": hidden,
            "dropout": dropout,
            "encoder_sizes": [x_dim + y_dim, 32, 64, 128, 128, 128, 64, 48, repr_dim],
            "decoder_sizes": [x_dim + repr_dim, 32, 64, 128, 128, 128, 64, 48, y_dim * 2],
            "theta_headers": runtime.theta_headers,
            "phi_headers": runtime.phi_headers,
            "input_headers": runtime.theta_headers + runtime.phi_headers,
            "target_headers": runtime.target_headers,
            "n_shells": runtime.n_shells,
            "epochs": runtime.epochs,
            "training_mode": training_mode,
            "context_mode": runtime.context_mode,
            "version": runtime.version,
            "loss": "weighted_conditional_cross_entropy",
        }, model_path,
    )

    hist_df = pd.DataFrame(history_rows)
    history_csv = runtime.out_dir / f"cnp_{runtime.version}_history_{runtime.epochs}epochs.csv"
    hist_df.to_csv(history_csv, index=False)

    history_plot = runtime.out_dir / f"cnp_{runtime.version}_training_curve_{runtime.epochs}epochs.png"
    _plot_training_history(hist_df, history_plot)

    # One sample-batch qualitative prediction plot.
    model.eval()
    with torch.no_grad():
        sample = pool.sample_batch(
            min(4096, runtime.batch_size_train),
            runtime.files_per_batch_train,
        )
    
        sx = sample.x.to(dev)
        sy = sample.y.to(dev).long()
    
        scx, scy, stx, sty = split_context_target_class(
            sx,
            sy,
            n_classes=runtime.n_shells,
            context_ratio=runtime.context_ratio,
            rng=rng,
            context_mode=runtime.context_mode,
        )
    
        slogits, _ssigma = model(scx, scy, stx)
        pred_shell = torch.argmax(slogits, dim=-1).cpu().numpy()
        truth_shell = sty.cpu().numpy()
    
    sample_plot = runtime.out_dir / f"cnp_{runtime.version}_sample_predictions_{runtime.epochs}epochs.png"
    _plot_sample_shell_predictions(pred_shell, truth_shell, sample_plot)

    return TrainResult(
        model_path=model_path,
        history_csv=history_csv,
        history_plot=history_plot,
        sample_plot=sample_plot,
    )


def load_model_checkpoint(model_path: str | Path, device: Optional[str] = None) -> DeterministicCNP:
    model_path = Path(model_path)
    dev = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = torch.load(model_path, map_location=dev)

    model = DeterministicCNP(
        x_dim=int(ckpt["x_dim"]),
        y_dim=int(ckpt["y_dim"]),
        repr_dim=int(ckpt["repr_dim"]),
        hidden=int(ckpt["hidden"]),
        dropout=float(ckpt["dropout"]),
        encoder_sizes=ckpt.get("encoder_sizes"),
        decoder_sizes=ckpt.get("decoder_sizes"),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(dev)
    model.eval()
    return model


def predict_cnp(
    runtime: CNPRuntimeConfig,
    model_path: str | Path,
    mc_samples: int = 30,
    output_suffix: Optional[str] = "event_shell_distribution",
    output_epochs: Optional[int] = None,
    chunk_size: int = 20000,
    device: Optional[str] = None,
    all_shells: bool = False,
) -> PredictResult:
    runtime.out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(runtime.seed)

    if model_path is None:
        raise ValueError("Model Path must be provided when running prediction")
    model_path = Path(model_path)
    model = load_model_checkpoint(model_path, device=device)
    dev = next(model.parameters()).device

    if output_epochs is None:
        output_epochs = runtime.epochs
    if output_suffix is None:
        output_suffix = "event_shell_distribution"

    theta_dim = len(runtime.theta_headers)
    n_shells = runtime.n_shells
    shell_indices = np.arange(1, n_shells + 1, dtype=np.int32)

    # MFGP focused csv output
    mfgp_csv = (
        runtime.out_dir / f"cnp_{runtime.version}_{output_suffix}_{output_epochs}epochs.csv"
    )

    # Diagnostic csv outputs
    all_shell_csv = (
        runtime.out_dir / f"cnp_{runtime.version}_{output_suffix}_{output_epochs}epochs_all_shells.csv"
    )
    best_shell_csv = (
        runtime.out_dir
        / f"cnp_{runtime.version}_{output_suffix}_{output_epochs}epochs_best_shell.csv"
    )
    for path in [mfgp_csv, all_shell_csv, best_shell_csv]:
        if path.exists():
            path.unlink()

    write_all_header = True
    write_best_header = True
    total_event_count = 0

    agg_chunks: list[pd.DataFrame] = []

    def meta_numeric(
        meta: dict[str, np.ndarray],
        key: str,
        n: int,
        dtype: np.dtype | type,
        default: int | float,
    ) -> np.ndarray:
        if key not in meta:
            return np.full(n, default, dtype=dtype)

        arr = np.asarray(meta[key]).reshape(-1)

        if len(arr) != n:
            return np.full(n, default, dtype=dtype)

        return arr.astype(dtype)

    def required_fidelity(
        meta: dict[str, np.ndarray],
        n: int,
        *,
        file_path: Path,
    ) -> np.ndarray:
        """Read binary fidelity metadata without inventing or coercing a fallback."""
        if "fidelity" not in meta:
            raise ValueError(
                f"{file_path.name}: missing required meta/fidelity. "
                "The value must originate from file_manifest.csv during data preparation."
            )
        return _validate_binary_fidelity_array(
            meta["fidelity"],
            n_events=n,
            context=f"{file_path.name} meta/fidelity",
        )

    def meta_strings(
        meta: dict[str, np.ndarray],
        key: str,
        n: int,
        default: str,
    ) -> np.ndarray:
        if key not in meta:
            return np.asarray([default] * n, dtype=object)

        arr = np.asarray(meta[key]).reshape(-1)

        if len(arr) != n:
            return np.asarray([default] * n, dtype=object)

        return np.asarray(
            [
                item.decode("utf-8") if isinstance(item, (bytes, np.bytes_)) else str(item)
                for item in arr
            ],
            dtype=object,
        )

    print("\n" + "=" * 80)
    print("Starting CNP prediction")
    print(f"Save all-shell CSV: {all_shells}")
    print("=" * 80)

    for i, pred_dir in enumerate(runtime.predict_dirs):
        # Folder names identify only the data split. Fidelity is read from
        # each event's H5 metadata and ultimately originates in file_manifest.csv.
        pred_dir_str = str(pred_dir).lower()
        if "validation" in pred_dir_str:
            dataset_type = "VALIDATION"
        elif "training" in pred_dir_str or "train" in pred_dir_str:
            dataset_type = "TRAINING"
        else:
            dataset_type = pred_dir.name or "PREDICTION"

        iteration = runtime.predict_iterations[i]

        pool = H5EventPool(
            pred_dir,
            theta_headers=runtime.theta_headers,
            phi_headers=runtime.phi_headers,
            target_headers=runtime.target_headers,
            n_shells=runtime.n_shells,
            seed=runtime.seed + i,
            cache_files=False,
        )

        # Setup tqdm progress bar
        files = list(Path(pred_dir).rglob("*.h5"))

        with tqdm(total=len(files), desc=f"{dataset_type}", unit="block") as pbar:
            for file_path, x_np, target_shell_np, meta in pool.iter_file_data():
                n = len(target_shell_np)

                if n == 0:
                    pbar.update(1)
                    continue

                # Build context from sampled events in this H5 block.
                rng = np.random.default_rng(runtime.seed + i + n)
                n_context = max(2, int(runtime.context_ratio * n))
                n_context = min(n_context, n - 1)
                c_idx = rng.choice(n, size=n_context, replace=False)

                context_x = torch.from_numpy(x_np[c_idx]).to(dev)
                context_y_idx = torch.from_numpy(target_shell_np[c_idx]).long().to(dev)
                context_y = F.one_hot(context_y_idx, num_classes=n_shells).float()

                event_indices = meta_numeric(
                    meta,
                    "event_index",
                    n,
                    dtype=np.int64,
                    default=0,
                )
                if "event_index" not in meta:
                    event_indices = np.arange(n, dtype=np.int64)

                original_event_ids = meta_numeric(
                    meta,
                    "original_event_id",
                    n,
                    dtype=np.int64,
                    default=-1,
                )

                source_files = meta_strings(
                    meta,
                    "source_file",
                    n,
                    default=file_path.name,
                )

                fidelities = required_fidelity(
                    meta,
                    n,
                    file_path=file_path,
                )

                true_shell_one = target_shell_np.astype(np.int64) + 1

                events_per_chunk = max(1, chunk_size)

                for event_start in range(0, len(x_np), events_per_chunk):
                    event_end = min(len(x_np), event_start + events_per_chunk)

                    x_chunk = x_np[event_start:event_end]
                    theta_chunk = x_chunk[:, :theta_dim]
                    event_index_chunk = event_indices[event_start:event_end]
                    original_event_id_chunk = original_event_ids[event_start:event_end]
                    source_file_chunk = source_files[event_start:event_end]
                    fidelity_chunk = fidelities[event_start:event_end]
                    true_shell_one_chunk = true_shell_one[event_start:event_end]
                    n_events_chunk = len(x_chunk)

                    if n_events_chunk == 0:
                        continue

                    target_x = torch.from_numpy(x_chunk).to(dev)

                    with torch.no_grad():
                        prob_t, std_t = model.predict_proba_mc(
                            context_x,
                            context_y,
                            target_x,
                            mc_samples=mc_samples,
                        )

                    probs = prob_t.cpu().numpy()  # (N_events_chunk, n_shells)
                    stds = std_t.cpu().numpy()    # (N_events_chunk, n_shells)

                    tiled_shell_index = np.tile(shell_indices, reps=n_events_chunk)
                    repeated_true_shell = np.repeat(true_shell_one_chunk, repeats=n_shells)
                    flat_y_cnp = probs.reshape(-1)
                    flat_y_cnp_err = stds.reshape(-1)
                    flat_y_raw = (tiled_shell_index == repeated_true_shell).astype(np.float32)

                    if all_shells:
                        repeated_event_index = np.repeat(event_index_chunk, repeats=n_shells)
                        repeated_original_event_id = np.repeat(
                            original_event_id_chunk,
                            repeats=n_shells,
                        )
                        repeated_source_file = np.repeat(source_file_chunk, repeats=n_shells)
                        repeated_fidelity = np.repeat(
                            fidelity_chunk,
                            repeats=n_shells,
                        )

                        out = pd.DataFrame(
                            {
                                "iteration": float(iteration),
                                "fidelity": repeated_fidelity,
                                "source_file": repeated_source_file,
                                "event_index": repeated_event_index,
                                "original_event_id": repeated_original_event_id,
                                "shell_index": tiled_shell_index,
                                "true_shell_index": repeated_true_shell,
                                "y_cnp": flat_y_cnp,
                                "y_cnp_err": flat_y_cnp_err,
                            }
                        )

                        for theta_col, theta_name in enumerate(runtime.theta_headers):
                            out[theta_name] = np.repeat(
                                theta_chunk[:, theta_col],
                                repeats=n_shells,
                            )

                        out["y_raw"] = flat_y_raw

                        # Already normalized by softmax.
                        out["p_shell"] = out["y_cnp"]

                        out.to_csv(
                            all_shell_csv,
                            mode="w" if write_all_header else "a",
                            header=write_all_header,
                            index=False,
                        )
                        write_all_header = False

                        del out

                    best_zero = np.argmax(probs, axis=1)
                    best_one = best_zero + 1

                    best_chunk = pd.DataFrame(
                        {
                            "iteration": float(iteration),
                            "fidelity": fidelity_chunk,
                            "source_file": source_file_chunk,
                            "event_index": event_index_chunk,
                            "original_event_id": original_event_id_chunk,
                            "true_shell_index": true_shell_one_chunk,
                            "predicted_shell_index": best_one.astype(np.int32),
                            "predicted_shell_probability": probs[
                                np.arange(n_events_chunk),
                                best_zero,
                            ],
                            "predicted_shell_score": probs[
                                np.arange(n_events_chunk),
                                best_zero,
                            ],
                            "y_cnp_err": stds[np.arange(n_events_chunk), best_zero],
                        }
                    )

                    for theta_col, theta_name in enumerate(runtime.theta_headers):
                        best_chunk[theta_name] = theta_chunk[:, theta_col]

                    best_chunk.to_csv(
                        best_shell_csv,
                        mode="w" if write_best_header else "a",
                        header=write_best_header,
                        index=False,
                    )
                    write_best_header = False

                    # Build the smallest possible frame needed for MFGP aggregation.
                    # This keeps the MFGP output unchanged, but avoids materializing the
                    # large diagnostic all-shell DataFrame when all_shells=False.
                    agg_source = pd.DataFrame(
                        {
                            "iteration": float(iteration),
                            "fidelity": np.repeat(fidelity_chunk, repeats=n_shells),
                            "shell_index": tiled_shell_index,
                            "y_cnp": flat_y_cnp,
                            "y_cnp_err_sq": np.square(flat_y_cnp_err),
                            "y_raw": flat_y_raw,
                        }
                    )

                    for theta_col, theta_name in enumerate(runtime.theta_headers):
                        agg_source[theta_name] = np.repeat(
                            theta_chunk[:, theta_col],
                            repeats=n_shells,
                        )

                    if not agg_source.empty:
                        agg_chunk = (
                            agg_source.groupby(
                                ["iteration", "fidelity", *runtime.theta_headers, "shell_index"],
                                as_index=False,
                            )
                            .agg(
                                y_cnp_sum=("y_cnp", "sum"),
                                y_cnp_err_sq_sum=("y_cnp_err_sq", "sum"),
                                y_raw_sum=("y_raw", "sum"),
                                n_samples=("y_cnp", "size"),
                            )
                        )

                        agg_chunks.append(agg_chunk)

                    total_event_count += n_events_chunk

                    del best_chunk, agg_source, probs, stds, target_x
                    del tiled_shell_index, repeated_true_shell, flat_y_cnp, flat_y_cnp_err, flat_y_raw

                    if "agg_chunk" in locals():
                        del agg_chunk

                del context_x, context_y, x_np, target_shell_np

                pbar.update(1)
                pbar.set_postfix(
                    events=f"{total_event_count:,}",
                )

    required_cols = [
        "iteration",
        "fidelity",
        "n_samples",
        *runtime.theta_headers,
        "shell_index",
        "y_cnp",
        "y_cnp_err",
        "y_raw",
        "log_prop",
        "bce",
        "source_file",
    ]

    if agg_chunks:
        agg_all = pd.concat(agg_chunks, ignore_index=True)

        agg_final = (
            agg_all.groupby(
                ["iteration", "fidelity", *runtime.theta_headers, "shell_index"],
                as_index=False,
            )
            .agg(
                y_cnp_sum=("y_cnp_sum", "sum"),
                y_cnp_err_sq_sum=("y_cnp_err_sq_sum", "sum"),
                y_raw_sum=("y_raw_sum", "sum"),
                n_samples=("n_samples", "sum"),
            )
        )

        agg_final["y_cnp"] = agg_final["y_cnp_sum"] / agg_final["n_samples"]

        agg_final["y_cnp_err"] = np.sqrt(
            agg_final["y_cnp_err_sq_sum"] / agg_final["n_samples"]
        )

        agg_final["y_raw"] = agg_final["y_raw_sum"] / agg_final["n_samples"]

        eps = 1e-6
        p = np.clip(
            agg_final["y_cnp"].to_numpy(dtype=float),
            eps,
            1.0 - eps,
        )
        y = np.clip(
            agg_final["y_raw"].to_numpy(dtype=float),
            0.0,
            1.0,
        )

        agg_final["log_prop"] = y * np.log(p) + (1.0 - y) * np.log(1.0 - p)
        agg_final["bce"] = -agg_final["log_prop"]
        agg_final["source_file"] = "aggregated_event_shell_predictions"

        agg_final = agg_final[required_cols]
        agg_final.to_csv(mfgp_csv, index=False)

    else:
        pd.DataFrame(columns=required_cols).to_csv(mfgp_csv, index=False)

    print("\n" + "=" * 80)
    print(f"MFGP CSV:       {mfgp_csv}")
    if all_shells:
        print(f"All-shell CSV:  {all_shell_csv}")
    else:
        print("All-shell CSV:  not saved (all_shells=False)")
    print(f"Best-shell CSV: {best_shell_csv}")
    print("=" * 80)

    return PredictResult(
        all_path=all_shell_csv if all_shells else None,
        best_path=best_shell_csv,
        mfgp_path=mfgp_csv,
    )

# -----------------------------
# Experiment wrappers
# -----------------------------


def _experiment_result_dict(
    *,
    config_path: Path,
    validation_config_path: Path,
    train_result: TrainResult,
    predict_result_train: PredictResult,
    predict_result_validation: PredictResult,
) -> Dict[str, str]:
    return {
        "config_path": str(config_path),
        "validation_config_path": str(validation_config_path),

        "model_path": str(train_result.model_path),
        "history_csv": str(train_result.history_csv),
        "history_plot": str(train_result.history_plot),
        "sample_plot": str(train_result.sample_plot),

        "train_all_shell_csv": str(predict_result_train.all_path),
        "train_best_shell_csv": str(predict_result_train.best_path),
        "train_mfgp_csv": str(predict_result_train.mfgp_path),

        "validation_all_shell_csv": str(predict_result_validation.all_path),
        "validation_best_shell_csv": str(predict_result_validation.best_path),
        "validation_mfgp_csv": str(predict_result_validation.mfgp_path),
    }


def run_minibatch_experiment(
    *,
    seed: int = 42,
    device: Optional[str] = None,
) -> Dict[str, str]:
    config_path = (
        Path(__file__).resolve().parents[1]
        / "xlzd"
        / "settings_minibatch.yaml"
    ).resolve()

    validation_config_path = (
        Path(__file__).resolve().parents[1]
        / "xlzd"
        / "settings_validation_minibatch.yaml"
    ).resolve()

    runtime = load_runtime_config(config_path, seed=seed)
    validation_runtime = load_runtime_config(validation_config_path, seed=seed)

    train_result = train_cnp(runtime, device=device)

    predict_result_train = predict_cnp(
        runtime,
        model_path=train_result.model_path,
        chunk_size=20000,
        device=device,
    )

    predict_result_validation = predict_cnp(
        validation_runtime,
        model_path=train_result.model_path,
        chunk_size=20000,
        device=device,
    )

    result = _experiment_result_dict(
        config_path=config_path,
        validation_config_path=validation_config_path,
        train_result=train_result,
        predict_result_train=predict_result_train,
        predict_result_validation=predict_result_validation,
    )

    print(json.dumps(result, indent=2))
    return result


def run_fullpass_experiment(
    *,
    seed: int = 42,
    device: Optional[str] = None,
) -> Dict[str, str]:
    config_path = (
        Path(__file__).resolve().parents[1]
        / "xlzd"
        / "settings_fullpass.yaml"
    ).resolve()

    validation_config_path = (
        Path(__file__).resolve().parents[1]
        / "xlzd"
        / "settings_validation_fullpass.yaml"
    ).resolve()

    runtime = load_runtime_config(config_path, seed=seed)
    validation_runtime = load_runtime_config(validation_config_path, seed=seed)

    train_result = train_cnp(runtime, device=device)

    predict_result_train = predict_cnp(
        runtime,
        model_path=train_result.model_path,
        chunk_size=20000,
        device=device,
    )

    predict_result_validation = predict_cnp(
        validation_runtime,
        model_path=train_result.model_path,
        chunk_size=20000,
        device=device,
    )

    result = _experiment_result_dict(
        config_path=config_path,
        validation_config_path=validation_config_path,
        train_result=train_result,
        predict_result_train=predict_result_train,
        predict_result_validation=predict_result_validation,
    )

    print(json.dumps(result, indent=2))
    return result


def run_fixed_context_experiment(
    *,
    seed: int = 42,
    device: Optional[str] = None,
) -> Dict[str, str]:
    config_path = (
        Path(__file__).resolve().parents[1]
        / "xlzd"
        / "settings_fixedcontext.yaml"
    ).resolve()

    validation_config_path = (
        Path(__file__).resolve().parents[1]
        / "xlzd"
        / "settings_validation_fixedcontext.yaml"
    ).resolve()

    runtime = load_runtime_config(config_path, seed=seed)
    validation_runtime = load_runtime_config(validation_config_path, seed=seed)

    train_result = train_cnp(runtime, device=device)

    predict_result_train = predict_cnp(
        runtime,
        model_path=train_result.model_path,
        chunk_size=20000,
        device=device,
    )

    predict_result_validation = predict_cnp(
        validation_runtime,
        model_path=train_result.model_path,
        chunk_size=20000,
        device=device,
    )

    result = _experiment_result_dict(
        config_path=config_path,
        validation_config_path=validation_config_path,
        train_result=train_result,
        predict_result_train=predict_result_train,
        predict_result_validation=predict_result_validation,
    )

    print(json.dumps(result, indent=2))
    return result


# -----------------------------
# Optional CLI utilities
# -----------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Clean CNP train/predict pipeline (no resum dependency)")
    default_config = Path(__file__).resolve().parents[1] / "config" / "settings_shell_minibatch.yaml"
    p.add_argument("--config", type=Path, default=default_config, help="Path to settings YAML")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--device", type=str, default=None, help="Torch device (e.g., cpu, cuda)")

    sub = p.add_subparsers(dest="cmd", required=False)
    p.set_defaults(
        cmd="full",
        steps_per_epoch=None,
        lr=1e-4,
        weight_decay=0.0,
        repr_dim=32,
        hidden=128,
        dropout=0.1,
        monitor_every=None,
        show_monitor_plots=False,
        mc_samples=30,
        chunk_size=20000,
        output_suffix=None,
        output_epochs=None,
        model_path=None,
    )

    tr = sub.add_parser("train", help="Train CNP")
    tr.add_argument("--steps-per-epoch", type=int, default=None)
    tr.add_argument("--lr", type=float, default=1e-4)
    tr.add_argument("--weight-decay", type=float, default=0.0)
    tr.add_argument("--repr-dim", type=int, default=32)
    tr.add_argument("--hidden", type=int, default=128)
    tr.add_argument("--dropout", type=float, default=0.1)
    tr.add_argument("--monitor-every", type=int, default=None, help="Log and save monitor plots every N steps (default: settings plot_after)")
    tr.add_argument("--show-monitor-plots", action="store_true", help="Display monitor plots inline when running in notebooks/IPython")
    tr.set_defaults(
        steps_per_epoch=None,
        lr=1e-4,
        weight_decay=0.0,
        repr_dim=32,
        hidden=128,
        dropout=0.1,
        monitor_every=None,
        show_monitor_plots=False,
    )

    pr = sub.add_parser("predict", help="Run prediction/export")
    pr.add_argument("--model-path", type=Path, default=None)
    pr.add_argument("--mc-samples", type=int, default=30)
    pr.add_argument("--chunk-size", type=int, default=20000)
    pr.add_argument("--output-suffix", type=str, default=None)
    pr.add_argument("--output-epochs", type=int, default=None)
    pr.set_defaults(
        mc_samples=30,
        chunk_size=20000,
        output_suffix=None,
        output_epochs=None,
    )

    fu = sub.add_parser("full", help="Train then predict")
    fu.add_argument("--steps-per-epoch", type=int, default=None)
    fu.add_argument("--lr", type=float, default=1e-4)
    fu.add_argument("--weight-decay", type=float, default=0.0)
    fu.add_argument("--repr-dim", type=int, default=32)
    fu.add_argument("--hidden", type=int, default=128)
    fu.add_argument("--dropout", type=float, default=0.1)
    fu.add_argument("--monitor-every", type=int, default=None, help="Log and save monitor plots every N steps (default: settings plot_after)")
    fu.add_argument("--show-monitor-plots", action="store_true", help="Display monitor plots inline when running in notebooks/IPython")
    fu.add_argument("--mc-samples", type=int, default=30)
    fu.add_argument("--chunk-size", type=int, default=20000)
    fu.add_argument("--output-suffix", type=str, default=None)
    fu.add_argument("--output-epochs", type=int, default=None)
    fu.set_defaults(
        steps_per_epoch=None,
        lr=1e-4,
        weight_decay=0.0,
        repr_dim=32,
        hidden=128,
        dropout=0.1,
        monitor_every=None,
        show_monitor_plots=False,
        mc_samples=30,
        chunk_size=20000,
        output_suffix=None,
        output_epochs=None,
    )

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    runtime = load_runtime_config(args.config, seed=args.seed)

    if args.cmd == "train":
        result = train_cnp(
            runtime,
            steps_per_epoch=args.steps_per_epoch,
            lr=args.lr,
            weight_decay=args.weight_decay,
            repr_dim=args.repr_dim,
            hidden=args.hidden,
            dropout=args.dropout,
            monitor_every=args.monitor_every,
            show_monitor_plots=args.show_monitor_plots,
            device=args.device,
        )
        print(json.dumps({
            "model_path": str(result.model_path),
            "history_csv": str(result.history_csv),
            "history_plot": str(result.history_plot),
            "sample_plot": str(result.sample_plot),
        }, indent=2))
        return

    if args.cmd == "predict":
        result = predict_cnp(
            runtime,
            model_path=args.model_path,
            mc_samples=args.mc_samples,
            output_suffix=args.output_suffix,
            output_epochs=args.output_epochs,
            chunk_size=args.chunk_size,
            device=args.device,
        )

        print(json.dumps({
            "all_shell_csv": str(result.all_path),
            "best_shell_csv": str(result.best_path),
            "mfgp_csv": str(result.mfgp_path),
        }, indent=2))

        return

    if args.cmd == "full":
        train_result = train_cnp(
            runtime,
            steps_per_epoch=args.steps_per_epoch,
            lr=args.lr,
            weight_decay=args.weight_decay,
            repr_dim=args.repr_dim,
            hidden=args.hidden,
            dropout=args.dropout,
            monitor_every=args.monitor_every,
            show_monitor_plots=args.show_monitor_plots,
            device=args.device,
        )

        predict_result = predict_cnp(
            runtime,
            model_path=train_result.model_path,
            mc_samples=args.mc_samples,
            output_suffix=args.output_suffix,
            output_epochs=args.output_epochs,
            chunk_size=args.chunk_size,
            device=args.device,
        )

        print(json.dumps({
            "model_path": str(train_result.model_path),
            "history_csv": str(train_result.history_csv),
            "history_plot": str(train_result.history_plot),
            "sample_plot": str(train_result.sample_plot),

            "all_shell_csv": str(predict_result.all_path),
            "best_shell_csv": str(predict_result.best_path),
            "mfgp_csv": str(predict_result.mfgp_path),
        }, indent=2))


if __name__ == "__main__":
    main()
