"""Commands for running the CNP on a genearlized distribution"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

CNP_CHECKPOINT_FORMAT = "generic_categorical_cnp_v1"

# -----------------------------------------------------------------------------
# Dataclasses
# -----------------------------------------------------------------------------
@dataclass
class ClassificationBatch:
    x: torch.Tensor
    y: torch.Tensor

@dataclass
class TrainResult:
    model_path: Path
    history_csv: Path
    history_plot: Path
    monitor_plot: Optional[Path]

@dataclass
class PredictionResults:
    probabilities: np.ndarray
    uncertainty: np.ndarray
    predicted_class: np.ndarray

BatchProvider = Callable[[int], ClassificationBatch]
EpochProvider = Callable[[int], Iterable[ClassificationBatch]]
LossFunction = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _ensure_batch(batch: ClassificationBatch | tuple[object, object]) -> ClassificationBatch:
    """Takes in general sequence of (x,y) or ClassificationBatch and makes sure its formatted correctly"""
    # Confirm input is correct
    if isinstance(batch, ClassificationBatch):
        x, y, = batch.x, batch.y
    elif isinstance(batch, tuple) and len(batch) == 2:
        x, y = batch
    else:
        raise TypeError("Batch provider must return ClassificationBatch or an (x, y) tuple.")

    # Assign
    x = torch.as_tensor(x, dtype=torch.float32)
    y = torch.as_tensor(y)

    # Confirm correct shape
    if x.ndim != 2:
        raise ValueError(f"x must have shape (N, x_dim); got {tuple(x.shape)}")
    if y.ndim not in {1, 2}:
        raise ValueError(
            "y must be either integer labels with shape (N,) or class distributions "
            f"with shape (N, n_classes); got {tuple(y.shape)}"
        )
    if len(x) != len(y):
        raise ValueError(f"x/y row mismatch: {len(x)} vs {len(y)}")

    return ClassificationBatch(x=x, y=y)

def target_distribution(y: torch.Tensor, n_classes: int) -> torch.Tensor:
    """Convert labels (hard or soft) to normalized class distributions"""
    y = torch.as_tensor(y)

    # If you have hard labels, convert to a one-hot distribution
    if y.ndim == 1:
        labels = y.long()
        if len(labels):
            y_min = int(labels.min().item())
            y_max = int(labels.max().item())
            if y_min < 0 or y_max >= n_classes:
                raise ValueError(f"Class labels must be in [0, {n_classes - 1}], got min={y_min}, max={y_max}")
        return F.one_hot(labels, num_classes=n_classes).float()

    # If soft target, confirm right shape (N, n_classes)
    if y.ndim != 2 or y.shape[1] != n_classes:
        raise ValueError(f"Soft targets must have shape (N, {n_classes}); got {tuple(y.shape)}")
    
    # Convert soft labels
    dist = y.float()
    if not torch.isfinite(dist).all():
        raise ValueError("Soft target distributions contain NaN or infinite values")
    if torch.any(dist < 0):
        raise ValueError("Soft target distribution must be non-negative")

    row_sum = dist.sum(dim=-1, keepdim=True)
    if torch.any(row_sum <= 0):
        raise ValueError("Each soft target distribution must have total positive count")

    return dist / row_sum

def default_classification_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Cross Entropy for either hard labels or externally prepared soft targets"""
    if target.ndim == 1:
        return F.cross_entropy(logits, target.long())

    dist = target_distribution(target, logits.shape[-1]).to(device=logits.device, dtype=logits.dtype)
    log_probs = F.log_softmax(logits, dim=-1)
    return -(dist*log_probs).sum(dim=-1).mean()

# -----------------------------------------------------------------------------
# CNP model
# -----------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, sizes: Sequence[int], dropout: float = 0.0) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i+1]))
            if i < len(sizes) - 2:
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class DeterministicCNP(nn.Module):
    """Deterministic CNP whos output is a categorical prob. distribution"""
    def __init__(
        self,
        x_dim: int,
        n_classes: int,
        repr_dim: int = 32,
        hidden: int = 128,
        dropout: float = 0.1,
        encoder_sizes: Optional[Sequence[int]] = None,
        decoder_sizes: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__()
        self.x_dim = int(x_dim)
        self.n_classes = int(n_classes)

        if encoder_sizes is None:
            encoder_sizes = [x_dim + n_classes, 32, 64, hidden, hidden, hidden, 64, 48, repr_dim]
        if decoder_sizes is None:
            decoder_sizes = [x_dim + repr_dim, 32, 64, hidden, hidden, hidden, 64, 48, n_classes*2]

        self.encoder_sizes = list(encoder_sizes)
        self.decoder_sizes = list(decoder_sizes)
        self.encoder = MLP(self.encoder_sizes, dropout=dropout)
        self.decoder = MLP(self.decoder_sizes, dropout=dropout)

    def encode(self, context_x: torch.Tensor, context_y: torch.Tensor) -> torch.Tensor:
        h = torch.cat([context_x, context_y], dim=-1)
        r_i = self.encoder(h)
        return r_i.mean(dim=0, keepdim=True)

    def forward(
        self,
        context_x: torch.Tensor,
        context_y: torch.Tensor,
        target_x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Setup and run model
        r = self.encode(context_x, context_y)
        r_rep = r.expand(target_x.shape[0], -1)
        out = self.decoder(torch.cat([target_x, r_rep], dim=-1))

        # Model outputs
        logits = out[:, : self.n_classes]
        raw_sigma = out[:, self.n_classes :]
        sigma = F.softplus(raw_sigma) + 1e-6
        return logits, sigma
        
    @torch.no_grad()
    def predict_proba_mc(
        self,
        context_x: torch.Tensor,
        context_y: torch.Tensor,
        target_x: torch.Tensor,
        mc_samples: int=30,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """MC-dropout mean/std of the categorical probability vector"""
        if mc_samples <= 0:
            raise ValueError("MC samples must be positive")

        was_training = self.training
        self.train() # Enable dropout

        predictions = []
        for _ in range(mc_samples):
            logits, _sigma = self.forward(context_x, context_y, target_x)
            predictions.append(F.softmax(logits, dim=-1))
        pred_stack = torch.stack(predictions, dim=0)
        mean = pred_stack.mean(dim=0)
        std = pred_stack.std(dim=0, unbiased=False)

        if not was_training:
            self.eval()
        return mean, std

# -----------------------------------------------------------------------------
# Context/target handling
# -----------------------------------------------------------------------------
def split_context_target(
    x: torch.Tensor,
    y: torch.Tensor,
    n_classes: int,
    context_ratio: float,
    rng: np.random.Generator,
    context_mode: str = "random",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split a generic classification batch into CNP context and target sets"""
    n = x.shape[0]
    if n < 4:
        raise ValueError("Batch too small. Need atleast 4 samples for context-target split")
    if not 0 < context_ratio < 1:
        raise ValueError("Context Ratio must be between 0 and 1")

    min_context = max(2, int(0.1*n))
    max_context = max(min_context, int(context_ratio*n))
    max_context = min(max_context, n-1)

    # Choose context based off mode
    if context_mode == "fixed":
        num_context = max_context
    elif context_mode == "random":
        num_context = int(rng.integers(min_context, max_context+1))
    else:
        raise ValueError(f"Unsupported context mode {context_mode!r}. Expected 'random' or 'fixed'")

    # Choose the actual context
    perm = rng.permutation(n)
    context_idx = torch.as_tensor(perm[:num_context], dtype=torch.long, device=x.device)
    context_x = x[context_idx]
    context_y = target_distribution(y[context_idx], n_classes).to(x.device)

    target_x = x
    target_y = y
    return context_x, context_y, target_x, target_y

# -----------------------------------------------------------------------------
# Diagnostics
# -----------------------------------------------------------------------------
def _classification_metrics(
    logits: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, float]:
    n_classes = logits.shape[-1]
    target_dist = target_distribution(target, n_classes).to(logits.device)
    target_mode = target_dist.argmax(dim=-1)
    probs = F.softmax(logits, dim=-1)
    pred = probs.argmax(dim=-1)

    accuracy = (pred == target_mode).float().mean().item()
    target_probability = (probs*target_dist).sum(dim=-1).mean().item()
    entropy = (-(probs*torch.log(probs.clamp_min(1e-12))).sum(dim=-1)).mean().item()

    return {
        "accuracy": float(accuracy),
        "target_probability": float(target_probability),
        "predictive_entropy": float(entropy),
    }

def _plot_training_history(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["step"], df["train_loss"], label="train loss")
    ax.plot(df["step"], df["val_loss"], label="validation loss")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title("CNP classification training history")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def _plot_probability_monitor(
    train_logits: torch.Tensor,
    train_target: torch.Tensor,
    val_logits: torch.Tensor,
    val_target: torch.Tensor,
    out_path: Path,
    step: int,
    train_loss: float | None = None,
    val_loss: float | None = None,
    max_wrong_samples: int = 200_000,
) -> None:

    train_probs = F.softmax(train_logits.detach(), dim=-1).cpu().numpy()

    val_probs = F.softmax(val_logits.detach(), dim=-1).cpu().numpy()

    train_true = train_target.detach().cpu().numpy().astype(np.int64)
    val_true = val_target.detach().cpu().numpy().astype(np.int64)
    n_classes = train_probs.shape[1]
    rng = np.random.default_rng(12345)

    def signal_background(probs, true_class):
        n_events = len(true_class)
        rows = np.arange(n_events)

        # Probability assigned to correct shell
        signal = probs[rows, true_class]

        # Probabilities assigned to every incorrect shell
        background_mask = np.ones_like(probs, dtype=bool)
        background_mask[rows, true_class] = False

        background = probs[background_mask]

        # Avoid plotting millions of background entries
        if len(background) > max_wrong_samples:
            indices = rng.choice(
                len(background),
                size=max_wrong_samples,
                replace=False,
            )
            background = background[indices]

        return signal, background


    train_signal, train_background = signal_background(train_probs, train_true)
    val_signal, val_background = signal_background(val_probs, val_true)

    # Useful diagnostics
    train_pred = np.argmax(train_probs, axis=1)
    val_pred = np.argmax(val_probs, axis=1)

    train_acc = np.mean(train_pred == train_true)
    val_acc = np.mean(val_pred == val_true)

    train_mae = np.mean(np.abs(train_pred - train_true))
    val_mae = np.mean(np.abs(val_pred - val_true))

    random_probability = 1.0 / n_classes

    bins = np.linspace(0, 1, 101)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle(f"Training Iteration {step}")
    axes[0].hist(train_background, bins=bins, alpha=0.75, label="background (wrong shells)")
    axes[0].hist( train_signal, bins=bins, alpha=0.75, label="signal (true shell)")
    axes[0].axvline(random_probability, linestyle="--", linewidth=1, label=f"random = {random_probability:.3f}")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Predicted probability")
    axes[0].set_ylabel("Count")
    title = f"Training: Signal vs Background\nacc={train_acc:.3f}, MAE={train_mae:.2f}"
    if train_loss is not None:
        title += f", loss={train_loss:.4f}"
    axes[0].set_title(title)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(val_background, bins=bins, alpha=0.75, label="background (wrong shells)")
    axes[1].hist(val_signal, bins=bins, alpha=0.75, label="signal (true shell)")

    axes[1].axvline(random_probability, linestyle="--", linewidth=1, label=f"random = {random_probability:.3f}")

    axes[1].set_yscale("log")
    axes[1].set_xlabel("Predicted probability")
    axes[1].set_ylabel("Count")

    title = (f"Validation: Signal vs Background\nacc={val_acc:.3f}, MAE={val_mae:.2f}")
    if val_loss is not None:
        title += f", loss={val_loss:.4f}"
    axes[1].set_title(title)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
def train_cnp(
    *,
    train_batch_fn: BatchProvider,
    validation_batch_fn: Optional[BatchProvider] = None,
    epoch_batches_fn: Optional[EpochProvider] = None,
    inference_context_fn: Optional[BatchProvider] = None,
    loss_fn: Optional[LossFunction] = None,
    n_classes: int,
    out_dir: str | Path,
    version: str = "default",
    epochs: int = 15,
    steps_per_epoch: int = 5000,
    batch_size: int = 4096,
    validation_batch_size: Optional[int] = None,
    context_ratio: float = 1.0/3.0,
    context_mode: str = "random",
    inference_context_size: int = 4096,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.0,
    repr_dim: int = 32,
    hidden: int = 128,
    dropout: float = 0.1,
    monitor_every: int = 5000,
    seed: int = 42,
    device: Optional[str] = None,
    input_names: Optional[Sequence[str]] = None,
    class_names: Optional[Sequence[str]] = None,
    checkpoint_metadata: Optional[dict] = None,
) -> TrainResult:
    """
    Train the Categorical CNP on externally supplied data
    Parameters carrying physical meaning are defined externally

    train_batch_fn(batch_size)
        Return a random/sampled training ClassificationBatch
    validation_batch_fn(batch_size)
        Optional independent validation provider.
        Defaults to train_batch_fn
    epoch_batches_fn(batch_size)
        Optioanl full-pass provider. When supplied, it defines all batches for each epochs and steps_per_epoch is ignored
    inference_context_fn(context_size)
        Optional provider used to construct the fixed context embedded in the checkpoint
        Defaults to train_batch_fn
    loss_fn(logits, target)
        Optional external loss
        Defaults to hard/soft categorical cross entropy
    """
    # Checks
    if n_classes < 2:
        raise ValueError("n_classes must be atleast 2")
    if epochs <= 0:
        raise ValueError("Epochs must be positive")
    if batch_size < 4:
        raise ValueError("Batch size must be atleast 4")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)

    # Set functions and sizes
    validation_batch_fn = validation_batch_fn or train_batch_fn
    inference_context_fn = inference_context_fn or train_batch_fn
    loss_fn = loss_fn or default_classification_loss
    validation_batch_size = int(validation_batch_size or max(128, batch_size // 40))

    # Ask each provider for a single batch to validate sizes
    initial_batch = _ensure_batch(train_batch_fn(batch_size))
    if len(initial_batch.x) < 4:
        raise ValueError("Training provider returned fewer than 4 samples")
    x_dim = int(initial_batch.x.shape[1])

    target_distribution(initial_batch.y, n_classes)
    if input_names is not None and len(input_names) != x_dim:
        raise ValueError(f"Input names has {len(input_names)} entries but x_dim={x_dim}")
    if class_names is not None and len(class_names) != n_classes:
        raise ValueError(f"Class names has {len(class_names)} entries but n_classes={n_classes}")

    model = DeterministicCNP(
        x_dim=x_dim,
        n_classes=n_classes,
        repr_dim=repr_dim,
        hidden=hidden,
        dropout=dropout,
    )

    dev = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    rng = np.random.default_rng(seed)

    history_rows: list[dict[str, float]] = []
    global_step = 0
    latest_monitor: Optional[Path] = None
    for epoch in range(epochs):
        model.train()
        if epoch_batches_fn is None:
            # Make sure to reuse the already sampled epoch 0 or else its discarded
            def sampled_batches() -> Iterable[ClassificationBatch]:
                if epoch == 0:
                    yield initial_batch
                    start=1
                else:
                    start=0
                for _ in range(start, steps_per_epoch):
                    yield _ensure_batch(train_batch_fn(batch_size))

            batch_iter = sampled_batches()
            total = steps_per_epoch
        else:
            batch_iter = (_ensure_batch(b) for b in epoch_batches_fn(batch_size))
            total = None

        pbar = tqdm(batch_iter, total=total, desc=f"Epoch {epoch+1}/{epochs}", unit="Batch")
        for batch in pbar:
            if len(batch.x) < 4:
                continue
            if batch.x.shape[1] != x_dim:
                raise ValueError(f"Training provider changed x_dim from {x_dim} to {batch.x.shape[1]}")

            x = batch.x.to(dev, dtype=torch.float32)
            y = batch.y.to(dev)
            target_distribution(y, n_classes)

            cx, cy, tx, ty = split_context_target(
                x, y, n_classes = n_classes,
                context_ratio = context_ratio, 
                rng = rng,
                context_mode = context_mode,
            )

            logits, _sigma = model(cx, cy, tx)
            train_loss = loss_fn(logits, ty)
            if not torch.is_tensor(train_loss) or train_loss.ndim != 0:
                raise ValueError("Loss Function must return one scalar torch.Tensor")
            
            optimizer.zero_grad()
            train_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Validation steps
            with torch.no_grad():
                val_batch = _ensure_batch(validation_batch_fn(validation_batch_size))
                if len(val_batch.x) < 4:
                    raise ValueError("Validation provider returned fewer than 4 samples")
                if val_batch.x.shape[1] != x_dim:
                    raise ValueError(f"Validation provider x_dim={val_batch.x.shape[1]} does not match {x_dim}")

                vx = val_batch.x.to(dev, dtype=torch.float32)
                vy = val_batch.y.to(dev)
                target_distribution(vy, n_classes)

                vcx, vcy, vtx, vty = split_context_target(
                    vx, vy, n_classes=n_classes,
                    context_ratio = context_ratio,
                    rng=rng,
                    context_mode = context_mode,
                )
                vlogits, _vsigma = model(vcx, vcy, vtx)
                val_loss = loss_fn(vlogits, vty)

                train_metrics = _classification_metrics(logits, ty)
                val_metrics = _classification_metrics(vlogits, vty)

            history_rows.append(
                {
                    "epoch": float(epoch),
                    "step": float(global_step),
                    "train_loss": float(train_loss.item()),
                    "val_loss": float(val_loss.item()),
                    "train_accuracy": train_metrics["accuracy"],
                    "val_accuracy": val_metrics["accuracy"],
                    "train_target_probability": train_metrics["target_probability"],
                    "val_target_probability": val_metrics["target_probability"],
                    "train_predictive_entropy": train_metrics["predictive_entropy"],
                    "val_predictive_entropy": val_metrics["predictive_entropy"],
                }
            )

            # End of step management. Update progress bar, plot, and update step
            pbar.set_postfix(
                train_loss=f"{train_loss.item():.4f}",
                val_loss=f"{val_loss.item():.4f}",
                train_acc=f"{train_metrics['accuracy']:.3f}",
                val_acc=f"{val_metrics['accuracy']:.3f}",
            )
            if monitor_every > 0 and global_step % monitor_every == 0:
                hist_df_live = pd.DataFrame(history_rows)
                history_csv_live = out_dir / f"cnp_{version}_history.csv"
                history_plot_live = out_dir / f"cnp_{version}_training_curve.png"
                latest_monitor = out_dir / f"cnp_{version}_class_monitor_latest.png"

                hist_df_live.to_csv(history_csv_live, index=False)
                _plot_training_history(hist_df_live, history_plot_live)
                _plot_probability_monitor(
                    train_logits=logits,
                    train_target=ty,
                    val_logits=vlogits,
                    val_target=vty,
                    out_path=latest_monitor,
                    step=global_step,
                )

                try:
                    from IPython.display import Image, display
                    display(Image(filename=str(latest_monitor)))
                except Exception as e:
                    print(f"[warn] Could not display monitor plot inline")
                
            global_step += 1

    # Keep the context to save with the model
    context_batch = _ensure_batch(inference_context_fn(inference_context_size))
    if len(context_batch.x) < 2:
        raise ValueError("Inference context provider returned fewer than 2 samples.")
    if context_batch.x.shape[1] != x_dim:
        raise ValueError(f"Inference context x_dim={context_batch.x.shape[1]} does not match {x_dim}")     
    inference_context_x = context_batch.x.detach().cpu().to(torch.float32).contiguous()
    inference_context_y = target_distribution(context_batch.y, n_classes).detach().cpu().contiguous()

    # Save model
    model_path = out_dir / f"cnp_{version}_model.pth"
    torch.save({
        "checkpoint_format": CNP_CHECKPOINT_FORMAT,
        "state_dict": model.state_dict(),
        "x_dim": x_dim,
        "n_classes": n_classes,
        "y_dim": n_classes,  # compatibility/convenience
        "repr_dim": repr_dim,
        "hidden": hidden,
        "dropout": dropout,
        "encoder_sizes": model.encoder_sizes,
        "decoder_sizes": model.decoder_sizes,
        "epochs": epochs,
        "context_mode": context_mode,
        "version": version,
        "input_names": list(input_names) if input_names is not None else None,
        "class_names": list(class_names) if class_names is not None else None,
        "inference_context_x": inference_context_x,
        "inference_context_y": inference_context_y,
        "metadata": dict(checkpoint_metadata or {}),
    },
    model_path,)

    history_df = pd.DataFrame(history_rows)
    history_csv = out_dir / f"cnp_{version}_history.csv"
    history_plot = out_dir / f"cnp_{version}_training_curve.png"
    history_df.to_csv(history_csv, index=False)
    _plot_training_history(history_df, history_plot)

    return TrainResult(
        model_path=model_path,
        history_csv=history_csv,
        history_plot=history_plot,
        monitor_plot=latest_monitor,
    )
        
# -----------------------------------------------------------------------------
# Checkpoint loading
# -----------------------------------------------------------------------------
def load_cnp_checkpoint(
    model_path: str | Path,
    device: Optional[str] = None,
) -> tuple[DeterministicCNP, dict]:
    model_path = Path(model_path)
    dev = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = torch.load(model_path, map_location=dev)

    required = {
        "state_dict",
        "x_dim",
        "n_classes",
        "repr_dim",
        "hidden",
        "dropout",
    }
    missing = sorted(required - set(ckpt))
    if missing:
        raise ValueError(f"CNP checkpoint {model_path} missing fields {missing}")

    model = DeterministicCNP(
        x_dim=int(ckpt["x_dim"]),
        n_classes=int(ckpt["n_classes"]),
        repr_dim=int(ckpt["repr_dim"]),
        hidden=int(ckpt["hidden"]),
        dropout=float(ckpt["dropout"]),
        encoder_sizes=ckpt.get("encoder_sizes"),
        decoder_sizes=ckpt.get("decoder_sizes"),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(dev)
    model.eval()
    return model, ckpt

def load_cnp_inference_checkpoint(
    model_path: str | Path,
    device: Optional[str] = None,
) -> tuple[DeterministicCNP, torch.Tensor, torch.Tensor, dict]:
    model, ckpt = load_cnp_checkpoint(model_path, device=device)
    dev = next(model.parameters()).device
    
    required = {"inference_context_x", "inference_context_y"}
    missing = sorted(required - set(ckpt))
    if missing:
        raise ValueError(f"CNP checkpoint {model_path} has no portable inference context: {missing}")

    context_x = torch.as_tensor(ckpt["inference_context_x"], dtype=torch.float32, device=dev)
    context_y = torch.as_tensor(ckpt["inference_context_y"], dtype=torch.float32, device=dev)

    if context_x.ndim != 2 or context_x.shape[1] != int(ckpt["x_dim"]):
        raise ValueError("Saved context_x is incompatible with checkpoint x_dim")
    if context_y.ndim != 2 or context_y.shape[1] != int(ckpt["n_classes"]):
        raise ValueError("Saved context_y is incompatible with checkpoint n_classes")
    if len(context_x) != len(context_y):
        raise ValueError("Saved context x/y lengths do not match")

    return model, context_x, context_y, ckpt

# -----------------------------------------------------------------------------
# Prediction
# -----------------------------------------------------------------------------
def predict_distribution(
    *,
    model_path: str | Path,
    x: np.ndarray | torch.Tensor,
    mc_samples: int = 30,
    chunk_size: int = 20000,
    device: Optional[str] = None,
) -> PredictionResults:
    """Return only generic categorical predictions - interpretation is external"""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    # Load the model
    model, context_x, context_y, ckpt = load_cnp_inference_checkpoint(
        model_path,
        device=device,
    )
    dev = next(model.parameters()).device

    x_tensor = torch.as_tensor(x, dtype=torch.float32)
    if x_tensor.ndim != 2:
        raise ValueError(f"x must have shape (N, x_dim); got {tuple(x_tensor.shape)}")
    if x_tensor.shape[1] != int(ckpt["x_dim"]):
        raise ValueError(f"Prediction x_dim={x_tensor.shape[1]} does not match model x_dim={ckpt['x_dim']}")

    # Do the prediction
    probability_chunks: list[np.ndarray] = []
    uncertainty_chunks: list[np.ndarray] = []
    for start in range(0, len(x_tensor), chunk_size):
        target_x = x_tensor[start : start + chunk_size].to(dev)
        mean, std = model.predict_proba_mc(
            context_x, 
            context_y,
            target_x,
            mc_samples=mc_samples,
        )
        probability_chunks.append(mean.cpu().numpy())
        uncertainty_chunks.append(std.cpu().numpy())

    n_classes = int(ckpt["n_classes"])
    if probability_chunks:
        probabilities = np.concatenate(probability_chunks, axis=0)
        uncertainty = np.concatenate(uncertainty_chunks, axis=0)
    else:
        probabilities = np.empty((0, n_classes), dtype=np.float32)
        uncertainty = np.empty((0, n_classes), dtype=np.float32)

    predicted_class = np.argmax(probabilities, axis=1) if len(probabilities) else np.empty(0, dtype=np.int64)
    return PredictionResults(
        probabilities=probabilities,
        uncertainty=uncertainty,
        predicted_class=predicted_class,
    )