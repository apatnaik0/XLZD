# Local Jitter Theta Augmentation

This experiment creates new LF-only theta points by applying small local perturbations to the original LF theta values.

Workflow:

1. Read the prepared LF block files from `outputs/training/lf`.
2. For each original LF block/theta, sample one or more nearby synthetic theta values.
3. Recompute `inside_theta` for that block under the synthetic theta.
4. Rerun the trained CNP on that synthetic-theta block.
5. Aggregate the event-level predictions into LF trial rows.
6. Write an augmented MF-GP training CSV that keeps the original LF rows and adds the synthetic-theta LF rows.

Run the augmentation step from the repo root:

```bash
python3 theta_augmentations/local_jitter/build_local_jitter_augmented_lf.py
```

Then open:

- `theta_augmentations/local_jitter/01_local_jitter_mfgp.ipynb`

and run it to compare baseline vs local-jitter MF-GP outputs across the same transform modes used elsewhere.
