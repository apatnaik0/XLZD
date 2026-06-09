# Midpoint Theta Augmentation

This experiment creates new LF-only theta points at midpoints between nearby existing LF theta values.

Pairing rule:

- use nearest-neighbor LF theta pairs
- reject pairs that are too close or too far apart in normalized theta space
- create one midpoint theta per accepted pair by default

Workflow:

1. Read original LF block files from `outputs/training/lf`.
2. Build nearest-neighbor LF theta pairs from the original LF training CSV.
3. Create midpoint theta values between accepted pairs.
4. Recompute `inside_theta` for the source block under the midpoint theta.
5. Rerun the trained CNP on that midpoint-theta block.
6. Aggregate event predictions into LF trial rows and write an MF-GP-ready training CSV.

Run the augmentation step from the repo root:

```bash
python3 theta_augmentations/midpoint/build_midpoint_augmented_lf.py
```

Then open:

- `theta_augmentations/midpoint/01_midpoint_mfgp.ipynb`

and run it to compare baseline vs midpoint MF-GP outputs across the same transform modes used elsewhere.
