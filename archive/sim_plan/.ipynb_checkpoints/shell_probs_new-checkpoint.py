# This file currently takes prediction outputs from the new version of the CNP and plots the distribution
# Headers of the CSV are: iteration, fidelity, source_file, event_index, predicted_shell_index, true_shell_index, predicted_shell_score, y_cnp_err, R_shell, Z_shell, y_raw, predicted_shell_probability

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

def plot_histogram(data: np.ndarray, nshells: int, title: str, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    
    ax.hist(data, np.arange(nshells))
    ax.set_xlabel("Shell Number")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.grid()
    ax.set_axisbelow(True)

def main(filename: Path | str, outpath: Path | str):
    data = pd.read_csv(filename)
    best_shells = data["predicted_shell_index"].to_numpy()
    true_shells = data["true_shell_index"].to_numpy()
    nshells = max(max(true_shells), max(best_shells))

    fig, (ax1, ax2) = plt.subplots(2, figsize=(10,10))
    plot_histogram(best_shells, nshells, "Predicted Shells", ax1)
    plot_histogram(true_shells, nshells, "True Shells", ax2)
    
    fig.tight_layout()
    plt.savefig(outpath)

if __name__ == "__main__":
    outpath = Path("./predicted_shell_probs.png")
    filename = Path("../data/out/cnp/cnp_xlzd_equal_volume_shell_v1_minibatch_train_15epochs_best_shell.csv")
    main(filename, outpath)
