"""Plot the two submitted heterogeneous-quadratic panels from rerun results."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.plotting import create_figure
from .summarize import metrics


def load(directory, name):
    receipt = json.loads((directory / f"{name}_receipt.json").read_text())
    with np.load(directory / f"{name}_results.npz") as result:
        if result["failed"].any():
            raise ValueError(f"{name} contains failed trajectories; inspect them before plotting")
        return receipt["configs"], result["losses"].copy()


def plot(directory, output):
    configurations, losses = load(directory, "random_confirm_heterogeneous_3")
    fig, axes = create_figure(1, 2, figsize=(10, 3.5))
    inset = axes[0].inset_axes([0.51, 0.42, 0.45, 0.52])
    for zeta, color, label in [(0, "#555555", "SS-AdamW"), (0.2, "#7552a3", "M-Lion")]:
        index = next(i for i, c in enumerate(configurations)
                     if c["selection"] == "auc" and c["zeta"] == zeta)
        relative = losses[:201, index] / losses[0, index]
        low, median, high = np.quantile(relative, [0.25, 0.5, 0.75], axis=1)
        steps = np.arange(len(median))
        for ax in (axes[0], inset):
            ax.plot(steps, median, color=color, label=label)
            ax.fill_between(steps, low, high, color=color, alpha=0.16)
        print(f"{label}: beta=0.95, zeta={zeta}, lr={configurations[index]['lr']}")
    inset.set_yscale("log")
    inset.tick_params(labelsize=8)
    axes[0].set(xlabel="Optimizer step", ylabel=r"Relative loss $L_t/L_0$",
                title=r"Heterogeneous quadratic, $\beta_2=0.95$")
    axes[0].legend(loc="upper left")
    for beta, color in [(0.9, "#2878b5"), (0.95, "#7552a3"), (0.975, "#00896b")]:
        name = ("random_confirm_heterogeneous_3" if beta == 0.95
                else f"beta_confirm_{beta}_3")
        configurations, losses = load(directory, name)
        indices = [i for i, c in enumerate(configurations) if c["selection"] == "auc"]
        zeta = [configurations[i]["zeta"] for i in indices]
        scores = metrics(losses)["auc"][indices]
        mean = scores.mean(1)
        margin = 1.96 * scores.std(1, ddof=1) / np.sqrt(scores.shape[1])
        axes[1].plot(zeta, mean, "o-", color=color, label=rf"$\beta_2={beta:g}$", ms=4)
        axes[1].fill_between(zeta, mean - margin, mean + margin, color=color, alpha=0.13)
    axes[1].set(xlabel=r"Curvature strength $\zeta$", ylabel="Mean log-loss score",
                title="Learning rate tuned at each curvature strength")
    axes[1].set_xscale("symlog", linthresh=0.01)
    axes[1].set_xticks([0, 0.01, 0.05, 0.2, 0.8], ["0", "0.01", "0.05", "0.2", "0.8"])
    axes[1].legend()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("outputs/quadratics/9d"))
    parser.add_argument("--output", type=Path, default=Path("outputs/quadratics/heterogeneous.pdf"))
    args = parser.parse_args()
    plot(args.input, args.output)


if __name__ == "__main__":
    main()
