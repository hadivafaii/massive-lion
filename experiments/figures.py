"""Render paper experiment panels from local summary CSVs (PDF output only)."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

from experiments.plotting import create_figure


def adaptive_mass(frame):
    fig, ax = create_figure(figsize=(5.3, 3.4))
    for beta, group in frame.groupby("beta2"):
        group = group.sort_values("mass")
        ax.plot(group.mass, group.test_accuracy, "o-", label=rf"$\beta_1=\beta_2={beta:g}$")
    ax.set_xscale("symlog", linthresh=1e-5)
    ax.set(xlabel=r"Rest mass $\rho_0$", ylabel="Test accuracy (%)")
    ax.legend()
    return fig


def fixed_mass(frame):
    families = list(frame.section.unique())
    fig, axes = create_figure(1, len(families), figsize=(5 * len(families), 3.5), reshape=True)
    for ax, family in zip(axes.ravel(), families):
        for mass, group in frame[frame.section == family].groupby("mass"):
            group = group.sort_values("lr")
            ax.plot(group.lr, group.test_accuracy, "o-", label=rf"$\rho_0={mass:g}$")
        ax.set_xscale("log")
        ax.set(xlabel=r"Learning rate $\eta$", ylabel="Test accuracy (%)", title=f"M-{family}")
        ax.legend()
    return fig


def cutoffs(frame):
    families = list(frame.family.unique())
    fig, axes = create_figure(1, len(families), figsize=(6 * len(families), 3.5), reshape=True)
    vmin, vmax = frame.test_accuracy.min(), frame.test_accuracy.max()
    for ax, family in zip(axes.ravel(), families):
        group = frame[frame.family == family]
        values = group.pivot(index="mass", columns="cutoff", values="test_accuracy").sort_index()
        image = ax.imshow(values.to_numpy(), origin="lower", aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_yticks(range(len(values.index)), [rf"${m:g}$" for m in values.index])
        indices = np.arange(0, len(values.columns), 2)
        epochs = [group[group.cutoff == values.columns[i]].epochs.iloc[0] for i in indices]
        ax.set_xticks(indices, [f"{epoch:.0f}" for epoch in epochs])
        chosen = group[group.selected_mass.astype(str).str.lower() == "true"]
        for row in chosen.itertuples():
            col = values.columns.get_loc(row.cutoff)
            index = values.index.get_loc(row.mass)
            ax.add_patch(Rectangle((col - .5, index - .5), 1, 1, fill=False, edgecolor="white", linewidth=1.7))
        ax.set(xlabel="Approximate training epochs", ylabel=r"Rest mass $\rho_0$", title=f"M-{family}")
        fig.colorbar(image, ax=ax, label="Test accuracy (%)")
    return fig


def language(frame):
    # Main-panel matched comparisons: shared high momentum, clipping enabled.
    frame = frame[np.isclose(frame.beta2, .975) & np.isclose(frame.grad_clip, 1.)]
    fig, ax = create_figure(figsize=(5.3, 3.4))
    for zeta, color, label in [(0., "#555555", "SS-AdamW"), (.05, "#2878b5", "M-Lion")]:
        group = frame[np.isclose(frame.zeta, zeta)].sort_values("lr")
        if group.empty:
            raise ValueError(f"Missing required language comparison at zeta={zeta}")
        means = group.mean_validation_perplexity.to_numpy()
        errors = np.array([means - group.ci95_low.to_numpy(), group.ci95_high.to_numpy() - means])
        ax.errorbar(group.lr, means, yerr=errors, fmt="o-", capsize=3, color=color, label=label)
    ax.set(xlabel=r"Learning rate $\eta$", ylabel="Final validation perplexity",
           title=r"SlimPajama, $\beta_2=0.975$")
    ax.legend()
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("adaptive-mass", "fixed-mass", "cutoffs", "language"))
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.suffix.lower() != ".pdf":
        parser.error("--output must be a PDF path")
    frame = pd.read_csv(args.summary)
    fig = {"adaptive-mass": adaptive_mass, "fixed-mass": fixed_mass,
           "cutoffs": cutoffs, "language": language}[args.mode](frame)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)
    plt.close(fig)


if __name__ == "__main__":
    main()
