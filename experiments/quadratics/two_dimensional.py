"""Reproduce the ICLR stability sweep and shrinking-update illustration."""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from massive_lion import create_optimizer


def trajectory(spectrum, optimizer_name, steps=100, **kwargs):
    parameter = torch.nn.Parameter(torch.tensor([-1., 3.], dtype=torch.float64))
    curvature = torch.tensor(spectrum, dtype=torch.float64)
    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW([parameter], **kwargs)
    else:
        optimizer = create_optimizer(optimizer_name, [parameter], **kwargs)
    rows = []
    for step in range(steps):
        previous = parameter.detach().clone()
        parameter.grad = curvature * previous
        optimizer.step()
        velocity = (parameter.detach() - previous) / kwargs["lr"]
        rows.append(dict(step=step + 1, theta1=parameter[0].item(), theta2=parameter[1].item(),
                         loss=(0.5 * curvature * parameter.detach().square()).sum().item(),
                         directional_curvature=(curvature * velocity.square()).sum().item(),
                         speed=velocity.norm().item()))
    return rows


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(output, plots=True):
    output.mkdir(parents=True, exist_ok=True)
    beta_values = (0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)
    stability = []
    for beta in beta_values:
        for row in trajectory([50., 1.], "m_lion", lr=0.3, mass=5., betas=(beta, 0.9),
                              adaptive_mass=False, weight_decay=0.):
            stability.append(dict(optimizer="M-Lion", beta1=beta, beta2=0.9, **row))
    for row in trajectory([50., 1.], "adamw", lr=0.3, betas=(0.9, 0.999), weight_decay=0.):
        stability.append(dict(optimizer="AdamW", beta1=0.9, beta2=0.999, **row))
    shrinking = []
    for mass in (0., 0.5):
        for row in trajectory([1., 10.], "m_signum", lr=0.15, mass=mass,
                              betas=(0.9, 0.9), adaptive_mass=False, weight_decay=0.):
            shrinking.append(dict(optimizer="Signum" if mass == 0 else "M-Signum", mass=mass, **row))
    write_csv(output / "stability.csv", stability)
    write_csv(output / "shrinking_updates.csv", shrinking)
    if plots:
        plot(output, stability, shrinking)
    return stability, shrinking


def plot(output, stability, shrinking):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from experiments.plotting import create_figure

    fig, axes = create_figure(1, 2, figsize=(9, 3.5))
    selected = (0.2, 0.4, 0.6, 0.8, 0.9, 0.99)
    for beta in selected:
        rows = [r for r in stability if r["optimizer"] == "M-Lion" and r["beta1"] == beta]
        axes[0].semilogy([r["step"] for r in rows], [r["directional_curvature"] for r in rows],
                         label=rf"$\beta_1={beta:g}$")
    rows = [r for r in stability if r["optimizer"] == "AdamW"]
    axes[0].semilogy([r["step"] for r in rows], [r["directional_curvature"] for r in rows],
                     color="black", linestyle="--", label="AdamW")
    beta_values = sorted({r["beta1"] for r in stability if r["optimizer"] == "M-Lion"})
    averages = [np.mean([r["directional_curvature"] for r in stability
                        if r["optimizer"] == "M-Lion" and r["beta1"] == b]) for b in beta_values]
    axes[1].semilogy([1 - b / 0.9 for b in beta_values], averages, "o-")
    axes[0].set(xlabel="Optimizer step", ylabel=r"Directional curvature $v^\top H v$")
    axes[0].legend(fontsize=8, ncols=2)
    axes[1].set(xlabel=r"$\zeta=1-\beta_1/\beta_2$", ylabel="Mean directional curvature")
    fig.savefig(output / "stability.pdf")
    plt.close(fig)
    fig, axes = create_figure(1, 2, figsize=(8, 3.3))
    for mass, label in [(0., "Signum"), (0.5, "M-Signum")]:
        rows = [r for r in shrinking if r["mass"] == mass]
        axes[0].plot([-1.] + [r["theta1"] for r in rows], [3.] + [r["theta2"] for r in rows], label=label)
        axes[1].semilogy([r["step"] for r in rows], [r["loss"] for r in rows], label=label)
    axes[0].set(xlabel=r"$\theta_1$", ylabel=r"$\theta_2$")
    axes[1].set(xlabel="Optimizer step", ylabel="Loss")
    axes[0].legend()
    axes[1].legend()
    fig.savefig(output / "shrinking_updates.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/quadratics/2d"))
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    run(args.output, not args.no_plots)


if __name__ == "__main__":
    main()
