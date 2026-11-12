"""Reusable Matplotlib visualizations for Epiflux simulation results."""
from __future__ import annotations

import matplotlib.pyplot as plt


def plot_sir(result, ax=None):
    """Plot susceptible, infected and recovered trajectories."""
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(result.t, result.S, label="Susceptible (S)", linewidth=2)
    ax.plot(result.t, result.I, label="Infected (I)", linewidth=2)
    ax.plot(result.t, result.R, label="Recovered (R)", linewidth=2)
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("People")
    ax.set_title(f"SIR simulation ({result.method.upper()})")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False)
    return ax


def plot_method_comparison(results, reference=None, ax=None):
    """Compare infected trajectories from multiple numerical methods."""
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4.5))
    for name, result in results.items():
        ax.plot(result.t, result.I, label=name.upper(), linewidth=2)
    if reference is not None:
        ax.plot(results["rk4"].t, reference[:, 1], "--",
                label="DOP853 reference", linewidth=1.8)
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Infected people")
    ax.set_title("Numerical method comparison")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False)
    return ax
