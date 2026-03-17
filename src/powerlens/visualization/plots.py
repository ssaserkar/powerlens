"""
Matplotlib-based power trace visualization.

Generates publication-quality plots of power consumption
over time with inference events highlighted.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving to file
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional

from powerlens.sensors.types import PowerSample
from powerlens.analysis.energy import EnergyReport


def plot_power_trace(
    samples: List[List[PowerSample]],
    report: Optional[EnergyReport] = None,
    filepath: str = "power_trace.png",
    title: str = "PowerLens Power Trace",
    figsize: tuple = (12, 6),
) -> str:
    """Plot power consumption over time.

    Shows total power and per-rail breakdown with inference
    events highlighted as shaded regions.

    Args:
        samples: Raw power samples from PowerSampler.get_samples().
        report: Optional EnergyReport to overlay inference markers.
        filepath: Output image path (.png, .pdf, .svg).
        title: Plot title.
        figsize: Figure size in inches (width, height).

    Returns:
        Absolute path to the saved plot.
    """
    if not samples:
        raise ValueError("No samples to plot")

    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Extract data from samples
    # Use first sample's timestamp as t=0
    t0 = samples[0][0].timestamp

    timestamps = []
    total_power = []
    rail_data = {}

    for cycle in samples:
        t = cycle[0].timestamp - t0
        timestamps.append(t)
        cycle_total = 0.0

        for sample in cycle:
            if sample.rail_name not in rail_data:
                rail_data[sample.rail_name] = {"times": [], "power": []}
            rail_data[sample.rail_name]["times"].append(t)
            rail_data[sample.rail_name]["power"].append(sample.power_w)
            cycle_total += sample.power_w

        total_power.append(cycle_total)

    timestamps = np.array(timestamps)
    total_power = np.array(total_power)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot total power
    ax.plot(timestamps, total_power, color="black", linewidth=1.5,
            label="Total Power", zorder=3)

    # Plot per-rail power
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]
    for i, (rail_name, data) in enumerate(sorted(rail_data.items())):
        color = colors[i % len(colors)]
        ax.plot(data["times"], data["power"], color=color,
                linewidth=1.0, alpha=0.7, label=rail_name)

    # Highlight inference events
    if report and report.inferences:
        for inf in report.inferences:
            start = inf.start_time - t0
            end = inf.end_time - t0
            ax.axvspan(start, end, alpha=0.15, color="orange", zorder=1)

        # Add annotation for first inference
        first = report.inferences[0]
        mid = (first.start_time + first.end_time) / 2 - t0
        ax.annotate(
            f"{first.energy_j:.4f} J",
            xy=(mid, total_power.max() * 0.9),
            fontsize=8,
            ha="center",
            color="darkorange",
        )

    ax.set_xlabel("Time (seconds)", fontsize=11)
    ax.set_ylabel("Power (watts)", fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(timestamps[0], timestamps[-1])
    ax.set_ylim(bottom=0)

    # Add summary text box if report available
    if report and report.num_inferences > 0:
        textstr = (
            f"Inferences: {report.num_inferences}\n"
            f"Energy/inf: {report.mean_energy_j:.4f} J\n"
            f"Avg power: {report.mean_power_w:.2f} W\n"
            f"Peak power: {report.peak_power_w:.2f} W"
        )
        props = dict(boxstyle="round", facecolor="wheat", alpha=0.8)
        ax.text(0.02, 0.97, textstr, transform=ax.transAxes, fontsize=9,
                verticalalignment="top", bbox=props)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)

    return str(path.resolve())