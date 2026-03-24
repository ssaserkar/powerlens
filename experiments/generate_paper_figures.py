#!/usr/bin/env python3
"""
Generate publication-ready figures for arXiv paper.
Run after paper_experiments.py has produced results.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

# Use LaTeX-compatible fonts
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

FIGURES_DIR = Path("experiments/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# PLACEHOLDER DATA — Replace with actual experiment results
# ============================================================

# These are structured so you can drop in real numbers
# after running paper_experiments.py

MODELS = ["MobileNetV2", "ResNet-18", "ResNet-34", 
          "ResNet-50", "EfficientNet-B0"]

# MAXN mode results (replace TBD with real values)
MAXN_RESULTS = {
    #                  latency_ms  energy_mJ  power_W  gpu_util
    "MobileNetV2":    [0.0,        0.0,       0.0,     0.0],
    "ResNet-18":      [0.0,        0.0,       0.0,     0.0],
    "ResNet-34":      [0.0,        0.0,       0.0,     0.0],
    "ResNet-50":      [0.0,        0.0,       0.0,     0.0],
    "EfficientNet-B0":[0.0,        0.0,       0.0,     0.0],
}

# Per-rail breakdown at MAXN (replace with real values)
# [VDD_CPU_GPU_CV_mJ, VDD_SOC_mJ]  (VDD_IN = sum + losses)
RAIL_BREAKDOWN = {
    "MobileNetV2":    [0.0, 0.0],
    "ResNet-18":      [0.0, 0.0],
    "ResNet-34":      [0.0, 0.0],
    "ResNet-50":      [0.0, 0.0],
    "EfficientNet-B0":[0.0, 0.0],
}

# Inferences per joule across power modes
# {model: [15W, 25W, MAXN]}
INF_PER_JOULE = {
    "MobileNetV2":    [0.0, 0.0, 0.0],
    "ResNet-18":      [0.0, 0.0, 0.0],
    "ResNet-34":      [0.0, 0.0, 0.0],
    "ResNet-50":      [0.0, 0.0, 0.0],
    "EfficientNet-B0":[0.0, 0.0, 0.0],
}

# Energy per inference across power modes (mJ)
# {model: [15W, 25W, MAXN]}
ENERGY_PER_MODE = {
    "MobileNetV2":    [0.0, 0.0, 0.0],
    "ResNet-18":      [0.0, 0.0, 0.0],
    "ResNet-34":      [0.0, 0.0, 0.0],
    "ResNet-50":      [0.0, 0.0, 0.0],
    "EfficientNet-B0":[0.0, 0.0, 0.0],
}

# Thermal time series (seconds, temperature °C)
# Replace with actual sampled data
THERMAL_TIME = np.arange(0, 150, 1)  # 150 seconds
THERMAL_GPU = np.zeros(150)  # Replace with real data

# Batch scaling (iteration count, per-inference energy mJ)
BATCH_ITERATIONS = [1, 5, 10, 25, 50, 100, 250, 500, 1000]
BATCH_ENERGY = [0.0] * 9  # Replace with real data
BATCH_STDDEV = [0.0] * 9  # Replace with real data


# ============================================================
# FIGURE 1: Per-inference energy across models (bar chart)
# ============================================================
def fig1_energy_per_model():
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    
    energies = [MAXN_RESULTS[m][1] for m in MODELS]
    colors = ["#2196F3", "#4CAF50", "#8BC34A", "#FF9800", "#9C27B0"]
    
    bars = ax.bar(range(len(MODELS)), energies, color=colors, 
                  edgecolor="black", linewidth=0.5)
    
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels(MODELS, rotation=30, ha="right")
    ax.set_ylabel("Energy per Inference (mJ)")
    ax.set_title("Per-Inference Energy (MAXN Mode)")
    
    # Add value labels on bars
    for bar, val in zip(bars, energies):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f"{val:.1f}", ha="center", va="bottom", fontsize=8)
    
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig1_energy_per_model.pdf")
    plt.savefig(FIGURES_DIR / "fig1_energy_per_model.png")
    print("  Saved fig1_energy_per_model")


# ============================================================
# FIGURE 2: Per-rail energy breakdown (stacked bar)
# ============================================================
def fig2_rail_breakdown():
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    
    cpu_gpu = [RAIL_BREAKDOWN[m][0] for m in MODELS]
    soc = [RAIL_BREAKDOWN[m][1] for m in MODELS]
    
    x = range(len(MODELS))
    ax.bar(x, soc, label="VDD_SOC", color="#FF9800",
           edgecolor="black", linewidth=0.5)
    ax.bar(x, cpu_gpu, bottom=soc, label="VDD_CPU_GPU_CV", 
           color="#2196F3", edgecolor="black", linewidth=0.5)
    
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS, rotation=30, ha="right")
    ax.set_ylabel("Energy per Inference (mJ)")
    ax.set_title("Per-Rail Energy Breakdown (MAXN)")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig2_rail_breakdown.pdf")
    plt.savefig(FIGURES_DIR / "fig2_rail_breakdown.png")
    print("  Saved fig2_rail_breakdown")


# ============================================================
# FIGURE 3: Inferences per joule across power modes
# ============================================================
def fig3_efficiency_by_mode():
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    
    x = np.arange(len(MODELS))
    width = 0.25
    
    modes = ["15W", "25W", "MAXN"]
    colors = ["#4CAF50", "#2196F3", "#F44336"]
    
    for i, (mode, color) in enumerate(zip(modes, colors)):
        values = [INF_PER_JOULE[m][i] for m in MODELS]
        ax.bar(x + i * width, values, width, label=mode,
               color=color, edgecolor="black", linewidth=0.5)
    
    ax.set_xticks(x + width)
    ax.set_xticklabels(MODELS, rotation=30, ha="right")
    ax.set_ylabel("Inferences per Joule")
    ax.set_title("Energy Efficiency by Power Mode")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig3_efficiency_by_mode.pdf")
    plt.savefig(FIGURES_DIR / "fig3_efficiency_by_mode.png")
    print("  Saved fig3_efficiency_by_mode")


# ============================================================
# FIGURE 4: Energy-latency frontier (scatter)
# ============================================================
def fig4_energy_latency_frontier():
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    
    markers = ["o", "s", "^", "D", "v"]
    colors = ["#4CAF50", "#2196F3", "#F44336"]
    mode_names = ["15W", "25W", "MAXN"]
    
    for i, mode in enumerate(mode_names):
        for j, model in enumerate(MODELS):
            energy = ENERGY_PER_MODE[model][i]
            # Need latency per mode too — add to data structure
            # For now placeholder
            latency = 0.0  # TBD
            
            ax.scatter(latency, energy, marker=markers[j],
                      color=colors[i], s=60, edgecolors="black",
                      linewidth=0.5, zorder=3)
    
    # Legend for modes (colors)
    for i, (mode, color) in enumerate(zip(mode_names, colors)):
        ax.scatter([], [], color=color, label=mode, s=60,
                  edgecolors="black", linewidth=0.5)
    
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Energy per Inference (mJ)")
    ax.set_title("Energy-Latency Trade-off")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig4_energy_latency.pdf")
    plt.savefig(FIGURES_DIR / "fig4_energy_latency.png")
    print("  Saved fig4_energy_latency")


# ============================================================
# FIGURE 5: Thermal time series
# ============================================================
def fig5_thermal():
    fig, ax = plt.subplots(figsize=(3.5, 2.0))
    
    ax.plot(THERMAL_TIME, THERMAL_GPU, color="#F44336", 
            linewidth=1.5, label="GPU Temperature")
    
    ax.axhline(y=85, color="black", linestyle="--", 
               linewidth=0.8, label="Throttle Threshold")
    
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title("Thermal Response Under Sustained Inference (MAXN)")
    ax.legend(loc="right", framealpha=0.9)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 150)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig5_thermal.pdf")
    plt.savefig(FIGURES_DIR / "fig5_thermal.png")
    print("  Saved fig5_thermal")


# ============================================================
# FIGURE 6: Batch scaling (energy vs iteration count)
# ============================================================
def fig6_batch_scaling():
    fig, ax = plt.subplots(figsize=(3.5, 2.0))
    
    ax.errorbar(BATCH_ITERATIONS, BATCH_ENERGY, 
                yerr=BATCH_STDDEV, fmt="o-", color="#2196F3",
                markersize=4, linewidth=1.2, capsize=3,
                ecolor="#90CAF9")
    
    ax.set_xscale("log")
    ax.set_xlabel("Iterations per Measurement Window")
    ax.set_ylabel("Per-Inference Energy (mJ)")
    ax.set_title("Energy Measurement vs Batch Size (ResNet-18, MAXN)")
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig6_batch_scaling.pdf")
    plt.savefig(FIGURES_DIR / "fig6_batch_scaling.png")
    print("  Saved fig6_batch_scaling")


# ============================================================
# FIGURE 7: Power draw time series during single profiling run
# ============================================================
def fig7_power_timeseries():
    """
    Shows raw power samples during a profiling session.
    Demonstrates what PowerLens captures.
    Replace with actual sampled data from a single run.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.5, 3.5),
                                     sharex=True)
    
    # Placeholder — replace with actual PowerLens samples
    t = np.linspace(0, 2, 200)  # 2 seconds of data
    vdd_in = np.zeros(200)
    vdd_cpu_gpu = np.zeros(200)
    vdd_soc = np.zeros(200)
    gpu_util = np.zeros(200)
    
    # Top: Power rails
    ax1.plot(t, vdd_in, label="VDD_IN", color="#F44336", linewidth=1)
    ax1.plot(t, vdd_cpu_gpu, label="VDD_CPU_GPU_CV", 
             color="#2196F3", linewidth=1)
    ax1.plot(t, vdd_soc, label="VDD_SOC", color="#FF9800", linewidth=1)
    ax1.set_ylabel("Power (W)")
    ax1.legend(loc="upper right", fontsize=7, framealpha=0.9)
    ax1.grid(alpha=0.3)
    ax1.set_title("Power Rails During Inference Profiling")
    
    # Bottom: GPU utilization
    ax2.fill_between(t, gpu_util, alpha=0.3, color="#4CAF50")
    ax2.plot(t, gpu_util, color="#4CAF50", linewidth=1)
    ax2.set_ylabel("GPU Utilization (%)")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylim(0, 100)
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig7_power_timeseries.pdf")
    plt.savefig(FIGURES_DIR / "fig7_power_timeseries.png")
    print("  Saved fig7_power_timeseries")


# ============================================================
# MAIN
# ============================================================
def main():
    print("Generating paper figures...\n")
    
    fig1_energy_per_model()
    fig2_rail_breakdown()
    fig3_efficiency_by_mode()
    fig4_energy_latency_frontier()
    fig5_thermal()
    fig6_batch_scaling()
    fig7_power_timeseries()
    
    print(f"\nAll figures saved to {FIGURES_DIR}/")
    print("\nFigure checklist for paper:")
    print("  Fig 1: Per-inference energy across models")
    print("  Fig 2: Per-rail energy breakdown")
    print("  Fig 3: Efficiency (inf/J) by power mode")
    print("  Fig 4: Energy-latency frontier")
    print("  Fig 5: Thermal time series")
    print("  Fig 6: Batch scaling validation")
    print("  Fig 7: Raw power time series")


if __name__ == "__main__":
    main()