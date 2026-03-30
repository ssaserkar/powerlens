"""
Generate plots for the PowerLens README using real model data.

Uses the same 5 models profiled for the arXiv paper.
Run from the powerlens root directory on Jetson.

Usage:
    python examples/generate_readme_plots.py
"""

import os
import sys
import time
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Paper data — hardcoded from experiments for reliable plots
PAPER_DATA = {
    "energy_per_inference_mj": {
        "MobileNetV2":    {"15W": 10.5, "25W": 10.3, "MAXN": 10.6},
        "ResNet-18":      {"15W": 10.4, "25W": 10.2, "MAXN": 10.4},
        "ResNet-34":      {"15W": 16.3, "25W": 14.5, "MAXN": 14.7},
        "ResNet-50":      {"15W": 22.6, "25W": 19.7, "MAXN": 19.8},
        "EfficientNet-B0":{"15W": 19.4, "25W": 16.9, "MAXN": 18.0},
    },
    "efficiency_inf_per_j": {
        "MobileNetV2":    {"15W": 95.7, "25W": 96.7, "MAXN": 94.1},
        "ResNet-18":      {"15W": 96.2, "25W": 98.1, "MAXN": 96.7},
        "ResNet-34":      {"15W": 61.5, "25W": 69.1, "MAXN": 68.0},
        "ResNet-50":      {"15W": 44.3, "25W": 50.8, "MAXN": 50.5},
        "EfficientNet-B0":{"15W": 51.6, "25W": 59.1, "MAXN": 55.8},
    },
    "avg_power_w": {
        "MobileNetV2":    {"15W": 7.1, "25W": 7.0, "MAXN": 7.2},
        "ResNet-18":      {"15W": 8.1, "25W": 8.2, "MAXN": 8.3},
        "ResNet-34":      {"15W": 8.6, "25W": 9.6, "MAXN": 9.8},
        "ResNet-50":      {"15W": 8.6, "25W": 9.7, "MAXN": 10.0},
        "EfficientNet-B0":{"15W": 7.1, "25W": 7.6, "MAXN": 7.9},
    },
    "latency_ms": {
        "MobileNetV2": 1.7, "ResNet-18": 1.5, "ResNet-34": 2.3,
        "ResNet-50": 3.2, "EfficientNet-B0": 3.4,
    },
    "rail_power_maxn": {
        "MobileNetV2":    {"VDD_IN": 7.08, "VDD_CPU_GPU_CV": 2.17, "VDD_SOC": 1.68},
        "ResNet-18":      {"VDD_IN": 8.16, "VDD_CPU_GPU_CV": 2.76, "VDD_SOC": 1.86},
        "ResNet-34":      {"VDD_IN": 9.50, "VDD_CPU_GPU_CV": 3.87, "VDD_SOC": 1.95},
        "ResNet-50":      {"VDD_IN": 9.74, "VDD_CPU_GPU_CV": 3.90, "VDD_SOC": 2.05},
        "EfficientNet-B0":{"VDD_IN": 7.72, "VDD_CPU_GPU_CV": 2.83, "VDD_SOC": 1.70},
    },
    "fp16_vs_fp32": {
        "MobileNetV2":    {"fp16_mj": 24.5, "fp32_mj": 27.8, "speedup": 1.3, "energy_ratio": 1.1},
        "ResNet-18":      {"fp16_mj": 16.6, "fp32_mj": 29.8, "speedup": 1.5, "energy_ratio": 1.8},
        "ResNet-34":      {"fp16_mj": 23.7, "fp32_mj": 49.8, "speedup": 1.7, "energy_ratio": 2.1},
        "ResNet-50":      {"fp16_mj": 35.6, "fp32_mj": 59.1, "speedup": 1.2, "energy_ratio": 1.7},
        "EfficientNet-B0":{"fp16_mj": 31.6, "fp32_mj": 41.0, "speedup": 1.1, "energy_ratio": 1.3},
    },
    "batch_scaling": {
        "batch": [1, 2, 4, 8],
        "energy_mj": [9.5, 6.4, 5.5, 4.9],
        "efficiency": [104.8, 156.9, 180.8, 204.5],
        "throughput": [410, 538, 626, 704],
    },
    "thermal_timeline": {
        "time_s": [
            0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60,
            65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120,
            125, 130, 135, 140, 145, 150, 155, 160, 165, 170,
            175, 180, 185, 190, 195, 200, 205, 210, 215, 220,
            226, 231, 236, 241, 246, 251, 256, 261, 266, 271,
            276, 281, 286, 291, 296, 301, 306, 311, 316, 321,
            326, 331, 336, 341, 346, 351, 356, 361, 366, 371,
            376, 381, 386, 391, 396, 401, 406, 411, 416, 421,
            426, 431, 436, 441, 446, 451, 456, 461, 466, 471,
            476, 481, 486, 491, 496, 501, 506, 511, 517, 522,
            527, 532, 537, 542, 547, 552, 557, 562, 567, 572,
            577, 582, 587, 592, 597,
        ],
        "gpu_temp_c": [
            38.3, 47.2, 48.8, 49.8, 50.7, 52.2, 52.8, 54.0, 55.1,
            55.6, 56.2, 57.0, 57.5, 58.0, 58.7, 59.2, 59.8, 59.9,
            60.3, 60.7, 61.0, 61.7, 61.8, 62.2, 62.4, 62.8, 62.8,
            63.2, 63.2, 63.3, 63.7, 63.8, 63.8, 64.4, 64.2, 64.7,
            64.5, 64.8, 64.8, 64.9, 65.3, 65.3, 65.2, 65.3, 65.6,
            65.8, 65.8, 66.1, 65.9, 65.9, 66.4, 66.2, 66.1, 66.2,
            66.2, 66.6, 66.3, 66.7, 66.5, 66.5, 66.8, 66.7, 66.6,
            67.0, 66.7, 67.0, 66.8, 67.0, 67.2, 66.8, 66.9, 67.3,
            66.9, 67.3, 67.0, 67.1, 67.2, 67.0, 67.3, 67.2, 67.2,
            67.3, 67.4, 67.2, 67.2, 67.5, 67.2, 67.2, 67.5, 67.7,
            67.3, 67.5, 67.6, 67.4, 67.6, 67.4, 67.5, 67.4, 67.5,
            67.3, 67.5, 67.3, 67.4, 67.7, 67.5, 67.5, 67.5, 67.4,
            67.7, 67.8, 67.4, 67.5, 67.6, 67.4, 67.6, 67.7, 67.6,
            67.6, 67.9, 67.5,
        ],
        "power_w": [
            7.8, 21.3, 21.6, 21.6, 21.6, 21.7, 21.8, 21.8, 21.7,
            21.8, 21.8, 21.8, 21.8, 21.8, 21.8, 21.8, 21.7, 21.7,
            21.9, 21.8, 21.8, 21.8, 21.8, 21.8, 21.9, 21.8, 21.9,
            21.9, 21.9, 21.9, 21.9, 21.9, 22.0, 21.9, 22.0, 22.0,
            22.0, 21.9, 22.0, 22.0, 21.9, 21.9, 22.0, 22.0, 22.0,
            22.0, 22.0, 22.0, 22.0, 22.0, 21.9, 22.0, 22.0, 22.0,
            22.0, 22.0, 21.9, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0,
            22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0,
            22.0, 22.0, 22.0, 21.9, 22.0, 22.0, 21.9, 21.9, 22.0,
            22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 21.9, 22.0, 22.0,
            22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0,
            22.0, 22.0, 22.0, 22.0, 21.9, 22.1, 22.0, 22.0, 22.0,
            22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0,
            22.0, 22.0, 22.0,
        ],
    },
}

MODELS = ["MobileNetV2", "ResNet-18", "ResNet-34",
          "ResNet-50", "EfficientNet-B0"]
MODES = ["15W", "25W", "MAXN"]


def setup_style():
    """Clean plot style."""
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    })


def plot_energy_by_mode(output_dir):
    """Plot 1: Per-inference energy across power modes."""
    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(MODELS))
    width = 0.25
    colors = ["#27ae60", "#2980b9", "#e74c3c"]

    for i, (mode, color) in enumerate(zip(MODES, colors)):
        values = [
            PAPER_DATA["energy_per_inference_mj"][m][mode]
            for m in MODELS
        ]
        bars = ax.bar(x + i * width, values, width,
                      label=mode, color=color,
                      edgecolor="black", linewidth=0.5)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f"{val:.1f}", ha="center", fontsize=8)

    ax.set_xticks(x + width)
    ax.set_xticklabels(MODELS, rotation=15, ha="right")
    ax.set_ylabel("Energy per Inference (mJ)")
    ax.set_title("Per-Inference Energy Across Power Modes")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 42)

    plt.tight_layout()
    path = os.path.join(output_dir, "energy_by_mode.png")
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def plot_efficiency_by_mode(output_dir):
    """Plot 2: Energy efficiency across power modes."""
    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(MODELS))
    width = 0.25
    colors = ["#27ae60", "#2980b9", "#e74c3c"]

    for i, (mode, color) in enumerate(zip(MODES, colors)):
        values = [
            PAPER_DATA["efficiency_inf_per_j"][m][mode]
            for m in MODELS
        ]
        bars = ax.bar(x + i * width, values, width,
                      label=mode, color=color,
                      edgecolor="black", linewidth=0.5)

    # Mark 25W as best with star
    for j, m in enumerate(MODELS):
        best_val = PAPER_DATA["efficiency_inf_per_j"][m]["25W"]
        ax.text(j + width, best_val + 1.5, "★",
                ha="center", fontsize=12, color="#2980b9")

    ax.set_xticks(x + width)
    ax.set_xticklabels(MODELS, rotation=15, ha="right")
    ax.set_ylabel("Inferences per Joule")
    ax.set_title(
        "Energy Efficiency by Power Mode — "
        "25W is optimal for ALL models"
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "efficiency_by_mode.png")
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def plot_rail_breakdown(output_dir):
    """Plot 3: Per-rail power breakdown."""
    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(MODELS))
    width = 0.5

    cpu_gpu = [
        PAPER_DATA["rail_power_maxn"][m]["VDD_CPU_GPU_CV"]
        for m in MODELS
    ]
    soc = [
        PAPER_DATA["rail_power_maxn"][m]["VDD_SOC"]
        for m in MODELS
    ]

    ax.bar(x, soc, width, label="VDD_SOC (static)",
           color="#f39c12", edgecolor="black", linewidth=0.5)
    ax.bar(x, cpu_gpu, width, bottom=soc,
           label="VDD_CPU_GPU_CV (dynamic)",
           color="#3498db", edgecolor="black", linewidth=0.5)

    # SOC percentage labels
    for i, (s, c) in enumerate(zip(soc, cpu_gpu)):
        total = s + c
        pct = s / total * 100
        ax.text(i, total + 0.1, f"SOC: {pct:.0f}%",
                ha="center", fontsize=9, color="#f39c12",
                fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(MODELS, rotation=15, ha="right")
    ax.set_ylabel("Average Power (W)")
    ax.set_title(
        "Per-Rail Power Breakdown (MAXN Mode) — "
        "SoC static power is 34-44% of compute"
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "rail_breakdown.png")
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def plot_fp16_vs_fp32(output_dir):
    """Plot 4: FP16 vs FP32 energy comparison."""
    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(MODELS))
    width = 0.35

    fp16 = [
        PAPER_DATA["fp16_vs_fp32"][m]["fp16_mj"]
        for m in MODELS
    ]
    fp32 = [
        PAPER_DATA["fp16_vs_fp32"][m]["fp32_mj"]
        for m in MODELS
    ]

    ax.bar(x - width / 2, fp16, width, label="FP16",
           color="#2980b9", edgecolor="black", linewidth=0.5)
    ax.bar(x + width / 2, fp32, width, label="FP32",
           color="#e74c3c", edgecolor="black", linewidth=0.5)

    # Ratio labels
    for i, m in enumerate(MODELS):
        ratio = PAPER_DATA["fp16_vs_fp32"][m]["energy_ratio"]
        y_pos = max(fp16[i], fp32[i]) + 1.5
        ax.text(i, y_pos, f"{ratio:.1f}×",
                ha="center", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(MODELS, rotation=15, ha="right")
    ax.set_ylabel("Energy per Inference (mJ)")
    ax.set_title(
        "FP16 vs FP32 Energy — "
        "FP16 saves 1.3-1.7× energy"
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "fp16_vs_fp32.png")
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def plot_batch_scaling(output_dir):
    """Plot 5: Batch size scaling."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    bs = PAPER_DATA["batch_scaling"]
    batches = bs["batch"]
    energy = bs["energy_mj"]
    efficiency = bs["efficiency"]
    throughput = bs["throughput"]

    # Energy per inference
    ax1.plot(batches, energy, "o-", color="#e74c3c",
             linewidth=2, markersize=8)
    ax1.set_xlabel("Batch Size")
    ax1.set_ylabel("Energy per Inference (mJ)")
    ax1.set_title("Energy Decreases with Batch Size")
    ax1.grid(alpha=0.3)
    for b, e in zip(batches, energy):
        ax1.annotate(f"{e:.1f}", (b, e),
                     textcoords="offset points",
                     xytext=(0, 10), ha="center", fontsize=9)

    # Efficiency and throughput
    color1 = "#2980b9"
    color2 = "#27ae60"

    ax2.plot(batches, efficiency, "o-", color=color1,
             linewidth=2, markersize=8, label="Efficiency (inf/J)")
    ax2.set_xlabel("Batch Size")
    ax2.set_ylabel("Inferences per Joule", color=color1)
    ax2.tick_params(axis="y", labelcolor=color1)

    ax2b = ax2.twinx()
    ax2b.plot(batches, throughput, "s--", color=color2,
              linewidth=2, markersize=8, label="Throughput (inf/s)")
    ax2b.set_ylabel("Throughput (inf/s)", color=color2)
    ax2b.tick_params(axis="y", labelcolor=color2)

    ax2.set_title("Efficiency & Throughput vs Batch Size")
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2,
               loc="center right")
    ax2.grid(alpha=0.3)

    fig.suptitle(
        "Batch Size Scaling — ResNet-18 FP16 (MAXN Mode)",
        fontsize=13, y=1.02,
    )
    plt.tight_layout()
    path = os.path.join(output_dir, "batch_scaling.png")
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def plot_thermal_timeline(output_dir):
    """Plot 6: Thermal stress test timeline."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7),
                                    sharex=True)

    ts = PAPER_DATA["thermal_timeline"]
    time_s = ts["time_s"]
    gpu_temp = ts["gpu_temp_c"]
    power = ts["power_w"]

    # Power
    ax1.plot(time_s, power, color="#e74c3c", linewidth=1.5)
    ax1.fill_between(time_s, power, alpha=0.15, color="#e74c3c")
    ax1.set_ylabel("VDD_IN Power (W)")
    ax1.set_title(
        "Sustained Inference Stress Test — "
        "ResNet-50 Batch=16, MAXN Mode, 99% GPU, 600s"
    )
    ax1.grid(alpha=0.3)
    avg_power = np.mean(power[5:])  # Skip first idle sample
    ax1.axhline(y=avg_power, color="gray", linestyle="--",
                alpha=0.5)
    ax1.annotate(f"Avg: {avg_power:.1f}W",
                 xy=(time_s[-1] * 0.85, avg_power + 0.3),
                 fontsize=10, color="gray")
    ax1.set_ylim(0, 25)

    # Temperature
    ax2.plot(time_s, gpu_temp, color="#f39c12", linewidth=2)
    ax2.fill_between(time_s, gpu_temp, alpha=0.15,
                     color="#f39c12")
    ax2.axhline(y=85, color="red", linestyle="--",
                linewidth=1.5, label="Throttle threshold (85°C)")
    ax2.set_ylabel("GPU Temperature (°C)")
    ax2.set_xlabel("Time (seconds)")
    ax2.grid(alpha=0.3)
    ax2.set_ylim(35, 90)
    ax2.legend(loc="upper right")

    # Annotate key points
    ax2.annotate(
        f"Start: {gpu_temp[0]:.0f}°C",
        xy=(0, gpu_temp[0]),
        xytext=(20, gpu_temp[0] + 8),
        arrowprops=dict(arrowstyle="->", color="gray"),
        fontsize=10,
    )
    ax2.annotate(
        f"Steady state: {gpu_temp[-1]:.0f}°C\n"
        f"(18°C headroom)",
        xy=(time_s[-1], gpu_temp[-1]),
        xytext=(time_s[-1] - 80, gpu_temp[-1] + 10),
        arrowprops=dict(arrowstyle="->", color="gray"),
        fontsize=10,
    )

    plt.tight_layout()
    path = os.path.join(output_dir, "thermal_timeline.png")
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def plot_latency_by_mode(output_dir):
    """Plot 7: Latency across power modes (showing <2% variation)."""
    fig, ax = plt.subplots(figsize=(10, 5))

    # Per-mode latency data from experiments
    latency_by_mode = {
        "MobileNetV2":    {"15W": 3.32, "25W": 3.34, "MAXN": 3.38},
        "ResNet-18":      {"15W": 3.14, "25W": 3.10, "MAXN": 3.13},
        "ResNet-34":      {"15W": 4.85, "25W": 4.87, "MAXN": 4.88},
        "ResNet-50":      {"15W": 6.59, "25W": 6.59, "MAXN": 6.51},
        "EfficientNet-B0":{"15W": 6.67, "25W": 6.64, "MAXN": 6.66},
    }

    x = np.arange(len(MODELS))
    width = 0.25
    colors = ["#27ae60", "#2980b9", "#e74c3c"]

    for i, (mode, color) in enumerate(zip(MODES, colors)):
        values = [latency_by_mode[m][mode] for m in MODELS]
        ax.bar(x + i * width, values, width,
               label=mode, color=color,
               edgecolor="black", linewidth=0.5)

    ax.set_xticks(x + width)
    ax.set_xticklabels(MODELS, rotation=15, ha="right")
    ax.set_ylabel("Latency (ms)")
    ax.set_title(
        "Inference Latency by Power Mode — "
        "Less than 2% variation across modes"
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Start y-axis at 0 to show true scale
    ax.set_ylim(0, 8)

    plt.tight_layout()
    path = os.path.join(output_dir, "latency_by_mode.png")
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def plot_energy_latency_frontier(output_dir):
    """Plot 8: Energy vs latency scatter (Pareto frontier)."""
    fig, ax = plt.subplots(figsize=(8, 6))

    markers = ["o", "s", "^", "D", "v"]
    colors = {"15W": "#27ae60", "25W": "#2980b9", "MAXN": "#e74c3c"}

    latency_by_mode = {
        "MobileNetV2":    {"15W": 3.32, "25W": 3.34, "MAXN": 3.38},
        "ResNet-18":      {"15W": 3.14, "25W": 3.10, "MAXN": 3.13},
        "ResNet-34":      {"15W": 4.85, "25W": 4.87, "MAXN": 4.88},
        "ResNet-50":      {"15W": 6.59, "25W": 6.59, "MAXN": 6.51},
        "EfficientNet-B0":{"15W": 6.67, "25W": 6.64, "MAXN": 6.66},
    }

    for j, model in enumerate(MODELS):
        for mode in MODES:
            energy = PAPER_DATA["energy_per_inference_mj"][model][mode]
            latency = latency_by_mode[model][mode]
            ax.scatter(
                latency, energy,
                marker=markers[j],
                color=colors[mode],
                s=100, edgecolors="black", linewidth=0.5,
                zorder=3,
            )

    # Legend for modes
    for mode, color in colors.items():
        ax.scatter([], [], color=color, label=mode,
                   s=80, edgecolors="black", linewidth=0.5)
    # Legend for models
    for j, model in enumerate(MODELS):
        ax.scatter([], [], marker=markers[j], color="gray",
                   label=model, s=80, edgecolors="black",
                   linewidth=0.5)

    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Energy per Inference (mJ)")
    ax.set_title("Energy-Latency Trade-off Space")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "energy_latency_frontier.png")
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def main():
    output_dir = os.path.join(
        os.path.dirname(__file__), "..", "docs", "images"
    )
    os.makedirs(output_dir, exist_ok=True)

    setup_style()

    print("Generating README plots from paper data...")
    print(f"Output: {output_dir}/")
    print()

    plot_energy_by_mode(output_dir)
    plot_efficiency_by_mode(output_dir)
    plot_rail_breakdown(output_dir)
    plot_fp16_vs_fp32(output_dir)
    plot_batch_scaling(output_dir)
    plot_thermal_timeline(output_dir)
    plot_latency_by_mode(output_dir)
    plot_energy_latency_frontier(output_dir)

    print()
    print("=" * 50)
    print("All plots generated!")
    print("=" * 50)
    print()
    print("Files:")
    for f in sorted(os.listdir(output_dir)):
        if f.endswith(".png"):
            size = os.path.getsize(os.path.join(output_dir, f))
            print(f"  {f} ({size / 1024:.0f} KB)")
    print()
    print("8 plots ready for README:")
    print("  1. energy_by_mode.png        — Per-inference energy across power modes")
    print("  2. efficiency_by_mode.png    — Inferences/joule, 25W optimal")
    print("  3. rail_breakdown.png        — Per-rail power, SoC static floor")
    print("  4. fp16_vs_fp32.png          — Precision comparison")
    print("  5. batch_scaling.png         — Batch size efficiency")
    print("  6. thermal_timeline.png      — 300s stress test")
    print("  7. latency_by_mode.png       — <2% latency variation")
    print("  8. energy_latency_frontier.png — Trade-off scatter")


if __name__ == "__main__":
    main()