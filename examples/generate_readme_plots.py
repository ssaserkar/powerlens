"""
Generate plots for the PowerLens README.

Run this after running full_showcase.py to generate
publication-ready comparison charts.

Usage:
    cd examples/
    python create_demo_model.py
    python generate_readme_plots.py
"""

import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    import powerlens
    from powerlens.sensors.auto import detect_sensor
    from powerlens.profiler.session import PowerLensContext
    from powerlens.profiler.tensorrt_runner import (
        build_engine_for_batch_size,
        run_trt_inference,
    )
    from powerlens.analysis.thermal import ThermalMonitor
    from powerlens.sensors.gpu_monitor import GpuMonitor
    from powerlens.analysis.thermal import (
            ThermalMonitor,
            discover_thermal_zones,
            read_temperatures,
        )

    # Check thermal availability once
    thermal_zones = discover_thermal_zones()
    thermal_available = len(thermal_zones) > 0
    if thermal_available:
        temps = read_temperatures(thermal_zones)
        gpu_temp = next((t.temperature_c for t in temps if "gpu" in t.zone_name), 0)
        print(f"Starting GPU temperature: {gpu_temp:.1f}°C")
    output_dir = "readme_plots"
    os.makedirs(output_dir, exist_ok=True)

    models = {
        "light": {"path": "demo_light.onnx", "desc": "2 blocks, 64ch, 0.5MB"},
        "medium": {"path": "demo_medium.onnx", "desc": "4 blocks, 256ch, 11MB"},
        "heavy": {"path": "demo_heavy.onnx", "desc": "8 blocks, 512ch, 81MB"},
    }

    for name, info in models.items():
        if not os.path.exists(info["path"]):
            print(f"ERROR: {info['path']} not found. Run create_demo_model.py first.")
            sys.exit(1)

    sensor = detect_sensor(use_mock_fallback=False)
    print(f"PowerLens v{powerlens.__version__}")
    print(f"Sensor: {type(sensor).__name__}")
    print()

    # ========================================
    # Collect data for all models
    # ========================================
    results = {}

    for name, info in models.items():
        print(f"Profiling {name} model ({info['desc']})...", flush=True)

        # Wait for thermal cooldown before each test
        if thermal_available:
            print("  Waiting for thermal cooldown...", end="", flush=True)
            cooldown_target = 40.0  # Target temperature before starting
            cooldown_timeout = 120  # Max wait seconds
            cooldown_start = time.monotonic()

            while time.monotonic() - cooldown_start < cooldown_timeout:
                temps = read_temperatures(thermal_zones)
                gpu_temp = next((t.temperature_c for t in temps if "gpu" in t.zone_name), 0)
                if gpu_temp <= cooldown_target:
                    print(f" GPU at {gpu_temp:.1f}°C ✓")
                    break
                print(f" {gpu_temp:.1f}°C", end="", flush=True)
                time.sleep(5)
            else:
                temps = read_temperatures(thermal_zones)
                gpu_temp = next((t.temperature_c for t in temps if "gpu" in t.zone_name), 0)
                print(f" timeout at {gpu_temp:.1f}°C (proceeding anyway)")
        engine = build_engine_for_batch_size(info["path"], batch_size=1)

        test_ts = run_trt_inference(engine, num_runs=1, warmup=5, iterations_per_run=1)
        single_latency = test_ts[0][1] - test_ts[0][0]
        ipr = max(1, int(0.15 / single_latency))

        thermal = ThermalMonitor(sample_interval_s=0.5)
        gpu_mon = GpuMonitor(sample_interval_s=0.1)

        ctx = PowerLensContext(sensor=sensor, sample_rate_hz=100)

        if thermal.available:
            thermal.start()
        if gpu_mon.available:
            gpu_mon.start()

        with ctx:
            timestamps = run_trt_inference(engine, num_runs=40, warmup=5,
                                           iterations_per_run=ipr)
            for start, end in timestamps:
                ctx._inference_timestamps.append((start, end))

        if thermal.available:
            thermal.stop()
        if gpu_mon.available:
            gpu_mon.stop()

        report = ctx.report()
        samples = ctx._sampler.get_samples()
        gpu_summary = gpu_mon.get_summary()
        thermal_report = thermal.analyze(report) if thermal.available else None

        gpu_temp_max = 0
        if thermal_report:
            gpu_temp_max = thermal_report.max_temperatures.get("gpu-thermal", 0)

        results[name] = {
            "latency_ms": single_latency * 1000,
            "energy_j": report.mean_energy_j / ipr,
            "power_w": report.mean_power_w,
            "peak_w": report.peak_power_w,
            "idle_w": report.idle_power_w,
            "gpu_util": gpu_summary.get("gpu_util_avg_pct", 0),
            "gpu_freq": gpu_summary.get("gpu_freq_avg_mhz", 0),
            "gpu_temp_max": gpu_temp_max,
            "samples": samples,
            "report": report,
            "desc": info["desc"],
            "ipr": ipr,
        }

        print(f"  {name}: {single_latency*1000:.1f}ms, {report.mean_energy_j/ipr:.4f}J, "
              f"{report.mean_power_w:.1f}W, GPU={gpu_summary.get('gpu_util_avg_pct', 0):.0f}%, "
              f"Tmax={gpu_temp_max:.1f}°C")

        del engine
        time.sleep(1.0)

    # ========================================
    # Plot 1: Model Comparison Bar Chart
    # ========================================
    print("\nGenerating plots...")

    model_names = ["light", "medium", "heavy"]
    colors = ["#2ecc71", "#f39c12", "#e74c3c"]

    fig, axes = plt.subplots(1, 5, figsize=(20, 5))

    # Energy per inference
    energies = [results[m]["energy_j"] for m in model_names]
    axes[0].bar(model_names, energies, color=colors, edgecolor="black", linewidth=0.5)
    axes[0].set_ylabel("Energy per Inference (J)")
    axes[0].set_title("Energy / Inference")
    for i, v in enumerate(energies):
        axes[0].text(i, v + max(energies) * 0.02, f"{v:.3f}J", ha="center", fontsize=9)

    # Average power
    powers = [results[m]["power_w"] for m in model_names]
    axes[1].bar(model_names, powers, color=colors, edgecolor="black", linewidth=0.5)
    axes[1].set_ylabel("Average Power (W)")
    axes[1].set_title("Average Power")
    for i, v in enumerate(powers):
        axes[1].text(i, v + max(powers) * 0.02, f"{v:.1f}W", ha="center", fontsize=9)

    # Latency
    latencies = [results[m]["latency_ms"] for m in model_names]
    axes[2].bar(model_names, latencies, color=colors, edgecolor="black", linewidth=0.5)
    axes[2].set_ylabel("Latency (ms)")
    axes[2].set_title("Inference Latency")
    for i, v in enumerate(latencies):
        axes[2].text(i, v + max(latencies) * 0.02, f"{v:.1f}ms", ha="center", fontsize=9)

    # GPU utilization
    gpu_utils = [results[m]["gpu_util"] for m in model_names]
    axes[3].bar(model_names, gpu_utils, color=colors, edgecolor="black", linewidth=0.5)
    axes[3].set_ylabel("GPU Utilization (%)")
    axes[3].set_title("GPU Utilization")
    axes[3].set_ylim(0, 100)
    for i, v in enumerate(gpu_utils):
        axes[3].text(i, v + 2, f"{v:.0f}%", ha="center", fontsize=9)

    # Max GPU temperature
    gpu_temps = [results[m]["gpu_temp_max"] for m in model_names]
    axes[4].bar(model_names, gpu_temps, color=colors, edgecolor="black", linewidth=0.5)
    axes[4].set_ylabel("Max GPU Temp (°C)")
    axes[4].set_title("GPU Temperature")
    for i, v in enumerate(gpu_temps):
        axes[4].text(i, v + 0.3, f"{v:.1f}°C", ha="center", fontsize=9)

    fig.suptitle("PowerLens — Model Complexity vs Energy (Jetson Orin Nano)", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_dir}/model_comparison.png")


    # ========================================
    # Plot 1b: Individual Power Traces (separate files)
    # ========================================
    for name, color in zip(model_names, colors):
        samples = results[name]["samples"]
        if not samples:
            continue

        t0 = samples[0][0].timestamp
        times = [(cycle[0].timestamp - t0) for cycle in samples]

        # Per-rail power
        rail_data = {}
        for cycle in samples:
            for s in cycle:
                if s.rail_name not in rail_data:
                    rail_data[s.rail_name] = {"times": [], "power": []}
                rail_data[s.rail_name]["times"].append(cycle[0].timestamp - t0)
                rail_data[s.rail_name]["power"].append(s.power_w)

        total_power = [sum(s.power_w for s in cycle) for cycle in samples]

        fig, ax = plt.subplots(figsize=(12, 6))

        # Total power
        ax.plot(times, total_power, color="black", linewidth=1.8, label="Total Power", zorder=3)

        # Per-rail
        rail_colors = {"VDD_IN": "#e74c3c", "VDD_CPU_GPU_CV": "#3498db", "VDD_SOC": "#2ecc71"}
        for rail_name, data in sorted(rail_data.items()):
            rc = rail_colors.get(rail_name, "#999999")
            ax.fill_between(data["times"], data["power"], alpha=0.15, color=rc)
            ax.plot(data["times"], data["power"], color=rc, linewidth=1.2,
                    alpha=0.8, label=rail_name)

        # Highlight inference regions
        report = results[name]["report"]
        if report and report.inferences:
            for inf in report.inferences[:10]:  # Show first 10 to avoid clutter
                start = inf.start_time - t0
                end = inf.end_time - t0
                ax.axvspan(start, end, alpha=0.08, color=color)

        r = results[name]

        # Dynamic Y-axis: pad 15% above max
        max_power = max(total_power) if total_power else 10
        ax.set_ylim(0, max_power * 1.15)

        # Add stats box
        textstr = (
            f"Model: {name} ({r['desc']})\n"
            f"Latency: {r['latency_ms']:.1f} ms\n"
            f"Energy/inf: {r['energy_j']:.4f} J\n"
            f"Avg power: {r['power_w']:.1f} W\n"
            f"Peak power: {r['peak_w']:.1f} W\n"
            f"Idle power: {r['idle_w']:.1f} W\n"
            f"GPU util: {r['gpu_util']:.0f}%\n"
            f"GPU temp: {r['gpu_temp_max']:.1f}°C"
        )
        props = dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.85)
        ax.text(0.02, 0.97, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment="top", bbox=props, family="monospace")

        # Add idle baseline
        idle = r["idle_w"]
        ax.axhline(y=idle, color="gray", linestyle="--", alpha=0.4, linewidth=1)
        ax.annotate(f"Idle: {idle:.1f}W", xy=(times[-1] * 0.75, idle + max_power * 0.02),
                    fontsize=9, color="gray")

        # Add average line
        avg = r["power_w"]
        ax.axhline(y=avg, color=color, linestyle=":", alpha=0.5, linewidth=1)
        ax.annotate(f"Avg: {avg:.1f}W", xy=(times[-1] * 0.75, avg + max_power * 0.02),
                    fontsize=9, color=color)

        ax.set_xlabel("Time (seconds)", fontsize=12)
        ax.set_ylabel("Power (watts)", fontsize=12)
        ax.set_title(
            f"PowerLens — {name.upper()} Model Power Trace (Jetson Orin Nano)",
            fontsize=14
        )
        ax.legend(fontsize=10, loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(times[0], times[-1])

        plt.tight_layout()
        filepath = os.path.join(output_dir, f"power_trace_{name}.png")
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {filepath}")
    # ========================================
    # Plot 2: Power Traces Overlaid
    # ========================================
    fig, ax = plt.subplots(figsize=(14, 6))

    for name, color in zip(model_names, colors):
        samples = results[name]["samples"]
        if not samples:
            continue

        t0 = samples[0][0].timestamp
        times = [(cycle[0].timestamp - t0) for cycle in samples]
        total_power = [sum(s.power_w for s in cycle) for cycle in samples]

        ax.plot(times, total_power, color=color, linewidth=1.2, alpha=0.8,
                label=f"{name} ({results[name]['desc']})")

    ax.set_xlabel("Time (seconds)", fontsize=12)
    ax.set_ylabel("Total Power (W)", fontsize=12)
    ax.set_title("PowerLens — Real-Time Power Traces (Jetson Orin Nano)", fontsize=14)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # Add idle baseline annotation
    idle_power = results["light"]["idle_w"]
    ax.axhline(y=idle_power, color="gray", linestyle="--", alpha=0.5)
    ax.annotate(f"Idle: {idle_power:.1f}W", xy=(0.5, idle_power),
                fontsize=9, color="gray")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "power_traces.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_dir}/power_traces.png")

    # ========================================
    # Plot 3: Energy Scaling with Iterations
    # ========================================
    print("\nProfiling iteration scaling (heavy model)...")


    # Cooldown before iteration scaling
    if thermal_available:
        print("  Waiting for thermal cooldown...", end="", flush=True)
        cooldown_start = time.monotonic()
        while time.monotonic() - cooldown_start < 120:
            temps = read_temperatures(thermal_zones)
            gpu_temp = next((t.temperature_c for t in temps if "gpu" in t.zone_name), 0)
            if gpu_temp <= 40.0:
                print(f" GPU at {gpu_temp:.1f}°C ✓")
                break
            print(f" {gpu_temp:.1f}°C", end="", flush=True)
            time.sleep(5)
        else:
            print(" timeout (proceeding)")
    
    engine = build_engine_for_batch_size("demo_heavy.onnx", batch_size=1)
    test_ts = run_trt_inference(engine, num_runs=1, warmup=5, iterations_per_run=1)
    single_latency = test_ts[0][1] - test_ts[0][0]

    iter_counts = [1, 5, 10, 25, 50, 100]
    iter_energy = []
    iter_power = []
    iter_gpu = []

    for iters in iter_counts:
        run_ipr = max(iters, ((int(0.1 / single_latency) + iters - 1) // iters) * iters)

        gpu_mon = GpuMonitor(sample_interval_s=0.1)
        ctx = PowerLensContext(sensor=sensor, sample_rate_hz=100)

        if gpu_mon.available:
            gpu_mon.start()

        with ctx:
            ts = run_trt_inference(engine, num_runs=15, warmup=3,
                                   iterations_per_run=run_ipr)
            for start, end in ts:
                ctx._inference_timestamps.append((start, end))

        if gpu_mon.available:
            gpu_mon.stop()

        r = ctx.report()
        gpu_s = gpu_mon.get_summary()

        energy_per_window = r.mean_energy_j / (run_ipr / iters) if run_ipr > 0 else 0
        energy_per_inf = energy_per_window / iters if iters > 0 else 0

        iter_energy.append(energy_per_inf)
        iter_power.append(r.mean_power_w)
        iter_gpu.append(gpu_s.get("gpu_util_avg_pct", 0))

        print(f"  {iters:>3d} iters: {energy_per_inf:.4f}J, {r.mean_power_w:.1f}W, GPU={gpu_s.get('gpu_util_avg_pct', 0):.0f}%")
        time.sleep(0.5)

    del engine

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Energy + Power vs iterations
    color1 = "#e74c3c"
    color2 = "#3498db"

    ax1.plot(iter_counts, iter_energy, "o-", color=color1, linewidth=2, markersize=8, label="Energy/inference")
    ax1.set_xlabel("Iterations per Window", fontsize=12)
    ax1.set_ylabel("Energy per Inference (J)", fontsize=12, color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)

    ax1b = ax1.twinx()
    ax1b.plot(iter_counts, iter_power, "s--", color=color2, linewidth=2, markersize=8, label="Avg Power")
    ax1b.set_ylabel("Average Power (W)", fontsize=12, color=color2)
    ax1b.tick_params(axis="y", labelcolor=color2)

    ax1.set_title("Energy & Power vs Iteration Count", fontsize=13)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1b.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")

    # GPU utilization vs iterations
    ax2.plot(iter_counts, iter_gpu, "o-", color="#2ecc71", linewidth=2, markersize=8)
    ax2.fill_between(iter_counts, iter_gpu, alpha=0.2, color="#2ecc71")
    ax2.set_xlabel("Iterations per Window", fontsize=12)
    ax2.set_ylabel("GPU Utilization (%)", fontsize=12)
    ax2.set_title("GPU Utilization vs Iteration Count", fontsize=13)
    ax2.set_ylim(0, 100)
    ax2.grid(True, alpha=0.3)

    fig.suptitle("PowerLens — Iteration Scaling Analysis (Heavy Model, Jetson Orin Nano)", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "iteration_scaling.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_dir}/iteration_scaling.png")

    # ========================================
    # Plot 4: Sustained Load Timeline
    # ========================================
    
    print("\nRunning 150s sustained load for timeline plot...")

    # Cooldown before sustained test
    if thermal_available:
        print("  Waiting for thermal cooldown...", end="", flush=True)
        cooldown_start = time.monotonic()
        while time.monotonic() - cooldown_start < 120:
            temps = read_temperatures(thermal_zones)
            gpu_temp = next((t.temperature_c for t in temps if "gpu" in t.zone_name), 0)
            if gpu_temp <= 40.0:
                print(f" GPU at {gpu_temp:.1f}°C ✓")
                break
            print(f" {gpu_temp:.1f}°C", end="", flush=True)
            time.sleep(5)
        else:
            print(" timeout (proceeding)")
    
    engine = build_engine_for_batch_size("demo_heavy.onnx", batch_size=1)
    test_ts = run_trt_inference(engine, num_runs=1, warmup=5, iterations_per_run=1)
    single_latency = test_ts[0][1] - test_ts[0][0]
    sustained_ipr = max(1, int(1.0 / single_latency))

    thermal = ThermalMonitor(sample_interval_s=1.0)
    gpu_mon = GpuMonitor(sample_interval_s=0.5)

    timeline_time = []
    timeline_power = []
    timeline_gpu_temp = []
    timeline_gpu_util = []

    if thermal.available:
        thermal.start()
    if gpu_mon.available:
        gpu_mon.start()

    ctx = PowerLensContext(sensor=sensor, sample_rate_hz=50)
    start_time = time.monotonic()

    with ctx:
        run_count = 0
        while time.monotonic() - start_time < 150:
            ts = run_trt_inference(engine, num_runs=1, warmup=0,
                                   iterations_per_run=sustained_ipr)
            for start, end in ts:
                ctx._inference_timestamps.append((start, end))
            run_count += 1

            elapsed = time.monotonic() - start_time

            if run_count % 3 == 0:
                temps = thermal.read_once() if thermal.available else []
                gpu_temp = next((t.temperature_c for t in temps if "gpu" in t.zone_name), 0)
                gpu_sample = gpu_mon.read_once() if gpu_mon.available else None
                gpu_util = gpu_sample.gpu_util_pct if gpu_sample else 0

                latest = ctx._sampler.get_samples()
                power = sum(s.power_w for s in latest[-1]) if latest else 0

                timeline_time.append(elapsed)
                timeline_power.append(power)
                timeline_gpu_temp.append(gpu_temp)
                timeline_gpu_util.append(gpu_util)

                if run_count % 15 == 0:
                    print(f"  [{elapsed:.0f}s] GPU: {gpu_temp:.1f}°C, Power: {power:.1f}W")

    if thermal.available:
        thermal.stop()
    if gpu_mon.available:
        gpu_mon.stop()

    del engine

    # Plot timeline
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    ax1.plot(timeline_time, timeline_power, color="#e74c3c", linewidth=1.5)
    ax1.set_ylabel("Power (W)", fontsize=11)
    ax1.set_title("Sustained GPU Load — 150 Second Timeline (Jetson Orin Nano)", fontsize=14)
    ax1.grid(True, alpha=0.3)
    if timeline_power:
        ax1.axhline(y=np.mean(timeline_power), color="#e74c3c", linestyle="--", alpha=0.5)
        ax1.annotate(f"Avg: {np.mean(timeline_power):.1f}W",
                     xy=(timeline_time[-1] * 0.85, np.mean(timeline_power) + 0.5),
                     fontsize=10, color="#e74c3c")

    ax2.plot(timeline_time, timeline_gpu_temp, color="#f39c12", linewidth=1.5)
    ax2.set_ylabel("GPU Temperature (°C)", fontsize=11)
    ax2.grid(True, alpha=0.3)
    if timeline_gpu_temp:
        ax2.annotate(f"{timeline_gpu_temp[0]:.0f}°C → {timeline_gpu_temp[-1]:.0f}°C",
                     xy=(timeline_time[-1] * 0.7, max(timeline_gpu_temp) - 1),
                     fontsize=10, color="#f39c12")

    ax3.plot(timeline_time, timeline_gpu_util, color="#3498db", linewidth=1.5)
    ax3.set_ylabel("GPU Utilization (%)", fontsize=11)
    ax3.set_xlabel("Time (seconds)", fontsize=11)
    ax3.set_ylim(0, 100)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "sustained_timeline.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_dir}/sustained_timeline.png")

    # ========================================
    # Plot 4b: Sustained Load Per-Rail Power Trace
    # ========================================
    sustained_samples = ctx._sampler.get_samples()

    if sustained_samples:
        fig, ax = plt.subplots(figsize=(14, 7))

        t0 = sustained_samples[0][0].timestamp
        times = [(cycle[0].timestamp - t0) for cycle in sustained_samples]
        total_power = [sum(s.power_w for s in cycle) for cycle in sustained_samples]

        ax.plot(times, total_power, color="black", linewidth=1.5, label="Total Power", zorder=3)

        rail_data = {}
        for cycle in sustained_samples:
            for s in cycle:
                if s.rail_name not in rail_data:
                    rail_data[s.rail_name] = {"times": [], "power": []}
                rail_data[s.rail_name]["times"].append(cycle[0].timestamp - t0)
                rail_data[s.rail_name]["power"].append(s.power_w)

        rail_colors = {"VDD_IN": "#e74c3c", "VDD_CPU_GPU_CV": "#3498db", "VDD_SOC": "#2ecc71"}
        for rail_name, data in sorted(rail_data.items()):
            rc = rail_colors.get(rail_name, "#999999")
            ax.fill_between(data["times"], data["power"], alpha=0.15, color=rc)
            ax.plot(data["times"], data["power"], color=rc, linewidth=1.2,
                    alpha=0.8, label=rail_name)

        # Dynamic Y-axis
        max_power = max(total_power) if total_power else 10
        ax.set_ylim(0, max_power * 1.15)

        # Average line
        avg_power = np.mean(total_power)
        ax.axhline(y=avg_power, color="gray", linestyle=":", alpha=0.5)
        ax.annotate(f"Avg: {avg_power:.1f}W", xy=(times[-1] * 0.85, avg_power + max_power * 0.02),
                    fontsize=10, color="gray")

        # Temperature on secondary axis
        if timeline_time and timeline_gpu_temp:
            ax2 = ax.twinx()
            ax2.plot(timeline_time, timeline_gpu_temp, color="#f39c12",
                     linewidth=2.5, linestyle="--", label="GPU Temp", zorder=4)
            ax2.set_ylabel("GPU Temperature (°C)", fontsize=12, color="#f39c12")
            ax2.tick_params(axis="y", labelcolor="#f39c12")

            # Dynamic temp axis
            min_temp = min(timeline_gpu_temp) - 2
            max_temp = max(timeline_gpu_temp) + 2
            ax2.set_ylim(min_temp, max_temp)

            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=10)
        else:
            ax.legend(fontsize=10, loc="upper right")

        # Stats box
        sustained_report_final = ctx.report()
        temp_start = timeline_gpu_temp[0] if timeline_gpu_temp else 0
        temp_end = timeline_gpu_temp[-1] if timeline_gpu_temp else 0
        textstr = (
            f"Duration: 150 seconds\n"
            f"Total runs: {sustained_report_final.num_inferences}\n"
            f"Avg power: {avg_power:.1f} W\n"
            f"Peak power: {max(total_power):.1f} W\n"
            f"GPU temp: {temp_start:.0f}°C → {temp_end:.0f}°C\n"
            f"Temp rise: +{temp_end - temp_start:.1f}°C"
        )
        props = dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.85)
        ax.text(0.02, 0.97, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment="top", bbox=props, family="monospace")

        ax.set_xlabel("Time (seconds)", fontsize=12)
        ax.set_ylabel("Power (watts)", fontsize=12)
        ax.set_title(
            "PowerLens — 150s Sustained GPU Load with Thermal (Jetson Orin Nano)",
            fontsize=14
        )
        ax.grid(True, alpha=0.3)
        ax.set_xlim(times[0], times[-1])

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "sustained_power_trace.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {output_dir}/sustained_power_trace.png")

    

    # ========================================
    # Summary
    ## ========================================
    print()
    print("=" * 60)
    print("README plots generated!")
    print("=" * 60)
    print()
    print("Files:")
    for f in sorted(os.listdir(output_dir)):
        size = os.path.getsize(os.path.join(output_dir, f))
        print(f"  {output_dir}/{f} ({size/1024:.1f} KB)")
    print()
    print("Copy these to your repo root and reference in README.md:")
    print("  ![Model Comparison](readme_plots/model_comparison.png)")
    print("  ![Power Traces](readme_plots/power_traces.png)")
    print("  ![Iteration Scaling](readme_plots/iteration_scaling.png)")
    print("  ![Sustained Timeline](readme_plots/sustained_timeline.png)")


if __name__ == "__main__":
    main()