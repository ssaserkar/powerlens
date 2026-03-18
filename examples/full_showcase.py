"""
PowerLens Full Showcase

Demonstrates ALL PowerLens features with real GPU stress:
1. Light vs Medium vs Heavy model comparison
2. Iteration scaling analysis
3. Sustained thermal stress test with GPU monitoring

Usage:
    cd examples/
    python3 create_demo_model.py
    python3 full_showcase.py
"""

import csv
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def save_csv(filepath, headers, rows):
    """Helper to save CSV data."""
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for row in rows:
            writer.writerow(row)
    print(f"  Saved: {filepath}")


def main():
    import powerlens
    from powerlens.sensors.auto import detect_sensor
    from powerlens.profiler.session import PowerLensContext
    from powerlens.profiler.tensorrt_runner import (
        build_engine_for_batch_size,
        run_trt_inference,
    )
    from powerlens.analysis.thermal import ThermalMonitor
    from powerlens.analysis.batch_scaling import (
        BatchResult, BatchScalingReport, find_sweet_spot,
    )
    from powerlens.sensors.gpu_monitor import GpuMonitor
    from powerlens.export.csv_export import export_summary_csv, export_raw_csv
    from powerlens.visualization.plots import plot_power_trace

    output_dir = "showcase_results"
    os.makedirs(output_dir, exist_ok=True)

    print(f"PowerLens v{powerlens.__version__} — Full Showcase")
    print("=" * 60)
    print()

    # Check models exist
    models = {
        "light": "demo_light.onnx",
        "medium": "demo_medium.onnx",
        "heavy": "demo_heavy.onnx",
    }
    for name, path in models.items():
        if not os.path.exists(path):
            print(f"ERROR: {path} not found. Run create_demo_model.py first.")
            sys.exit(1)

    sensor = detect_sensor(use_mock_fallback=False)
    print(f"Power sensor: {type(sensor).__name__}")
    print()

    # ========================================
    # Part 1: Light vs Medium vs Heavy
    # ========================================
    print("PART 1: Model Complexity vs Energy")
    print("-" * 60)

    model_results = {}

    for name, path in models.items():
        print(f"\n  Building {name} model...", flush=True)
        engine = build_engine_for_batch_size(path, batch_size=1)

        test_ts = run_trt_inference(engine, num_runs=1, warmup=5, iterations_per_run=1)
        single_latency = test_ts[0][1] - test_ts[0][0]
        ipr = max(1, int(0.1 / single_latency))

        print(f"  {name}: {single_latency*1000:.1f}ms/inference, {ipr} iters/run", flush=True)

        thermal = ThermalMonitor(sample_interval_s=0.5)
        gpu_mon = GpuMonitor(sample_interval_s=0.1)

        ctx = PowerLensContext(sensor=sensor, sample_rate_hz=100)

        if thermal.available:
            thermal.start()
        if gpu_mon.available:
            gpu_mon.start()

        with ctx:
            timestamps = run_trt_inference(engine, num_runs=30, warmup=5,
                                           iterations_per_run=ipr)
            for start, end in timestamps:
                ctx._inference_timestamps.append((start, end))

        if thermal.available:
            thermal.stop()
        if gpu_mon.available:
            gpu_mon.stop()

        report = ctx.report()
        samples = ctx._sampler.get_samples()
        energy_per_inf = report.mean_energy_j / ipr

        gpu_summary = gpu_mon.get_summary()

        model_results[name] = {
            "latency_ms": single_latency * 1000,
            "energy_j": energy_per_inf,
            "power_w": report.mean_power_w,
            "peak_w": report.peak_power_w,
            "idle_w": report.idle_power_w,
            "gpu_util_pct": gpu_summary.get("gpu_util_avg_pct", 0),
            "gpu_util_max_pct": gpu_summary.get("gpu_util_max_pct", 0),
            "gpu_freq_mhz": gpu_summary.get("gpu_freq_avg_mhz", 0),
            "report": report,
        }

        print(f"  {name}: {energy_per_inf:.4f}J, {report.mean_power_w:.1f}W, GPU={gpu_summary.get('gpu_util_avg_pct', 0):.0f}%")

        # Save per-model data
        export_summary_csv(report, os.path.join(output_dir, f"part1_{name}_summary.csv"))
        export_raw_csv(samples, os.path.join(output_dir, f"part1_{name}_raw.csv"))
        plot_power_trace(samples, report,
                         os.path.join(output_dir, f"part1_{name}_power_trace.png"),
                         f"PowerLens — {name} model")

        del engine
        time.sleep(1.0)

    # Print comparison table
    print(f"\n{'Model':>10s} {'Latency':>10s} {'Energy/inf':>12s} {'Avg Power':>12s} {'Peak Power':>12s} {'GPU Util':>10s}")
    print("-" * 70)
    for name in ["light", "medium", "heavy"]:
        r = model_results[name]
        print(f"{name:>10s} {r['latency_ms']:>9.1f}ms {r['energy_j']:>11.4f}J {r['power_w']:>11.2f}W {r['peak_w']:>11.2f}W {r['gpu_util_pct']:>8.0f}%")

    if model_results["light"]["energy_j"] > 0:
        ratio = model_results["heavy"]["energy_j"] / model_results["light"]["energy_j"]
        print(f"\n→ Heavy model uses {ratio:.1f}x more energy per inference than light model")

    # Save comparison CSV
    save_csv(
        os.path.join(output_dir, "part1_model_comparison.csv"),
        ["model", "latency_ms", "energy_per_inference_j", "avg_power_w",
         "peak_power_w", "idle_power_w", "gpu_util_avg_pct", "gpu_freq_avg_mhz"],
        [[name, f"{r['latency_ms']:.4f}", f"{r['energy_j']:.6f}",
          f"{r['power_w']:.4f}", f"{r['peak_w']:.4f}", f"{r['idle_w']:.4f}",
          f"{r['gpu_util_pct']:.1f}", f"{r['gpu_freq_mhz']:.0f}"]
         for name, r in model_results.items()]
    )

    # ========================================
    # Part 2: Iteration Scaling with Heavy Model
    # ========================================
    print(f"\n\nPART 2: Iteration Scaling (Heavy Model)")
    print("-" * 60)

    engine = build_engine_for_batch_size("demo_heavy.onnx", batch_size=1)
    test_ts = run_trt_inference(engine, num_runs=1, warmup=5, iterations_per_run=1)
    single_latency = test_ts[0][1] - test_ts[0][0]

    iter_counts = [1, 5, 10, 50, 100]
    batch_results = []

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
        throughput = iters / (single_latency * iters)
        efficiency = iters * (run_ipr / iters) / r.mean_energy_j if r.mean_energy_j > 0 else 0

        result = BatchResult(
            batch_size=iters,
            latency_ms=single_latency * iters * 1000,
            energy_per_batch_j=energy_per_window,
            energy_per_inference_j=energy_per_inf,
            avg_power_w=r.mean_power_w,
            peak_power_w=r.peak_power_w,
            throughput_inf_per_s=throughput,
            efficiency_inf_per_j=efficiency,
        )
        batch_results.append(result)

        print(f"  {iters:>3d} iters: {energy_per_inf:.4f} J/inf, {r.mean_power_w:.1f}W, GPU={gpu_s.get('gpu_util_avg_pct', 0):.0f}%")
        time.sleep(0.5)

    best_eff = max(batch_results, key=lambda x: x.efficiency_inf_per_j)
    scaling_report = BatchScalingReport(
        model_name="demo_heavy.onnx",
        results=batch_results,
        best_efficiency=best_eff,
        best_latency=min(batch_results, key=lambda x: x.latency_ms),
        sweet_spot=find_sweet_spot(batch_results),
    )
    print(scaling_report.summary())

    # Save scaling CSV
    save_csv(
        os.path.join(output_dir, "part2_iteration_scaling.csv"),
        ["iterations", "latency_ms", "energy_per_inference_j",
         "avg_power_w", "efficiency_inf_per_j"],
        [[r.batch_size, f"{r.latency_ms:.4f}", f"{r.energy_per_inference_j:.6f}",
          f"{r.avg_power_w:.4f}", f"{r.efficiency_inf_per_j:.4f}"]
         for r in batch_results]
    )

    # ========================================
    # Part 3: Sustained Thermal Stress
    # ========================================
    print("PART 3: Sustained GPU Stress Test (90 seconds)")
    print("-" * 60)

    thermal = ThermalMonitor(sample_interval_s=1.0)
    gpu_mon = GpuMonitor(sample_interval_s=0.5)
    sustained_ipr = max(1, int(1.0 / single_latency))

    if thermal.available:
        thermal.start()
    if gpu_mon.available:
        gpu_mon.start()

    print(f"Running heavy model continuously for 90 seconds...")
    print(f"({sustained_ipr} iterations per run)")
    print()

    ctx = PowerLensContext(sensor=sensor, sample_rate_hz=50)
    start_time = time.monotonic()

    timeline_data = []

    with ctx:
        run_count = 0
        while time.monotonic() - start_time < 90:
            ts = run_trt_inference(engine, num_runs=1, warmup=0,
                                   iterations_per_run=sustained_ipr)
            for start, end in ts:
                ctx._inference_timestamps.append((start, end))
            run_count += 1

            elapsed = time.monotonic() - start_time

            if run_count % 5 == 0:
                temps = thermal.read_once() if thermal.available else []
                gpu_temp = next((t.temperature_c for t in temps if "gpu" in t.zone_name), 0)
                cpu_temp = next((t.temperature_c for t in temps if "cpu" in t.zone_name), 0)

                gpu_sample = gpu_mon.read_once() if gpu_mon.available else None
                gpu_util = gpu_sample.gpu_util_pct if gpu_sample else 0
                gpu_freq = gpu_sample.gpu_freq_mhz if gpu_sample else 0

                latest_samples = ctx._sampler.get_samples()
                power = sum(s.power_w for s in latest_samples[-1]) if latest_samples else 0

                timeline_data.append({
                    "time_s": elapsed,
                    "gpu_temp_c": gpu_temp,
                    "cpu_temp_c": cpu_temp,
                    "power_w": power,
                    "gpu_util_pct": gpu_util,
                    "gpu_freq_mhz": gpu_freq,
                    "runs": run_count,
                })

                print(f"  [{elapsed:5.0f}s] GPU: {gpu_temp:.1f}°C | Power: {power:.1f}W | GPU util: {gpu_util:.0f}% | Freq: {gpu_freq:.0f}MHz | Runs: {run_count}")

    if thermal.available:
        thermal.stop()
    if gpu_mon.available:
        gpu_mon.stop()

    sustained_report = ctx.report()
    samples = ctx._sampler.get_samples()

    print(f"\nSustained load results:")
    print(f"  Duration: 90 seconds")
    print(f"  Total runs: {sustained_report.num_inferences}")
    print(f"  Avg power: {sustained_report.mean_power_w:.2f}W")
    print(f"  Peak power: {sustained_report.peak_power_w:.2f}W")
    print(f"  Idle power: {sustained_report.idle_power_w:.2f}W")
    power_increase = sustained_report.mean_power_w - sustained_report.idle_power_w
    power_pct = (sustained_report.mean_power_w / sustained_report.idle_power_w - 1) * 100 if sustained_report.idle_power_w > 0 else 0
    print(f"  Power increase: {power_increase:.2f}W ({power_pct:.0f}%)")

    if thermal.available:
        thermal_report = thermal.analyze(sustained_report, throttle_temp_c=85.0)
        print(thermal_report.summary())

    if gpu_mon.available:
        print(gpu_mon.format_summary())

    if timeline_data:
        first_temp = timeline_data[0]["gpu_temp_c"]
        last_temp = timeline_data[-1]["gpu_temp_c"]
        print(f"  GPU temp rise: {first_temp:.1f}°C → {last_temp:.1f}°C (+{last_temp - first_temp:.1f}°C)")

    # Save sustained load data
    export_summary_csv(sustained_report, os.path.join(output_dir, "part3_sustained_summary.csv"))
    export_raw_csv(samples, os.path.join(output_dir, "part3_sustained_raw.csv"))
    plot_power_trace(samples, sustained_report,
                     os.path.join(output_dir, "part3_sustained_power_trace.png"),
                     "PowerLens — 90s Sustained GPU Load")

    # Save timeline CSV
    if timeline_data:
        save_csv(
            os.path.join(output_dir, "part3_timeline.csv"),
            ["time_s", "gpu_temp_c", "cpu_temp_c", "power_w",
             "gpu_util_pct", "gpu_freq_mhz", "runs"],
            [[f"{d['time_s']:.1f}", f"{d['gpu_temp_c']:.1f}", f"{d['cpu_temp_c']:.1f}",
              f"{d['power_w']:.2f}", f"{d['gpu_util_pct']:.0f}", f"{d['gpu_freq_mhz']:.0f}",
              d["runs"]]
             for d in timeline_data]
        )

    del engine

    # ========================================
    # Summary
    # ========================================
    print()
    print("=" * 60)
    print("SHOWCASE COMPLETE")
    print("=" * 60)
    print(f"\nResults saved to: {os.path.abspath(output_dir)}/")
    print()
    print("Key findings:")
    print(f"  Light model:  {model_results['light']['latency_ms']:.1f}ms, {model_results['light']['energy_j']:.4f}J/inf, GPU={model_results['light']['gpu_util_pct']:.0f}%")
    print(f"  Medium model: {model_results['medium']['latency_ms']:.1f}ms, {model_results['medium']['energy_j']:.4f}J/inf, GPU={model_results['medium']['gpu_util_pct']:.0f}%")
    print(f"  Heavy model:  {model_results['heavy']['latency_ms']:.1f}ms, {model_results['heavy']['energy_j']:.4f}J/inf, GPU={model_results['heavy']['gpu_util_pct']:.0f}%")
    if model_results["light"]["energy_j"] > 0:
        ratio = model_results["heavy"]["energy_j"] / model_results["light"]["energy_j"]
        print(f"  Heavy/Light energy ratio: {ratio:.1f}x")
    print(f"  Sustained load: {sustained_report.mean_power_w:.1f}W avg for 90s")
    if thermal.available and timeline_data:
        print(f"  GPU temp: {timeline_data[0]['gpu_temp_c']:.1f}°C → {timeline_data[-1]['gpu_temp_c']:.1f}°C")
        print(f"  Throttling: {'YES ⚠' if thermal_report.throttling_detected else 'NO ✓'}")

    # List all saved files
    print(f"\nSaved files:")
    for f in sorted(os.listdir(output_dir)):
        size = os.path.getsize(os.path.join(output_dir, f))
        print(f"  {f} ({size/1024:.1f} KB)")


if __name__ == "__main__":
    main()