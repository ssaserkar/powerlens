"""
PowerLens command-line interface.

Usage:
    powerlens demo                          Mock sensor demo
    powerlens demo --real                   Real sensor with CPU stress
    powerlens detect                        Detect available sensors
    powerlens profile --onnx model.onnx     Profile TensorRT model
    powerlens compare a.onnx b.onnx         Compare two models
    powerlens --version                     Show version
"""

import argparse
import os
import sys
import time

import numpy as np

import powerlens
from powerlens.profiler.session import PowerLensContext
from powerlens.sensors.mock import MockSensor
from powerlens.export.csv_export import export_summary_csv, export_raw_csv
from powerlens.visualization.plots import plot_power_trace


def _cpu_stress(duration_s: float):
    """CPU-intensive workload for real power measurement."""
    end_time = time.monotonic() + duration_s
    while time.monotonic() < end_time:
        a = np.random.randn(200, 200).astype(np.float32)
        b = np.random.randn(200, 200).astype(np.float32)
        np.dot(a, b)


def cmd_demo(args):
    """Run a demo profiling session."""
    print(f"PowerLens v{powerlens.__version__}")
    print(f"Running demo: {args.runs} inferences")
    print()

    if args.real:
        from powerlens.sensors.auto import detect_sensor
        sensor = detect_sensor(use_mock_fallback=False)
        print(f"Using real sensor: {type(sensor).__name__}")
        print("Running CPU stress workload...")
    else:
        sensor = MockSensor()
        print("Using mock sensor (use --real on Jetson for real measurements)")

    print()

    ctx = PowerLensContext(sensor=sensor, sample_rate_hz=args.rate)

    with ctx:
        for i in range(args.runs):
            ctx.mark_inference_start()
            if isinstance(sensor, MockSensor):
                sensor.set_load(args.load)
                time.sleep(args.duration)
                sensor.set_load(0.0)
            else:
                _cpu_stress(args.duration)
            ctx.mark_inference_end()
            time.sleep(0.01)

    report = ctx.report()
    samples = ctx._sampler.get_samples()

    print(report.summary())

    if args.output:
        _export_results(report, samples, args.output, f"PowerLens Demo — {args.runs} inferences")


def cmd_detect(args):
    """Detect available power sensors."""
    from powerlens.sensors.auto import get_sensor_info

    print(f"PowerLens v{powerlens.__version__}")
    print("Detecting power sensors...")
    print()

    info = get_sensor_info()

    print(f"Platform:       {info['platform']}")
    print()
    print("Sensor backends:")
    print(f"  I2C (direct):   {'YES' if info['i2c_available'] else 'NO'}")
    print(f"    Detail:       {info['i2c_detail']}")
    print(f"  Sysfs (hwmon):  {'YES' if info['sysfs_available'] else 'NO'}")
    print(f"    Detail:       {info['sysfs_detail']}")
    print(f"  Mock (testing): {'YES' if info['mock_available'] else 'NO'}")
    print()
    print(f"Recommended:      {info['recommended']}")
    print()

    if info["recommended"] == "mock":
        print("No real sensor detected.")
        print("If you are on a Jetson device:")
        print("  1. Install smbus2: pip install smbus2")
        print("  2. Add I2C permissions: sudo usermod -aG i2c $USER")
        print("  3. Re-login and try again")
    elif info["recommended"] == "sysfs":
        print("Sysfs sensor available. Run with:")
        print("  powerlens demo --real")
        print("  powerlens profile --onnx model.onnx")
    elif info["recommended"] == "i2c":
        print("Direct I2C sensor available. Run with:")
        print("  powerlens demo --real")
        print("  powerlens profile --onnx model.onnx")


def cmd_profile(args):
    """Profile a TensorRT model."""
    print(f"PowerLens v{powerlens.__version__}")
    print()

    # Check TensorRT availability
    try:
        import tensorrt  # noqa: F401
    except ImportError:
        print("ERROR: TensorRT not found.")
        print("This command requires TensorRT (available on Jetson with JetPack).")
        sys.exit(1)

    from powerlens.profiler.tensorrt_runner import (
        build_engine_from_onnx,
        load_engine,
        get_engine_info,
        run_trt_inference,
    )
    from powerlens.sensors.auto import detect_sensor

    # Load or build engine
    if args.engine:
        engine = load_engine(args.engine)
        model_name = os.path.basename(args.engine)
    elif args.onnx:
        engine = build_engine_from_onnx(args.onnx)
        model_name = os.path.basename(args.onnx)
    else:
        print("ERROR: Specify --onnx or --engine")
        sys.exit(1)

    # Print model info
    info = get_engine_info(engine)
    print(f"Model: {model_name}")
    for inp in info["inputs"]:
        print(f"  Input:  {inp['name']} {inp['shape']}")
    for out in info["outputs"]:
        print(f"  Output: {out['name']} {out['shape']}")
    print()

    # Setup sensor
    sensor = detect_sensor(use_mock_fallback=False)
    print(f"Power sensor: {type(sensor).__name__}")

    # Auto-detect iterations per run
    ipr = args.iterations
    if ipr == 0:
        # Auto: run one inference to measure latency
        print("Auto-detecting iterations per run...")
        test_ts = run_trt_inference(engine, num_runs=1, warmup=3, iterations_per_run=1)
        single_latency = test_ts[0][1] - test_ts[0][0]
        # Target ~100ms per profiling run for good energy resolution
        ipr = max(1, int(0.1 / single_latency))
        print(f"  Single inference: {single_latency*1000:.1f} ms")
        print(f"  Using {ipr} iterations per run (~{ipr * single_latency * 1000:.0f} ms)")
    print()

    # Setup thermal monitoring
    from powerlens.analysis.thermal import ThermalMonitor
    thermal = ThermalMonitor(sample_interval_s=0.5)

    # Setup GPU utilization monitoring
    from powerlens.sensors.gpu_monitor import GpuMonitor
    gpu_monitor = GpuMonitor(sample_interval_s=0.1)

    # Profile
    ctx = PowerLensContext(sensor=sensor, sample_rate_hz=args.rate)

    print(f"Profiling {args.runs} runs ({ipr} iterations each)...")
    if thermal.available:
        print(f"Thermal monitoring: {len(thermal.zone_names)} zones")
        thermal.start()

    if gpu_monitor.available:
        print("GPU utilization monitoring: active")
        gpu_monitor.start()

    with ctx:
        timestamps = run_trt_inference(
            engine, num_runs=args.runs, warmup=args.warmup,
            iterations_per_run=ipr,
        )
        for start, end in timestamps:
            ctx._inference_timestamps.append((start, end))

    if thermal.available:
        thermal.stop()

    if gpu_monitor.available:
        gpu_monitor.stop()

    report = ctx.report()
    ctx._sampler.get_samples()

    # Adjust energy per single inference
    if ipr > 1:
        print(f"\nNote: Each run = {ipr} iterations")
        print(f"  Energy per run:        {report.mean_energy_j:.4f} J")
        print(f"  Energy per inference:  {report.mean_energy_j / ipr:.4f} J")
        print(f"  Latency per inference: {report.total_duration_s / report.num_inferences / ipr * 1000:.1f} ms")

    print(report.summary())
    # Thermal analysis
    thermal_report = None
    if thermal.available:
        thermal_report = thermal.analyze(
            energy_report=report,
            throttle_temp_c=85.0,
        )
        print(thermal_report.summary())

    if gpu_monitor.available:
        print(gpu_monitor.format_summary())

    # Per-inference details
    print("First 5 runs:")
    for inf in report.inferences[:5]:
        print(
            f"  #{inf.index}: {inf.energy_j:.4f} J, "
            f"{inf.duration_s*1000:.1f} ms, "
            f"{inf.avg_power_w:.2f} W avg"
        )
    print()

    if args.output:
        # Generate text report
        from powerlens.export.report import generate_text_report
        report_path = generate_text_report(
            report=report,
            thermal_report=thermal_report if thermal.available else None,
            gpu_summary=gpu_monitor.get_summary() if gpu_monitor.available else None,
            model_name=model_name,
            platform="Jetson Orin Nano",
            output_path=os.path.join(args.output, "powerlens_report.txt"),
        )
        print(f"Report:      {report_path}")


def cmd_compare(args):
    """Compare energy efficiency of two TensorRT models."""
    print(f"PowerLens v{powerlens.__version__}")
    print()

    try:
        import tensorrt  # noqa: F401
    except ImportError:
        print("ERROR: TensorRT not found.")
        sys.exit(1)

    from powerlens.profiler.tensorrt_runner import (
        build_engine_from_onnx,
        load_engine,
        run_trt_inference,
    )
    from powerlens.sensors.auto import detect_sensor

    sensor = detect_sensor(use_mock_fallback=False)
    print(f"Power sensor: {type(sensor).__name__}")
    print()

    results = []

    for model_path in [args.model_a, args.model_b]:
        model_name = os.path.basename(model_path)
        print(f"Profiling: {model_name}")

        # Load engine
        if model_path.endswith(".engine"):
            engine = load_engine(model_path)
        else:
            engine = build_engine_from_onnx(model_path)

        # Auto-detect iterations
        test_ts = run_trt_inference(engine, num_runs=1, warmup=3, iterations_per_run=1)
        single_latency = test_ts[0][1] - test_ts[0][0]
        ipr = max(1, int(0.1 / single_latency))
        print(f"  Latency: {single_latency*1000:.1f} ms, using {ipr} iterations/run")

        # Profile
        ctx = PowerLensContext(sensor=sensor, sample_rate_hz=args.rate)
        with ctx:
            timestamps = run_trt_inference(
                engine, num_runs=args.runs, warmup=args.warmup,
                iterations_per_run=ipr,
            )
            for start, end in timestamps:
                ctx._inference_timestamps.append((start, end))

        report = ctx.report()

        results.append({
            "name": model_name,
            "energy_per_run_j": report.mean_energy_j,
            "energy_per_inference_j": report.mean_energy_j / ipr,
            "avg_power_w": report.mean_power_w,
            "peak_power_w": report.peak_power_w,
            "latency_ms": single_latency * 1000,
            "efficiency_inf_per_j": ipr / report.mean_energy_j if report.mean_energy_j > 0 else 0,
            "ipr": ipr,
            "report": report,
        })

        print(f"  Energy/inference: {results[-1]['energy_per_inference_j']:.4f} J")
        print(f"  Avg power: {report.mean_power_w:.2f} W")
        print()

        # Pause between models for power to settle
        time.sleep(2.0)

    # Comparison table
    a = results[0]
    b = results[1]

    print()
    print("=" * 60)
    print("Model Comparison Report")
    print("=" * 60)
    print(f"{'':25s} {'Model A':>15s} {'Model B':>15s}")
    print(f"{'Name':25s} {a['name']:>15s} {b['name']:>15s}")
    print("-" * 60)
    print(f"{'Energy/inference (J)':25s} {a['energy_per_inference_j']:>15.4f} {b['energy_per_inference_j']:>15.4f}")
    print(f"{'Avg power (W)':25s} {a['avg_power_w']:>15.2f} {b['avg_power_w']:>15.2f}")
    print(f"{'Peak power (W)':25s} {a['peak_power_w']:>15.2f} {b['peak_power_w']:>15.2f}")
    print(f"{'Latency (ms)':25s} {a['latency_ms']:>15.1f} {b['latency_ms']:>15.1f}")
    print(f"{'Efficiency (inf/J)':25s} {a['efficiency_inf_per_j']:>15.1f} {b['efficiency_inf_per_j']:>15.1f}")
    print("-" * 60)

    # Winner
    if a['energy_per_inference_j'] < b['energy_per_inference_j']:
        ratio = b['energy_per_inference_j'] / a['energy_per_inference_j']
        print(f"\n→ {a['name']} is {ratio:.1f}x more energy efficient")
    else:
        ratio = a['energy_per_inference_j'] / b['energy_per_inference_j']
        print(f"\n→ {b['name']} is {ratio:.1f}x more energy efficient")

    if args.output:
        os.makedirs(args.output, exist_ok=True)
        # Save comparison as CSV
        import csv
        csv_path = os.path.join(args.output, "comparison.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "model_a", "model_b"])
            writer.writerow(["name", a["name"], b["name"]])
            writer.writerow(["energy_per_inference_j", f"{a['energy_per_inference_j']:.6f}", f"{b['energy_per_inference_j']:.6f}"])
            writer.writerow(["avg_power_w", f"{a['avg_power_w']:.4f}", f"{b['avg_power_w']:.4f}"])
            writer.writerow(["peak_power_w", f"{a['peak_power_w']:.4f}", f"{b['peak_power_w']:.4f}"])
            writer.writerow(["latency_ms", f"{a['latency_ms']:.4f}", f"{b['latency_ms']:.4f}"])
            writer.writerow(["efficiency_inf_per_j", f"{a['efficiency_inf_per_j']:.4f}", f"{b['efficiency_inf_per_j']:.4f}"])
        print(f"\nComparison CSV: {os.path.abspath(csv_path)}")

def cmd_power_modes(args):
    """Profile across power modes."""
    sys.stdout.reconfigure(line_buffering=True)
    print(f"PowerLens v{powerlens.__version__}")
    print()

    try:
        import tensorrt  # noqa: F401
    except ImportError:
        print("ERROR: TensorRT not found.")
        sys.exit(1)

    from powerlens.profiler.tensorrt_runner import (
        build_engine_from_onnx,
        load_engine,
        run_trt_inference,
    )
    from powerlens.sensors.auto import detect_sensor
    from powerlens.analysis.power_mode import (
        get_current_power_mode,
        get_available_modes,
        set_power_mode,
        PowerModeResult,
        PowerModeReport,
    )

    # Load model
    if args.onnx:
        engine = build_engine_from_onnx(args.onnx)
        model_name = os.path.basename(args.onnx)
    elif args.engine:
        engine = load_engine(args.engine)
        model_name = os.path.basename(args.engine)
    else:
        print("ERROR: Specify --onnx or --engine")
        sys.exit(1)

    print(f"Model: {model_name}")

    original_mode = get_current_power_mode()
    modes = get_available_modes()

    if not modes:
        print("ERROR: Could not detect power modes. Is nvpmodel available?")
        sys.exit(1)

    print(f"Current mode: {original_mode['name']} (ID={original_mode['id']})")
    print(f"Available modes: {[m['name'] for m in modes]}")
    print()
    print("NOTE: This requires sudo access to change power modes.")
    print()

    sensor = detect_sensor(use_mock_fallback=False)
    results = []

    for mode in modes:
        print(f"--- Mode: {mode['name']} (ID={mode['id']}) ---")

        if not set_power_mode(mode["id"]):
            print(f"  SKIPPED: Could not set mode {mode['id']}")
            continue

        print("  Waiting for clocks to stabilize...")
        time.sleep(3.0)

        test_ts = run_trt_inference(engine, num_runs=1, warmup=3, iterations_per_run=1)
        single_latency = test_ts[0][1] - test_ts[0][0]
        ipr = max(1, int(0.1 / single_latency))

        ctx = PowerLensContext(sensor=sensor, sample_rate_hz=args.rate)
        with ctx:
            timestamps = run_trt_inference(
                engine, num_runs=args.runs, warmup=args.warmup,
                iterations_per_run=ipr,
            )
            for start, end in timestamps:
                ctx._inference_timestamps.append((start, end))

        report = ctx.report()

        result = PowerModeResult(
            mode_id=mode["id"],
            mode_name=mode["name"],
            energy_per_inference_j=report.mean_energy_j / ipr if ipr > 0 else 0,
            avg_power_w=report.mean_power_w,
            peak_power_w=report.peak_power_w,
            latency_ms=single_latency * 1000,
            efficiency_inf_per_j=ipr / report.mean_energy_j if report.mean_energy_j > 0 else 0,
            num_inferences=report.num_inferences,
        )
        results.append(result)

        print(f"  Latency: {result.latency_ms:.1f}ms")
        print(f"  Energy/inf: {result.energy_per_inference_j:.4f}J")
        print(f"  Avg power: {result.avg_power_w:.2f}W")
        print()

        time.sleep(2.0)

    print(f"Restoring original mode: {original_mode['name']}")
    set_power_mode(original_mode["id"])

    if results:
        best_eff = max(results, key=lambda r: r.efficiency_inf_per_j)
        best_lat = min(results, key=lambda r: r.latency_ms)

        mode_report = PowerModeReport(
            results=results,
            best_efficiency=best_eff,
            best_latency=best_lat,
        )
        print(mode_report.summary())

        if args.output:
            os.makedirs(args.output, exist_ok=True)
            import csv
            csv_path = os.path.join(args.output, "power_modes.csv")
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "mode_id", "mode_name", "energy_per_inference_j",
                    "avg_power_w", "peak_power_w", "latency_ms", "efficiency_inf_per_j",
                ])
                for r in results:
                    writer.writerow([
                        r.mode_id, r.mode_name,
                        f"{r.energy_per_inference_j:.6f}",
                        f"{r.avg_power_w:.4f}",
                        f"{r.peak_power_w:.4f}",
                        f"{r.latency_ms:.4f}",
                        f"{r.efficiency_inf_per_j:.4f}",
                    ])
            print(f"Results CSV: {os.path.abspath(csv_path)}")

def _export_results(report, samples, output_dir, title):
    """Export report results to files."""
    os.makedirs(output_dir, exist_ok=True)

    csv_path = export_summary_csv(
        report, os.path.join(output_dir, "energy_summary.csv")
    )
    print(f"Summary CSV: {csv_path}")

    raw_path = export_raw_csv(
        samples, os.path.join(output_dir, "raw_samples.csv")
    )
    print(f"Raw CSV:     {raw_path}")

    if samples:
        plot_path = plot_power_trace(
            samples=samples,
            report=report,
            filepath=os.path.join(output_dir, "power_trace.png"),
            title=title,
        )
        print(f"Plot:        {plot_path}")
    print()

def cmd_batch_scaling(args):
    """Profile energy scaling across batch sizes."""
    sys.stdout.reconfigure(line_buffering=True)

    print(f"PowerLens v{powerlens.__version__}")
    print()

    try:
        import tensorrt  # noqa: F401
    except ImportError:
        print("ERROR: TensorRT not found.")
        sys.exit(1)

    from powerlens.sensors.auto import detect_sensor
    from powerlens.analysis.batch_scaling import (
        BatchResult,
        BatchScalingReport,
        find_sweet_spot,
    )

    batch_sizes = [int(b) for b in args.batches.split(",")]
    model_name = os.path.basename(args.onnx or args.engine or "model")

    print(f"Model: {model_name}")
    print(f"Batch sizes: {batch_sizes}")
    print()

    # We build a separate engine per batch size since ONNX may have fixed batch
    from powerlens.profiler.tensorrt_runner import (
        build_engine_for_batch_size,
        run_trt_inference,
    )

    sensor = detect_sensor(use_mock_fallback=False)
    print(f"Power sensor: {type(sensor).__name__}")
    print()

    results = []

    for batch_size in batch_sizes:
        print(f"--- Batch size: {batch_size} ---")

        # Build engine for this batch size
        if args.onnx:
            print(f"  Building engine (batch={batch_size})...")
            engine = build_engine_for_batch_size(args.onnx, batch_size)
        elif args.engine:
            from powerlens.profiler.tensorrt_runner import load_engine
            engine = load_engine(args.engine)
            print("  WARNING: Pre-built engine may not support this batch size")
        else:
            print("ERROR: Specify --onnx or --engine")
            sys.exit(1)

        # Auto-detect iterations
        test_ts = run_trt_inference(
            engine, num_runs=1, warmup=3, iterations_per_run=1,
        )
        single_latency = test_ts[0][1] - test_ts[0][0]
        ipr = max(1, int(0.1 / single_latency))

        # Profile
        ctx = PowerLensContext(sensor=sensor, sample_rate_hz=args.rate)
        with ctx:
            timestamps = run_trt_inference(
                engine, num_runs=args.runs, warmup=args.warmup,
                iterations_per_run=ipr,
            )
            for start, end in timestamps:
                ctx._inference_timestamps.append((start, end))

        report = ctx.report()

        latency_ms = single_latency * 1000
        energy_per_batch = report.mean_energy_j / ipr
        energy_per_inf = energy_per_batch / batch_size
        throughput = batch_size / single_latency
        efficiency = batch_size * ipr / report.mean_energy_j if report.mean_energy_j > 0 else 0

        result = BatchResult(
            batch_size=batch_size,
            latency_ms=latency_ms,
            energy_per_batch_j=energy_per_batch,
            energy_per_inference_j=energy_per_inf,
            avg_power_w=report.mean_power_w,
            peak_power_w=report.peak_power_w,
            throughput_inf_per_s=throughput,
            efficiency_inf_per_j=efficiency,
        )
        results.append(result)

        print(f"  Latency: {latency_ms:.1f}ms")
        print(f"  Energy/inf: {energy_per_inf:.4f}J")
        print(f"  Throughput: {throughput:.0f} inf/s")
        print()

        # Clean up engine for this batch size
        del engine
        time.sleep(1.0)

    # Generate report
    if results:
        best_eff = max(results, key=lambda r: r.efficiency_inf_per_j)
        best_lat = min(results, key=lambda r: r.latency_ms)
        sweet = find_sweet_spot(results)

        scaling_report = BatchScalingReport(
            model_name=model_name,
            results=results,
            best_efficiency=best_eff,
            best_latency=best_lat,
            sweet_spot=sweet,
        )
        print(scaling_report.summary())

        if args.output:
            os.makedirs(args.output, exist_ok=True)
            import csv
            csv_path = os.path.join(args.output, "batch_scaling.csv")
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "batch_size", "latency_ms", "energy_per_batch_j",
                    "energy_per_inference_j", "avg_power_w",
                    "throughput_inf_per_s", "efficiency_inf_per_j",
                ])
                for r in results:
                    writer.writerow([
                        r.batch_size,
                        f"{r.latency_ms:.4f}",
                        f"{r.energy_per_batch_j:.6f}",
                        f"{r.energy_per_inference_j:.6f}",
                        f"{r.avg_power_w:.4f}",
                        f"{r.throughput_inf_per_s:.2f}",
                        f"{r.efficiency_inf_per_j:.4f}",
                    ])
            print(f"Results CSV: {os.path.abspath(csv_path)}")

def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        prog="powerlens",
        description="PowerLens: Per-inference energy profiling for NVIDIA Jetson",
    )
    parser.add_argument(
        "--version", action="version", version=f"powerlens {powerlens.__version__}"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run demo profiling session")
    demo_parser.add_argument("--runs", type=int, default=20)
    demo_parser.add_argument("--load", type=float, default=0.8)
    demo_parser.add_argument("--duration", type=float, default=0.05)
    demo_parser.add_argument("--rate", type=float, default=100.0)
    demo_parser.add_argument("--output", "-o", type=str, default=None)
    demo_parser.add_argument("--real", action="store_true")

    # Detect command
    subparsers.add_parser("detect", help="Detect available power sensors")

    # Profile command
    profile_parser = subparsers.add_parser(
        "profile", help="Profile a TensorRT model's energy consumption"
    )
    profile_parser.add_argument("--onnx", type=str, help="Path to ONNX model")
    profile_parser.add_argument("--engine", type=str, help="Path to TensorRT engine")
    profile_parser.add_argument("--runs", type=int, default=50)
    profile_parser.add_argument("--warmup", type=int, default=5)
    profile_parser.add_argument(
        "--iterations", type=int, default=0,
        help="Iterations per run (0=auto-detect)"
    )
    profile_parser.add_argument("--rate", type=float, default=100.0)
    profile_parser.add_argument("--output", "-o", type=str, default=None)

    # Compare command
    compare_parser = subparsers.add_parser(
        "compare", help="Compare energy efficiency of two models"
    )
    compare_parser.add_argument("model_a", type=str, help="First model (ONNX or engine)")
    compare_parser.add_argument("model_b", type=str, help="Second model (ONNX or engine)")
    compare_parser.add_argument("--runs", type=int, default=30)
    compare_parser.add_argument("--warmup", type=int, default=5)
    compare_parser.add_argument("--rate", type=float, default=100.0)
    compare_parser.add_argument("--output", "-o", type=str, default=None)


    # Power modes command
    modes_parser = subparsers.add_parser(
        "power-modes",
        help="Profile across Jetson power modes to find most efficient"
    )
    modes_parser.add_argument("--onnx", type=str, help="Path to ONNX model")
    modes_parser.add_argument("--engine", type=str, help="Path to TensorRT engine")
    modes_parser.add_argument("--runs", type=int, default=20)
    modes_parser.add_argument("--warmup", type=int, default=5)
    modes_parser.add_argument("--rate", type=float, default=100.0)
    modes_parser.add_argument("--output", "-o", type=str, default=None)

    # Batch scaling command
    batch_parser = subparsers.add_parser(
        "batch-scaling",
        help="Profile energy scaling across batch sizes"
    )
    batch_parser.add_argument("--onnx", type=str, help="Path to ONNX model")
    batch_parser.add_argument("--engine", type=str, help="Path to TensorRT engine")
    batch_parser.add_argument(
        "--batches", type=str, default="1,2,4,8",
        help="Comma-separated batch sizes (default: 1,2,4,8)"
    )
    batch_parser.add_argument("--runs", type=int, default=20)
    batch_parser.add_argument("--warmup", type=int, default=5)
    batch_parser.add_argument("--rate", type=float, default=100.0)
    batch_parser.add_argument("--output", "-o", type=str, default=None)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "demo":
        cmd_demo(args)
    elif args.command == "detect":
        cmd_detect(args)
    elif args.command == "profile":
        cmd_profile(args)
    elif args.command == "compare":
        cmd_compare(args)
    elif args.command == "power-modes":
        cmd_power_modes(args)
    elif args.command == "batch-scaling":
        cmd_batch_scaling(args)

if __name__ == "__main__":
    main()
