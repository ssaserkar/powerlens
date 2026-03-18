"""
PowerLens command-line interface.

Usage:
    powerlens demo              Run demo with mock sensor
    powerlens detect            Detect available sensors
    powerlens --version         Show version
"""

import argparse
import os
import sys
import time

import powerlens
from powerlens.profiler.session import PowerLensContext
from powerlens.sensors.mock import MockSensor
from powerlens.export.csv_export import export_summary_csv, export_raw_csv
from powerlens.visualization.plots import plot_power_trace


def cmd_demo(args):
    """Run a demo profiling session."""
    print(f"PowerLens v{powerlens.__version__}")
    print(f"Running demo: {args.runs} inferences at {args.load:.0%} load")
    print()

    if args.real:
        # Use auto-detected real sensor
        from powerlens.sensors.auto import detect_sensor
        sensor = detect_sensor(use_mock_fallback=False)
        print(f"Using real sensor: {type(sensor).__name__}")
    else:
        sensor = MockSensor()

    ctx = PowerLensContext(sensor=sensor, sample_rate_hz=args.rate)

    with ctx:
        for i in range(args.runs):
            ctx.mark_inference_start()
            if isinstance(sensor, MockSensor):
                sensor.set_load(args.load)
            time.sleep(args.duration)
            if isinstance(sensor, MockSensor):
                sensor.set_load(0.0)
            ctx.mark_inference_end()
            time.sleep(0.005)

    report = ctx.report()
    samples = ctx._sampler.get_samples()

    print(report.summary())

    if args.output:
        os.makedirs(args.output, exist_ok=True)

        csv_path = export_summary_csv(
            report, os.path.join(args.output, "energy_summary.csv")
        )
        print(f"Summary CSV: {csv_path}")

        raw_path = export_raw_csv(
            samples, os.path.join(args.output, "raw_samples.csv")
        )
        print(f"Raw CSV:     {raw_path}")

        plot_path = plot_power_trace(
            samples=samples,
            report=report,
            filepath=os.path.join(args.output, "power_trace.png"),
            title=f"PowerLens — {args.runs} inferences",
        )
        print(f"Plot:        {plot_path}")
        print()


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
    elif info["recommended"] == "i2c":
        print("Direct I2C sensor available (best performance). Run with:")
        print("  powerlens demo --real")


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
    demo_parser.add_argument(
        "--runs", type=int, default=20, help="Number of inferences (default: 20)"
    )
    demo_parser.add_argument(
        "--load", type=float, default=0.8, help="Simulated load 0.0-1.0 (default: 0.8)"
    )
    demo_parser.add_argument(
        "--duration", type=float, default=0.05,
        help="Inference duration in seconds (default: 0.05)"
    )
    demo_parser.add_argument(
        "--rate", type=float, default=100.0,
        help="Sample rate in Hz (default: 100)"
    )
    demo_parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output directory for CSV and plot files"
    )
    demo_parser.add_argument(
        "--real", action="store_true",
        help="Use real sensor instead of mock (requires Jetson)"
    )

    # Detect command
    subparsers.add_parser("detect", help="Detect available power sensors")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "demo":
        cmd_demo(args)
    elif args.command == "detect":
        cmd_detect(args)


if __name__ == "__main__":
    main()
