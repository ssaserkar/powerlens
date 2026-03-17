"""
PowerLens Demo — Mock Sensor

Demonstrates PowerLens functionality without Jetson hardware.
Run this on any machine to see what PowerLens does.

Usage:
    python examples/demo_mock.py
"""

import time
import sys
import os

# Add src to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import powerlens
from powerlens.profiler.session import PowerLensContext
from powerlens.sensors.mock import MockSensor
from powerlens.export.csv_export import export_summary_csv, export_raw_csv
from powerlens.visualization.plots import plot_power_trace


def demo_one_call():
    """Demo 1: Simple one-call profiling."""
    print("=" * 50)
    print("Demo 1: One-call profiling")
    print("=" * 50)

    report = powerlens.profile(
        num_runs=20,
        load_level=0.8,
        inference_duration_s=0.05,
        sample_rate_hz=100,
    )

    print(report.summary())
    return report


def demo_context_manager():
    """Demo 2: Context manager with simulated varying workloads."""
    print("=" * 50)
    print("Demo 2: Context manager with varying loads")
    print("=" * 50)

    sensor = MockSensor()
    ctx = PowerLensContext(sensor=sensor, sample_rate_hz=100)

    with ctx:
        # Simulate light model (e.g., MobileNet)
        print("  Running 'light model' (load=0.3)...")
        for i in range(10):
            ctx.mark_inference_start()
            sensor.set_load(0.3)
            time.sleep(0.03)
            sensor.set_load(0.0)
            ctx.mark_inference_end()
            time.sleep(0.01)

        # Simulate heavy model (e.g., YOLOv8)
        print("  Running 'heavy model' (load=0.9)...")
        for i in range(10):
            ctx.mark_inference_start()
            sensor.set_load(0.9)
            time.sleep(0.06)
            sensor.set_load(0.0)
            ctx.mark_inference_end()
            time.sleep(0.01)

    report = ctx.report()
    print(report.summary())
    return report, ctx


def demo_export(report, samples):
    """Demo 3: Export results to CSV and plot."""
    print("=" * 50)
    print("Demo 3: Exporting results")
    print("=" * 50)

    # Create output directory
    os.makedirs("output", exist_ok=True)

    # Export CSV
    summary_path = export_summary_csv(report, "output/energy_summary.csv")
    print(f"  Summary CSV: {summary_path}")

    raw_path = export_raw_csv(samples, "output/raw_samples.csv")
    print(f"  Raw CSV:     {raw_path}")

    # Export plot
    plot_path = plot_power_trace(
        samples=samples,
        report=report,
        filepath="output/power_trace.png",
        title="PowerLens Demo — Mock Sensor",
    )
    print(f"  Plot:        {plot_path}")
    print()


if __name__ == "__main__":
    print()
    print("PowerLens v{} — Demo".format(powerlens.__version__))
    print()

    # Demo 1: Simple profiling
    report1 = demo_one_call()

    # Demo 2: Context manager
    report2, ctx = demo_context_manager()

    # Demo 3: Export (using context manager data which has raw samples)
    samples = ctx._sampler.get_samples()
    demo_export(report2, samples)

    print("Done! Check the 'output/' folder for CSV and plot files.")