"""
PowerLens Quickstart — Profile Your Own Inference Code

This shows how to use PowerLens to measure the energy cost
of your own AI inference code on Jetson.

Usage:
    python examples/quickstart.py
"""

import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import powerlens


def my_fake_inference():
    """Replace this with your actual inference code."""
    import numpy as np
    # Simulate computation
    a = np.random.randn(500, 500).astype(np.float32)
    b = np.random.randn(500, 500).astype(np.float32)
    result = np.dot(a, b)
    return result


def main():
    print(f"PowerLens v{powerlens.__version__} — Quickstart")
    print()

    # Method 1: Profile with context manager (recommended)
    print("Method 1: Context manager")
    print("-" * 40)

    with powerlens.context() as ctx:
        for i in range(20):
            ctx.mark_inference_start()
            my_fake_inference()
            ctx.mark_inference_end()

    report = ctx.report()
    print(report.summary())

    # Method 2: One-call profiling (quick test)
    print("Method 2: One-call profiling")
    print("-" * 40)

    report2 = powerlens.profile(num_runs=10, real_workload=True)
    print(report2.summary())


if __name__ == "__main__":
    main()