"""
Profiling session: the main user-facing API.

Two ways to use:

1. Context manager (for custom inference code):

    ctx = PowerLensContext(sample_rate_hz=100)
    with ctx:
        for image in images:
            ctx.mark_inference_start()
            result = my_model.infer(image)
            ctx.mark_inference_end()
    report = ctx.report()
    print(report.summary())

2. One-call with dummy workload (for testing/demo):

    report = powerlens.profile(num_runs=50, load_level=0.8)
    print(report.summary())
"""

import time
import logging
from typing import List, Optional

from powerlens.sensors.mock import MockSensor
from powerlens.profiler.sampler import PowerSampler
from powerlens.analysis.energy import compute_energy_report, EnergyReport

logger = logging.getLogger(__name__)


class PowerLensContext:
    """Context manager for profiling custom inference code.

    Usage:
        ctx = PowerLensContext(sample_rate_hz=100)
        with ctx:
            for image in images:
                ctx.mark_inference_start()
                result = model.infer(image)
                ctx.mark_inference_end()
        report = ctx.report()
        print(report.summary())
    """

    def __init__(self, sensor=None, sample_rate_hz: float = 100.0):
        """Initialize profiling context.

        Args:
            sensor: Sensor object with read_all() method.
                    If None, uses MockSensor (for development/testing).
            sample_rate_hz: Power sampling rate in Hz.
        """
        self._sensor = sensor
        self._sample_rate_hz = sample_rate_hz
        self._sampler: Optional[PowerSampler] = None
        self._idle_samples: Optional[List] = None
        self._inference_timestamps: List[tuple] = []
        self._current_start: Optional[float] = None
        self._owns_sensor = False

    def __enter__(self):
        # Create mock sensor if none provided
        if self._sensor is None:
            self._sensor = MockSensor()
            self._owns_sensor = True

        self._sensor.open()

        # Collect idle baseline (1 second)
        logger.info("Collecting idle baseline...")
        idle_sampler = PowerSampler(self._sensor, self._sample_rate_hz)
        idle_sampler.start()
        time.sleep(1.0)
        idle_sampler.stop()
        self._idle_samples = idle_sampler.get_samples()
        logger.info("Idle baseline: %d samples collected", len(self._idle_samples))

        # Start inference sampling
        self._sampler = PowerSampler(self._sensor, self._sample_rate_hz)
        self._sampler.start()
        self._inference_timestamps = []

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._sampler:
            self._sampler.stop()
        if self._owns_sensor and self._sensor:
            self._sensor.close()

    def mark_inference_start(self):
        """Call immediately before each inference."""
        self._current_start = time.monotonic()

    def mark_inference_end(self):
        """Call immediately after each inference."""
        if self._current_start is None:
            raise RuntimeError(
                "mark_inference_end() called without mark_inference_start()"
            )
        end = time.monotonic()
        self._inference_timestamps.append((self._current_start, end))
        self._current_start = None

    @property
    def inference_count(self) -> int:
        """Number of inferences recorded so far."""
        return len(self._inference_timestamps)

    def report(self) -> EnergyReport:
        """Compute and return the energy report.

        Call this after exiting the context manager.
        """
        if self._sampler is None:
            raise RuntimeError("No profiling data. Use within a 'with' block.")

        samples = self._sampler.get_samples()
        return compute_energy_report(
            samples=samples,
            inference_timestamps=self._inference_timestamps,
            idle_samples=self._idle_samples,
        )


def profile(
    num_runs: int = 50,
    inference_duration_s: float = 0.05,
    load_level: float = 0.8,
    sample_rate_hz: float = 100.0,
    sensor=None,
    real_workload: bool = False,
) -> EnergyReport:
    """One-call profiling function.

    Args:
        num_runs: Number of inferences to simulate.
        inference_duration_s: Duration of each inference in seconds.
        load_level: Simulated GPU load 0.0-1.0 (mock sensor only).
        sample_rate_hz: Power sampling rate in Hz.
        sensor: Sensor object. If None, auto-detects or uses mock.
        real_workload: If True, run actual CPU computation instead
                       of time.sleep(). Creates measurable power change
                       on real hardware.

    Returns:
        EnergyReport with per-inference energy statistics.
    """
    from powerlens.sensors.mock import MockSensor

    if sensor is None:
        from powerlens.sensors.auto import detect_sensor
        sensor = detect_sensor(use_mock_fallback=True)

    use_mock = isinstance(sensor, MockSensor)
    sensor.open()

    logger.info("PowerLens profiling: %d runs", num_runs)

    # Collect idle baseline
    logger.info("Measuring idle baseline...")
    idle_sampler = PowerSampler(sensor, sample_rate_hz)
    idle_sampler.start()
    time.sleep(1.0)
    idle_sampler.stop()
    idle_samples = idle_sampler.get_samples()

    # Run inferences
    sampler = PowerSampler(sensor, sample_rate_hz)
    sampler.start()

    inference_timestamps = []
    for i in range(num_runs):
        start = time.monotonic()

        if use_mock:
            sensor.set_load(load_level)
            time.sleep(inference_duration_s)
            sensor.set_load(0.0)
        elif real_workload:
            _cpu_stress(inference_duration_s)
        else:
            time.sleep(inference_duration_s)

        end = time.monotonic()
        inference_timestamps.append((start, end))
        time.sleep(0.005)

    sampler.stop()
    sensor.close()

    logger.info("Computing energy report...")
    return compute_energy_report(
        samples=sampler.get_samples(),
        inference_timestamps=inference_timestamps,
        idle_samples=idle_samples,
    )


def _cpu_stress(duration_s: float):
    """Run CPU-intensive computation for a specified duration.

    This creates a measurable power increase on real hardware,
    unlike time.sleep() which keeps the CPU idle.
    """
    import numpy as np

    end_time = time.monotonic() + duration_s
    while time.monotonic() < end_time:
        # Matrix multiply — stresses CPU and memory
        a = np.random.randn(200, 200).astype(np.float32)
        b = np.random.randn(200, 200).astype(np.float32)
        np.dot(a, b)
