"""Tests for the profiling session API."""

import time

import powerlens
from powerlens.profiler.session import PowerLensContext
from powerlens.analysis.energy import EnergyReport
from powerlens.sensors.mock import MockSensor


def test_one_call_profile():
    """powerlens.profile() should return a valid report."""
    report = powerlens.profile(num_runs=5, load_level=0.8)

    assert isinstance(report, EnergyReport)
    assert report.num_inferences > 0
    assert report.mean_energy_j > 0
    assert report.peak_power_w > 0


def test_one_call_profile_summary():
    """profile() report should produce readable summary."""
    report = powerlens.profile(num_runs=3)
    summary = report.summary()

    assert "PowerLens" in summary
    assert "Energy/inference" in summary


def test_context_manager_basic():
    """PowerLensContext should work as a context manager."""
    sensor = MockSensor()
    ctx = PowerLensContext(sensor=sensor, sample_rate_hz=100)

    with ctx:
        for i in range(5):
            ctx.mark_inference_start()
            sensor.set_load(0.8)
            time.sleep(0.05)
            sensor.set_load(0.0)
            ctx.mark_inference_end()

    report = ctx.report()
    assert isinstance(report, EnergyReport)
    assert report.num_inferences > 0
    assert report.mean_energy_j > 0


def test_context_manager_inference_count():
    """inference_count should track marked inferences."""
    ctx = PowerLensContext(sample_rate_hz=100)

    with ctx:
        assert ctx.inference_count == 0

        ctx.mark_inference_start()
        time.sleep(0.02)
        ctx.mark_inference_end()
        assert ctx.inference_count == 1

        ctx.mark_inference_start()
        time.sleep(0.02)
        ctx.mark_inference_end()
        assert ctx.inference_count == 2


def test_context_manager_end_without_start_raises():
    """Calling mark_inference_end without start should raise."""
    ctx = PowerLensContext(sample_rate_hz=100)

    with ctx:
        raised = False
        try:
            ctx.mark_inference_end()
        except RuntimeError:
            raised = True
        assert raised


def test_profile_more_runs_more_energy():
    """More inferences should result in more total energy."""
    report_few = powerlens.profile(num_runs=3, load_level=0.8)
    report_many = powerlens.profile(num_runs=10, load_level=0.8)

    assert report_many.total_energy_j > report_few.total_energy_j
    assert report_many.num_inferences > report_few.num_inferences
