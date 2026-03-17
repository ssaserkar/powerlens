"""Tests for the energy computation engine."""

import time

from powerlens.sensors.mock import MockSensor
from powerlens.profiler.sampler import PowerSampler
from powerlens.analysis.energy import (
    compute_energy_report,
    EnergyReport,
    InferenceEnergy,
)


def test_energy_report_with_mock_data():
    """Full integration test: sample mock sensor, compute energy."""
    sensor = MockSensor()
    sensor.open()
    sampler = PowerSampler(sensor, sample_rate_hz=100)

    # Collect idle baseline
    sampler.start()
    time.sleep(0.3)
    sampler.stop()
    idle_samples = sampler.get_samples()

    # Collect inference samples
    sampler2 = PowerSampler(sensor, sample_rate_hz=100)
    sampler2.start()

    # Simulate 5 inferences
    inference_timestamps = []
    for i in range(5):
        start = time.monotonic()
        sensor.set_load(0.8)  # Simulate GPU load
        time.sleep(0.05)  # 50ms inference
        end = time.monotonic()
        sensor.set_load(0.0)  # Back to idle
        inference_timestamps.append((start, end))
        time.sleep(0.02)  # Gap between inferences

    sampler2.stop()
    sensor.close()

    samples = sampler2.get_samples()
    report = compute_energy_report(samples, inference_timestamps, idle_samples)

    assert isinstance(report, EnergyReport)
    assert report.num_inferences > 0
    assert report.mean_energy_j > 0
    assert report.peak_power_w > 0
    assert report.idle_power_w > 0


def test_energy_report_summary_string():
    """summary() should return a readable string."""
    sensor = MockSensor()
    sensor.open()
    sampler = PowerSampler(sensor, sample_rate_hz=100)
    sampler.start()

    inference_timestamps = []
    for i in range(3):
        start = time.monotonic()
        sensor.set_load(0.5)
        time.sleep(0.05)
        end = time.monotonic()
        sensor.set_load(0.0)
        inference_timestamps.append((start, end))
        time.sleep(0.02)

    sampler.stop()
    sensor.close()

    report = compute_energy_report(sampler.get_samples(), inference_timestamps)
    summary = report.summary()

    assert "PowerLens" in summary
    assert "Energy/inference" in summary
    assert "J" in summary


def test_energy_per_inference_is_positive():
    """Each inference should have positive energy."""
    sensor = MockSensor()
    sensor.open()
    sampler = PowerSampler(sensor, sample_rate_hz=100)
    sampler.start()

    inference_timestamps = []
    for i in range(5):
        start = time.monotonic()
        sensor.set_load(1.0)
        time.sleep(0.05)
        end = time.monotonic()
        sensor.set_load(0.0)
        inference_timestamps.append((start, end))
        time.sleep(0.02)

    sampler.stop()
    sensor.close()

    report = compute_energy_report(sampler.get_samples(), inference_timestamps)

    for inf in report.inferences:
        assert isinstance(inf, InferenceEnergy)
        assert inf.energy_j > 0
        assert inf.duration_s > 0
        assert inf.avg_power_w > 0


def test_higher_load_means_more_energy():
    """Inferences at higher load should use more energy."""
    sensor = MockSensor()
    sensor.open()

    # Low load inferences
    sampler_low = PowerSampler(sensor, sample_rate_hz=100)
    sampler_low.start()
    low_timestamps = []
    for i in range(5):
        start = time.monotonic()
        sensor.set_load(0.2)
        time.sleep(0.05)
        end = time.monotonic()
        sensor.set_load(0.0)
        low_timestamps.append((start, end))
        time.sleep(0.02)
    sampler_low.stop()
    low_report = compute_energy_report(
        sampler_low.get_samples(), low_timestamps
    )

    # High load inferences
    sampler_high = PowerSampler(sensor, sample_rate_hz=100)
    sampler_high.start()
    high_timestamps = []
    for i in range(5):
        start = time.monotonic()
        sensor.set_load(1.0)
        time.sleep(0.05)
        end = time.monotonic()
        sensor.set_load(0.0)
        high_timestamps.append((start, end))
        time.sleep(0.02)
    sampler_high.stop()
    sensor.close()
    high_report = compute_energy_report(
        sampler_high.get_samples(), high_timestamps
    )

    assert high_report.mean_energy_j > low_report.mean_energy_j


def test_rail_breakdown_present():
    """Report should include per-rail power breakdown."""
    sensor = MockSensor()
    sensor.open()
    sampler = PowerSampler(sensor, sample_rate_hz=100)
    sampler.start()

    inference_timestamps = []
    for i in range(3):
        start = time.monotonic()
        sensor.set_load(0.5)
        time.sleep(0.05)
        end = time.monotonic()
        sensor.set_load(0.0)
        inference_timestamps.append((start, end))
        time.sleep(0.02)

    sampler.stop()
    sensor.close()

    report = compute_energy_report(sampler.get_samples(), inference_timestamps)

    assert len(report.rail_breakdown) > 0
    assert "VDD_GPU_SOC" in report.rail_breakdown
    assert report.rail_breakdown["VDD_GPU_SOC"] > 0


def test_empty_samples_returns_empty_report():
    """Empty input should return a valid but empty report."""
    report = compute_energy_report([], [])
    assert report.num_inferences == 0
    assert report.mean_energy_j == 0.0
    assert isinstance(report.inferences, list)
    assert len(report.inferences) == 0
