"""Tests for the background power sampler."""

import time

from powerlens.sensors.mock import MockSensor
from powerlens.profiler.sampler import PowerSampler


def test_sampler_collects_samples():
    """Sampler should collect samples while running."""
    sensor = MockSensor()
    sensor.open()
    sampler = PowerSampler(sensor, sample_rate_hz=100)

    sampler.start()
    time.sleep(0.5)  # Let it run for 500ms
    sampler.stop()
    sensor.close()

    samples = sampler.get_samples()
    # At 100Hz for 0.5s, expect roughly 40-60 samples
    # (not exactly 50 due to timing jitter)
    assert len(samples) > 20
    assert len(samples) < 100


def test_sampler_samples_have_three_channels():
    """Each sample cycle should contain readings from all 3 channels."""
    sensor = MockSensor()
    sensor.open()
    sampler = PowerSampler(sensor, sample_rate_hz=50)

    sampler.start()
    time.sleep(0.3)
    sampler.stop()
    sensor.close()

    samples = sampler.get_samples()
    assert len(samples) > 0

    # Each sample is a list of 3 PowerSample (one per channel)
    for sample_cycle in samples:
        assert len(sample_cycle) == 3
        channels = [s.channel for s in sample_cycle]
        assert 1 in channels
        assert 2 in channels
        assert 3 in channels


def test_sampler_timestamps_increase():
    """Timestamps should be monotonically increasing."""
    sensor = MockSensor()
    sensor.open()
    sampler = PowerSampler(sensor, sample_rate_hz=50)

    sampler.start()
    time.sleep(0.3)
    sampler.stop()
    sensor.close()

    samples = sampler.get_samples()
    assert len(samples) > 2

    # Check that first channel timestamp increases across cycles
    timestamps = [cycle[0].timestamp for cycle in samples]
    for i in range(1, len(timestamps)):
        assert timestamps[i] > timestamps[i - 1]


def test_sampler_stop_is_safe_when_not_running():
    """Calling stop() when not running should not crash."""
    sensor = MockSensor()
    sensor.open()
    sampler = PowerSampler(sensor, sample_rate_hz=50)

    # Stop without starting — should not raise
    sampler.stop()
    sensor.close()


def test_sampler_cannot_start_twice():
    """Starting an already-running sampler should raise."""
    sensor = MockSensor()
    sensor.open()
    sampler = PowerSampler(sensor, sample_rate_hz=50)

    sampler.start()
    try:
        raised = False
        try:
            sampler.start()
        except RuntimeError:
            raised = True
        assert raised
    finally:
        sampler.stop()
        sensor.close()


def test_sampler_sample_count_property():
    """sample_count should reflect actual samples collected."""
    sensor = MockSensor()
    sensor.open()
    sampler = PowerSampler(sensor, sample_rate_hz=100)

    assert sampler.sample_count == 0

    sampler.start()
    time.sleep(0.3)
    sampler.stop()
    sensor.close()

    assert sampler.sample_count > 0
    assert sampler.sample_count == len(sampler.get_samples())
