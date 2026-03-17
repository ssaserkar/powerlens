"""Tests for the mock sensor."""

from powerlens.sensors.mock import MockSensor
from powerlens.sensors.types import PowerSample


def test_mock_sensor_returns_three_channels():
    """Mock sensor should return exactly 3 channel readings."""
    sensor = MockSensor()
    with sensor:
        samples = sensor.read_all()

    assert len(samples) == 3
    assert samples[0].channel == 1
    assert samples[1].channel == 2
    assert samples[2].channel == 3


def test_mock_sensor_returns_power_samples():
    """Each reading should be a PowerSample with valid data."""
    sensor = MockSensor()
    with sensor:
        samples = sensor.read_all()

    for sample in samples:
        assert isinstance(sample, PowerSample)
        assert sample.voltage_v > 0
        assert sample.current_a >= 0
        assert sample.power_w >= 0
        assert sample.rail_name != ""
        assert sample.timestamp > 0


def test_mock_sensor_idle_power_range():
    """At idle (default), total power should be roughly 3-4W."""
    sensor = MockSensor()
    with sensor:
        total = sensor.read_total_power()

    # Idle should be around 3.2W (1.8 + 0.9 + 0.5) +/- noise
    assert 2.0 < total < 5.0


def test_mock_sensor_load_power_range():
    """At full load, total power should be roughly 10-12W."""
    sensor = MockSensor()
    with sensor:
        sensor.set_load(1.0)
        total = sensor.read_total_power()

    # Load should be around 11.5W (7.2 + 2.8 + 1.5) +/- noise
    assert 8.0 < total < 15.0


def test_mock_sensor_load_increases_power():
    """Higher load should produce higher power readings."""
    sensor = MockSensor()
    with sensor:
        sensor.set_load(0.0)
        idle_power = sensor.read_total_power()

        sensor.set_load(1.0)
        load_power = sensor.read_total_power()

    assert load_power > idle_power


def test_mock_sensor_rail_names():
    """Rail names should match Jetson Orin Nano conventions."""
    sensor = MockSensor()
    with sensor:
        samples = sensor.read_all()

    names = [s.rail_name for s in samples]
    assert "VDD_GPU_SOC" in names
    assert "VDD_CPU_CV" in names
    assert "VIN_SYS_5V0" in names