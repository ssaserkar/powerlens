"""Tests for Jetson platform configurations."""

from powerlens.sensors.jetson import (
    ORIN_NANO_CONFIG,
    PLATFORM_CONFIGS,
    create_jetson_sensor,
)
from powerlens.sensors.ina3221 import INA3221


def test_orin_nano_config_exists():
    """Orin Nano config should be defined."""
    assert "orin-nano" in PLATFORM_CONFIGS
    assert ORIN_NANO_CONFIG["name"] == "Jetson Orin Nano Developer Kit"


def test_orin_nano_config_has_three_channels():
    """Orin Nano should have 3 power rail channels."""
    channels = ORIN_NANO_CONFIG["channels"]
    assert len(channels) == 3


def test_orin_nano_rail_names():
    """Orin Nano should have expected rail names."""
    names = [ch.rail_name for ch in ORIN_NANO_CONFIG["channels"]]
    assert "VDD_GPU_SOC" in names
    assert "VDD_CPU_CV" in names
    assert "VIN_SYS_5V0" in names


def test_orin_nano_shunt_resistors():
    """Orin Nano shunt resistors should be 5 milliohm."""
    for ch in ORIN_NANO_CONFIG["channels"]:
        assert ch.shunt_resistor_ohm == 0.005


def test_create_jetson_sensor_returns_ina3221():
    """create_jetson_sensor should return an INA3221 instance."""
    sensor = create_jetson_sensor("orin-nano")
    assert isinstance(sensor, INA3221)
    assert sensor.bus_number == 1
    assert sensor.address == 0x40


def test_create_jetson_sensor_unknown_platform_raises():
    """Unknown platform should raise ValueError."""
    raised = False
    try:
        create_jetson_sensor("unknown-board")
    except ValueError as e:
        raised = True
        assert "unknown-board" in str(e).lower() or "Unknown" in str(e)
    assert raised
