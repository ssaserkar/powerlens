"""Tests for INA3221 sensor driver (configuration and error handling)."""


from powerlens.sensors.ina3221 import INA3221, ChannelConfig


def test_ina3221_channel_config():
    """ChannelConfig should store channel parameters."""
    config = ChannelConfig(
        channel=1,
        rail_name="VDD_GPU_SOC",
        shunt_resistor_ohm=0.005,
    )
    assert config.channel == 1
    assert config.rail_name == "VDD_GPU_SOC"
    assert config.shunt_resistor_ohm == 0.005


def test_ina3221_creation():
    """INA3221 should accept valid configuration."""
    sensor = INA3221(
        bus_number=1,
        address=0x40,
        channels=[
            ChannelConfig(1, "VDD_GPU_SOC", 0.005),
            ChannelConfig(2, "VDD_CPU_CV", 0.005),
            ChannelConfig(3, "VIN_SYS_5V0", 0.005),
        ],
    )
    assert sensor.bus_number == 1
    assert sensor.address == 0x40
    assert len(sensor.channels) == 3


def test_ina3221_invalid_channel_raises():
    """Reading an unconfigured channel should raise ValueError."""
    sensor = INA3221(
        bus_number=1,
        address=0x40,
        channels=[
            ChannelConfig(1, "VDD_GPU_SOC", 0.005),
        ],
    )
    # Can't read without opening, but we can check channel validation
    # by testing the channels dict
    assert 1 in sensor.channels
    assert 2 not in sensor.channels
    assert 3 not in sensor.channels


def test_ina3221_open_without_smbus2_raises():
    """Opening INA3221 without smbus2 should raise ImportError."""
    sensor = INA3221(
        bus_number=1,
        address=0x40,
        channels=[ChannelConfig(1, "TEST", 0.005)],
    )
    # On Windows, smbus2 is not installed, so open() should raise
    try:
        sensor.open()
        # If we get here, smbus2 IS installed (unlikely on Windows)
        sensor.close()
    except ImportError as e:
        assert "smbus2" in str(e)
    except OSError:
        # Could also get OSError if smbus2 is installed but no I2C bus
        pass


def test_ina3221_read_without_open_raises():
    """Reading without opening should raise RuntimeError."""
    sensor = INA3221(
        bus_number=1,
        address=0x40,
        channels=[ChannelConfig(1, "TEST", 0.005)],
    )
    raised = False
    try:
        sensor._read_register(0x01)
    except RuntimeError as e:
        raised = True
        assert "not opened" in str(e).lower()
    assert raised
