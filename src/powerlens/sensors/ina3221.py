"""
Driver for Texas Instruments INA3221 triple-channel power monitor.

The INA3221 measures voltage and current on up to 3 channels simultaneously.
On NVIDIA Jetson Orin Nano, INA3221 devices are connected to the I2C bus
and monitor the main power rails (GPU, CPU, system).

Datasheet: https://www.ti.com/lit/ds/symlink/ina3221.pdf

Register map:
    Each channel has:
    - Shunt voltage register (current sense): 2 bytes, signed, 40uV/bit
    - Bus voltage register (rail voltage): 2 bytes, unsigned, 8mV/bit

    Channel 1: shunt=0x01, bus=0x02
    Channel 2: shunt=0x03, bus=0x04
    Channel 3: shunt=0x05, bus=0x06
"""

import time
import logging
from dataclasses import dataclass
from typing import List

from powerlens.sensors.types import PowerSample

logger = logging.getLogger(__name__)

# INA3221 register addresses
_REG_SHUNT_VOLTAGE = {1: 0x01, 2: 0x03, 3: 0x05}
_REG_BUS_VOLTAGE = {1: 0x02, 2: 0x04, 3: 0x06}

# Conversion factors from datasheet
_SHUNT_VOLTAGE_LSB_UV = 40  # 40 microvolts per bit
_BUS_VOLTAGE_LSB_MV = 8  # 8 millivolts per bit


@dataclass
class ChannelConfig:
    """Configuration for one INA3221 channel."""

    channel: int  # 1, 2, or 3
    rail_name: str  # human-readable name
    shunt_resistor_ohm: float  # shunt resistor value in ohms


class INA3221:
    """
    Driver for INA3221 triple-channel power monitor via I2C.

    Reads voltage and current from the INA3221 by directly
    accessing I2C registers via smbus2. This provides higher
    sampling rates and more control than sysfs/hwmon.

    Requires:
        - smbus2 package: pip install smbus2
        - I2C permissions: sudo usermod -aG i2c $USER

    Usage:
        sensor = INA3221(
            bus_number=1,
            address=0x40,
            channels=[
                ChannelConfig(1, "VDD_GPU_SOC", 0.005),
                ChannelConfig(2, "VDD_CPU_CV", 0.005),
                ChannelConfig(3, "VIN_SYS_5V0", 0.005),
            ]
        )
        with sensor:
            samples = sensor.read_all()
            for s in samples:
                print(s)
    """

    def __init__(
        self,
        bus_number: int,
        address: int,
        channels: List[ChannelConfig],
    ):
        self.bus_number = bus_number
        self.address = address
        self.channels = {ch.channel: ch for ch in channels}
        self._bus = None

    def open(self):
        """Open the I2C bus connection."""
        try:
            from smbus2 import SMBus
        except ImportError:
            raise ImportError(
                "smbus2 is required for direct I2C access. "
                "Install with: pip install smbus2\n"
                "This only works on Linux/Jetson, not Windows."
            )
        self._bus = SMBus(self.bus_number)
        logger.info(
            "INA3221 opened on bus %d, address 0x%02x",
            self.bus_number, self.address,
        )

    def close(self):
        """Close the I2C bus connection."""
        if self._bus:
            self._bus.close()
            self._bus = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _read_register(self, register: int) -> int:
        """Read a 16-bit big-endian register from the INA3221."""
        if self._bus is None:
            raise RuntimeError("Sensor not opened. Use 'with' or call open().")
        data = self._bus.read_i2c_block_data(self.address, register, 2)
        return (data[0] << 8) | data[1]

    def _read_shunt_voltage_uv(self, channel: int) -> float:
        """Read shunt voltage in microvolts.

        The shunt voltage register is a signed 16-bit value
        with the 3 LSBs unused. Each bit = 40uV.
        """
        raw = self._read_register(_REG_SHUNT_VOLTAGE[channel])
        # Convert to signed 16-bit
        if raw & 0x8000:
            raw = raw - 0x10000
        # Right-shift by 3 to get actual value
        raw = raw >> 3
        return raw * _SHUNT_VOLTAGE_LSB_UV

    def _read_bus_voltage_mv(self, channel: int) -> float:
        """Read bus voltage in millivolts.

        The bus voltage register is unsigned 16-bit
        with the 3 LSBs unused. Each bit = 8mV.
        """
        raw = self._read_register(_REG_BUS_VOLTAGE[channel])
        raw = raw >> 3
        return raw * _BUS_VOLTAGE_LSB_MV

    def read_channel(self, channel: int) -> PowerSample:
        """Read voltage, current, and power from one channel."""
        if channel not in self.channels:
            raise ValueError(
                f"Channel {channel} not configured. "
                f"Available: {list(self.channels.keys())}"
            )
        config = self.channels[channel]
        timestamp = time.monotonic()

        shunt_uv = self._read_shunt_voltage_uv(channel)
        bus_mv = self._read_bus_voltage_mv(channel)

        voltage_v = bus_mv / 1000.0
        current_a = (shunt_uv / 1_000_000.0) / config.shunt_resistor_ohm
        power_w = voltage_v * current_a

        return PowerSample(
            timestamp=timestamp,
            channel=channel,
            rail_name=config.rail_name,
            voltage_v=voltage_v,
            current_a=current_a,
            power_w=power_w,
        )

    def read_all(self) -> List[PowerSample]:
        """Read all configured channels."""
        return [self.read_channel(ch) for ch in sorted(self.channels.keys())]

    def read_total_power(self) -> float:
        """Total power across all channels in watts."""
        return sum(s.power_w for s in self.read_all())
