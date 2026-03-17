"""
Mock sensor for development and testing.

Generates realistic-looking power data without real hardware.
Use this on Windows or any non-Jetson platform to develop
and test the profiler and analysis modules.

The mock simulates:
- Idle baseline power (~3-4W total)
- Load-dependent power spikes (~8-12W during inference)
- Per-rail distribution (GPU > CPU > System)
- Realistic noise (+/- 3%)
"""

import time
import random
from typing import List

from powerlens.sensors.types import PowerSample


class MockSensor:
    """
    Simulates INA3221 readings with realistic power patterns.

    Usage:
        sensor = MockSensor()
        with sensor:
            samples = sensor.read_all()
            for s in samples:
                print(s)
    """

    RAILS = {
        1: {
            "name": "VDD_GPU_SOC",
            "idle_power_w": 1.8,
            "load_power_w": 7.2,
            "voltage_nominal_v": 5.0,
        },
        2: {
            "name": "VDD_CPU_CV",
            "idle_power_w": 0.9,
            "load_power_w": 2.8,
            "voltage_nominal_v": 5.0,
        },
        3: {
            "name": "VIN_SYS_5V0",
            "idle_power_w": 0.5,
            "load_power_w": 1.5,
            "voltage_nominal_v": 5.0,
        },
    }

    def __init__(self):
        self._open = False
        self._load_level = 0.0  # 0.0 = idle, 1.0 = full load
        self._noise_pct = 0.03  # 3% noise

    def open(self):
        """Open the mock sensor (no-op, for API consistency)."""
        self._open = True

    def close(self):
        """Close the mock sensor (no-op, for API consistency)."""
        self._open = False

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def set_load(self, level: float):
        """Set simulated load level.

        Args:
            level: Load from 0.0 (idle) to 1.0 (full load).
        """
        self._load_level = max(0.0, min(1.0, level))

    def _add_noise(self, value: float) -> float:
        """Add realistic measurement noise."""
        noise = random.gauss(0, value * self._noise_pct)
        return max(0.0, value + noise)

    def read_channel(self, channel: int) -> PowerSample:
        """Read one simulated channel.

        Args:
            channel: Channel number (1, 2, or 3).

        Returns:
            PowerSample with simulated readings.
        """
        rail = self.RAILS[channel]
        timestamp = time.monotonic()

        # Interpolate between idle and load power
        target_power = (
            rail["idle_power_w"]
            + self._load_level * (rail["load_power_w"] - rail["idle_power_w"])
        )

        power_w = self._add_noise(target_power)
        voltage_v = self._add_noise(rail["voltage_nominal_v"])
        current_a = power_w / voltage_v if voltage_v > 0 else 0.0

        return PowerSample(
            timestamp=timestamp,
            channel=channel,
            rail_name=rail["name"],
            voltage_v=voltage_v,
            current_a=current_a,
            power_w=power_w,
        )

    def read_all(self) -> List[PowerSample]:
        """Read all simulated channels.

        Returns:
            List of PowerSample, one per channel.
        """
        return [self.read_channel(ch) for ch in sorted(self.RAILS.keys())]

    def read_total_power(self) -> float:
        """Total simulated power in watts."""
        return sum(s.power_w for s in self.read_all())
