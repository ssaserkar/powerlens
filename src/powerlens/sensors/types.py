"""
Core data types for power measurements.

These are used across all sensor backends (INA3221, sysfs, mock)
and by the analysis modules.
"""

from dataclasses import dataclass


@dataclass
class PowerSample:
    """A single power measurement from one sensor channel.

    Attributes:
        timestamp: Monotonic clock time in seconds (time.monotonic()).
                   Using monotonic avoids issues with system clock changes.
        channel: Sensor channel number (1, 2, or 3 for INA3221).
        rail_name: Human-readable name of the power rail
                   (e.g., "VDD_GPU_SOC").
        voltage_v: Bus voltage in volts.
        current_a: Current in amperes.
        power_w: Power in watts (voltage_v * current_a).
    """

    timestamp: float
    channel: int
    rail_name: str
    voltage_v: float
    current_a: float
    power_w: float

    def __repr__(self) -> str:
        return (
            f"PowerSample({self.rail_name}: "
            f"{self.voltage_v:.3f}V, "
            f"{self.current_a:.3f}A, "
            f"{self.power_w:.3f}W "
            f"@ {self.timestamp:.4f})"
        )