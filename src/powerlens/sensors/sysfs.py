"""
Fallback sensor reader using Linux sysfs/hwmon interface.

If the user doesn't have I2C permissions or the INA3221 is
already claimed by the kernel driver, we can read through
the standard hwmon sysfs interface.

This is less flexible (no control over sample rate) but
works without special permissions on Jetson.

Typical paths on Jetson Orin Nano:
    /sys/bus/i2c/devices/1-0040/hwmon/hwmon*/
        in1_input   (bus voltage channel 1, mV)
        curr1_input (current channel 1, mA)
        power1_input (power channel 1, uW) — may not exist
"""

import glob
import os
import time
import logging
from typing import List, Optional, Dict

from powerlens.sensors.types import PowerSample

logger = logging.getLogger(__name__)


# Default rail names for Jetson Orin Nano channels
_DEFAULT_RAIL_NAMES = {
    1: "VDD_GPU_SOC",
    2: "VDD_CPU_CV",
    3: "VIN_SYS_5V0",
}


def find_ina3221_hwmon_paths() -> List[str]:
    """Find all hwmon sysfs paths for INA3221 devices.

    Returns list of paths like /sys/class/hwmon/hwmon3/
    """
    paths = []
    search_patterns = [
        "/sys/bus/i2c/devices/*/hwmon/hwmon*",
        "/sys/class/hwmon/hwmon*",
    ]
    for pattern in search_patterns:
        for path in glob.glob(pattern):
            name_file = os.path.join(path, "name")
            if os.path.exists(name_file):
                try:
                    with open(name_file, "r") as f:
                        name = f.read().strip()
                    if "ina3221" in name.lower():
                        if path not in paths:
                            paths.append(path)
                except OSError:
                    continue
    return paths


class SysfsSensor:
    """
    Read INA3221 power data through Linux sysfs/hwmon interface.

    This is the fallback when direct I2C access is not available.
    Works without i2c group membership on most Jetson configurations.

    Usage:
        sensor = SysfsSensor()
        if sensor.available():
            with sensor:
                samples = sensor.read_all()
                for s in samples:
                    print(s)
    """

    def __init__(
        self,
        hwmon_path: Optional[str] = None,
        rail_names: Optional[Dict[int, str]] = None,
    ):
        """Initialize sysfs sensor.

        Args:
            hwmon_path: Path to hwmon directory. Auto-detected if None.
            rail_names: Dict mapping channel number to rail name.
                        Uses Jetson Orin Nano defaults if None.
        """
        self._path = hwmon_path
        self._rail_names = rail_names or _DEFAULT_RAIL_NAMES
        self._detected = False

    def _detect(self):
        """Auto-detect hwmon path if not provided."""
        if self._path is None:
            paths = find_ina3221_hwmon_paths()
            if paths:
                self._path = paths[0]
                logger.info("Found INA3221 at sysfs path: %s", self._path)
            else:
                logger.debug("No INA3221 found in sysfs")
        self._detected = True

    def available(self) -> bool:
        """Check if sysfs sensor is available and readable."""
        if not self._detected:
            self._detect()
        if self._path is None:
            return False
        return os.path.exists(os.path.join(self._path, "in1_input"))

    def open(self):
        """Open sensor (detect path if needed)."""
        if not self._detected:
            self._detect()
        if self._path is None:
            raise RuntimeError(
                "No INA3221 sensor found in sysfs. "
                "Are you running on a Jetson device?"
            )

    def close(self):
        """Close sensor (no-op for sysfs)."""
        pass

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _read_sysfs_value(self, filename: str) -> float:
        """Read a single value from a sysfs file."""
        filepath = os.path.join(self._path, filename)
        with open(filepath, "r") as f:
            return float(f.read().strip())

    def read_channel(self, channel: int) -> PowerSample:
        """Read one channel from sysfs.

        sysfs units:
            in{n}_input: millivolts
            curr{n}_input: milliamps
        """
        timestamp = time.monotonic()

        voltage_mv = self._read_sysfs_value(f"in{channel}_input")
        current_ma = self._read_sysfs_value(f"curr{channel}_input")

        voltage_v = voltage_mv / 1000.0
        current_a = current_ma / 1000.0
        power_w = voltage_v * current_a

        rail_name = self._rail_names.get(channel, f"channel_{channel}")

        return PowerSample(
            timestamp=timestamp,
            channel=channel,
            rail_name=rail_name,
            voltage_v=voltage_v,
            current_a=current_a,
            power_w=power_w,
        )

    def read_all(self) -> List[PowerSample]:
        """Read all 3 channels."""
        samples = []
        for ch in sorted(self._rail_names.keys()):
            try:
                samples.append(self.read_channel(ch))
            except (FileNotFoundError, OSError) as e:
                logger.debug("Could not read channel %d: %s", ch, e)
                continue
        return samples

    def read_total_power(self) -> float:
        """Total power across all readable channels."""
        return sum(s.power_w for s in self.read_all())
