"""
Fallback sensor reader using Linux sysfs/hwmon interface.

If the user doesn't have I2C permissions or the INA3221 is
already claimed by the kernel driver, we can read through
the standard hwmon sysfs interface.

On Jetson, the kernel INA3221 driver typically claims the device,
making this the PRIMARY sensor backend (not a fallback).

Sysfs provides label files that tell us the actual rail names,
so we don't need to hardcode them.
"""

import glob
import os
import time
import logging
from typing import List, Optional, Dict

from powerlens.sensors.types import PowerSample

logger = logging.getLogger(__name__)


def find_ina3221_hwmon_paths() -> List[str]:
    """Find all hwmon sysfs paths for INA3221 devices."""
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

    This is the primary sensor backend on Jetson because the kernel
    INA3221 driver claims the I2C device, preventing direct access.

    Auto-detects rail names from sysfs label files.

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
        self._path = hwmon_path
        self._rail_names = rail_names  # None means auto-detect from labels
        self._detected = False
        self._channels: List[int] = []

    def _detect(self):
        """Auto-detect hwmon path and channel configuration."""
        if self._path is None:
            paths = find_ina3221_hwmon_paths()
            if paths:
                self._path = paths[0]
                logger.info("Found INA3221 at sysfs path: %s", self._path)
            else:
                logger.debug("No INA3221 found in sysfs")
                self._detected = True
                return

        # Auto-detect available channels
        # Only include channels that have BOTH:
        #   1. in{n}_input and curr{n}_input files (voltage + current)
        #   2. in{n}_label file (proves it's a real named rail)
        # This filters out computed/derived channels (like channel 4)
        if self._rail_names is None:
            self._rail_names = {}
            for ch in range(1, 4):  # INA3221 has exactly 3 physical channels
                voltage_file = os.path.join(self._path, f"in{ch}_input")
                current_file = os.path.join(self._path, f"curr{ch}_input")
                label_file = os.path.join(self._path, f"in{ch}_label")

                if not (os.path.exists(voltage_file) and os.path.exists(current_file)):
                    continue

                # Read label if available
                if os.path.exists(label_file):
                    try:
                        with open(label_file, "r") as f:
                            label = f.read().strip()
                        self._rail_names[ch] = label
                    except OSError:
                        self._rail_names[ch] = f"channel_{ch}"
                else:
                    self._rail_names[ch] = f"channel_{ch}"

        self._channels = sorted(self._rail_names.keys())
        logger.info(
            "Detected %d channels: %s",
            len(self._channels),
            {ch: self._rail_names[ch] for ch in self._channels},
        )
        self._detected = True

    def available(self) -> bool:
        """Check if sysfs sensor is available and readable."""
        if not self._detected:
            self._detect()
        if self._path is None:
            return False
        return len(self._channels) > 0

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
        """Read all detected channels."""
        samples = []
        for ch in self._channels:
            try:
                samples.append(self.read_channel(ch))
            except (FileNotFoundError, OSError) as e:
                logger.debug("Could not read channel %d: %s", ch, e)
                continue
        return samples

    def read_total_power(self) -> float:
        """Total power across all readable channels."""
        return sum(s.power_w for s in self.read_all())

    @property
    def detected_rails(self) -> Dict[int, str]:
        """Return detected channel-to-rail-name mapping."""
        if not self._detected:
            self._detect()
        return dict(self._rail_names) if self._rail_names else {}
