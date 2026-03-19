"""
Auto-detection of the best available sensor backend.

Tries in order:
1. Direct I2C via INA3221 driver (best resolution, needs permissions)
2. Sysfs/hwmon fallback (works without special permissions)
3. Mock sensor (for development/testing)

Usage:
    sensor = detect_sensor()
    with sensor:
        samples = sensor.read_all()
"""

import os
import logging
import platform as platform_mod

from powerlens.sensors.mock import MockSensor

logger = logging.getLogger(__name__)


def detect_jetson_board() -> dict:
    """Detect which Jetson board we're running on.

    Reads /proc/device-tree/model which NVIDIA populates
    on all Jetson platforms.

    Returns:
        Dict with 'model', 'chip', and 'is_jetson' info.
    """
    info = {
        "model": "unknown",
        "chip": "unknown",
        "is_jetson": False,
    }

    model_path = "/proc/device-tree/model"
    if os.path.exists(model_path):
        try:
            with open(model_path, "r", errors="replace") as f:
                model = f.read().strip().rstrip("\x00")
            info["model"] = model
            info["is_jetson"] = "jetson" in model.lower() or "orin" in model.lower()
        except OSError:
            pass

    chip_path = "/sys/module/tegra_fuse/parameters/tegra_chip_id"
    if os.path.exists(chip_path):
        try:
            with open(chip_path, "r") as f:
                chip_id = f.read().strip()
            chip_map = {
                "33": "TX1",
                "24": "TX2",
                "25": "Xavier",
                "35": "Orin",
            }
            info["chip"] = chip_map.get(chip_id, f"unknown ({chip_id})")
            info["is_jetson"] = True
        except OSError:
            pass

    jetpack_path = "/etc/nv_tegra_release"
    if os.path.exists(jetpack_path):
        try:
            with open(jetpack_path, "r") as f:
                info["jetpack_release"] = f.readline().strip()
        except OSError:
            pass

    return info


def detect_sensor(use_mock_fallback: bool = True):
    """Detect and return the best available power sensor.

    Args:
        use_mock_fallback: If True, return MockSensor when no real
                          sensor is found. If False, raise RuntimeError.

    Returns:
        Sensor object with open/close/read_all interface.

    Raises:
        RuntimeError: If no sensor found and use_mock_fallback is False.
    """
    system = platform_mod.system()

    if system == "Linux":
        sensor = _try_i2c_sensor()
        if sensor is not None:
            return sensor

        sensor = _try_sysfs_sensor()
        if sensor is not None:
            return sensor

        logger.warning(
            "No real power sensor found on this Linux system. "
            "For direct I2C: sudo usermod -aG i2c $USER (then re-login). "
            "For sysfs: check /sys/class/hwmon/ for ina3221 entries."
        )
    else:
        logger.info(
            "Not running on Linux (detected: %s). "
            "Real sensors not available.",
            system,
        )

    if use_mock_fallback:
        logger.info("Using mock sensor for development/testing")
        return MockSensor()

    raise RuntimeError(
        "No power sensor found. PowerLens requires:\n"
        "  - NVIDIA Jetson with INA3221 sensors, OR\n"
        "  - Use MockSensor for development\n"
        "For direct I2C: sudo usermod -aG i2c $USER\n"
        "For sysfs: check /sys/class/hwmon/ for ina3221"
    )


def _try_i2c_sensor():
    """Try to open INA3221 via direct I2C."""
    try:
        from powerlens.sensors.jetson import create_jetson_sensor
        sensor = create_jetson_sensor("orin-nano")
        sensor.open()
        samples = sensor.read_all()
        if len(samples) > 0 and all(s.voltage_v > 0 for s in samples):
            logger.info(
                "INA3221 detected via I2C (bus=%d, addr=0x%02x)",
                sensor.bus_number, sensor.address,
            )
            return sensor
        else:
            sensor.close()
            logger.debug("I2C sensor opened but readings invalid")
            return None
    except ImportError:
        logger.debug("smbus2 not installed, skipping I2C")
        return None
    except Exception as e:
        logger.debug("I2C sensor not available: %s", e)
        return None


def _try_sysfs_sensor():
    """Try to open INA3221 via sysfs/hwmon."""
    try:
        from powerlens.sensors.sysfs import SysfsSensor
        sensor = SysfsSensor()
        if sensor.available():
            sensor.open()
            samples = sensor.read_all()
            if len(samples) > 0:
                logger.info("INA3221 detected via sysfs at %s", sensor._path)
                return sensor
            else:
                logger.debug("Sysfs sensor found but no readable channels")
                return None
        else:
            logger.debug("No INA3221 found in sysfs")
            return None
    except Exception as e:
        logger.debug("Sysfs sensor not available: %s", e)
        return None


def get_sensor_info() -> dict:
    """Get information about available sensors.

    Returns a dict with detection results for display/debugging.
    """
    system = platform_mod.system()

    board = detect_jetson_board() if system == "Linux" else {"model": "N/A", "is_jetson": False}

    info = {
        "platform": system,
        "board_model": board.get("model", "unknown"),
        "board_chip": board.get("chip", "unknown"),
        "is_jetson": board.get("is_jetson", False),
        "i2c_available": False,
        "i2c_detail": "",
        "sysfs_available": False,
        "sysfs_detail": "",
        "mock_available": True,
        "recommended": "mock",
    }

    if system != "Linux":
        info["i2c_detail"] = f"Not Linux (detected: {system})"
        info["sysfs_detail"] = f"Not Linux (detected: {system})"
        return info

    # Check smbus2
    try:
        import smbus2  # noqa: F401
        info["i2c_detail"] = "smbus2 installed"
    except ImportError:
        info["i2c_detail"] = "smbus2 not installed (pip install smbus2)"
        try:
            from powerlens.sensors.sysfs import find_ina3221_hwmon_paths
            paths = find_ina3221_hwmon_paths()
            if paths:
                info["sysfs_available"] = True
                info["sysfs_detail"] = f"Found at: {paths[0]}"
                info["recommended"] = "sysfs"
            else:
                info["sysfs_detail"] = "No INA3221 found in /sys/class/hwmon/"
        except Exception as e:
            info["sysfs_detail"] = f"Error: {e}"
        return info

    # Check I2C
    try:
        from powerlens.sensors.jetson import create_jetson_sensor
        sensor = create_jetson_sensor("orin-nano")
        sensor.open()
        samples = sensor.read_all()
        sensor.close()
        if len(samples) > 0 and all(s.voltage_v > 0 for s in samples):
            info["i2c_available"] = True
            info["i2c_detail"] = (
                f"INA3221 on bus {sensor.bus_number}, "
                f"address 0x{sensor.address:02x}"
            )
            info["recommended"] = "i2c"
        else:
            info["i2c_detail"] = "Opened but readings invalid"
    except Exception as e:
        info["i2c_detail"] = f"Error: {e}"

    # Check sysfs
    try:
        from powerlens.sensors.sysfs import find_ina3221_hwmon_paths
        paths = find_ina3221_hwmon_paths()
        if paths:
            info["sysfs_available"] = True
            info["sysfs_detail"] = f"Found at: {paths[0]}"
            if not info["i2c_available"]:
                info["recommended"] = "sysfs"
        else:
            info["sysfs_detail"] = "No INA3221 found in /sys/class/hwmon/"
    except Exception as e:
        info["sysfs_detail"] = f"Error: {e}"

    return info
