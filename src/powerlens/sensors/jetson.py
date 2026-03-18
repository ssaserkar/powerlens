"""
Jetson platform sensor configurations.

Pre-built configurations for known Jetson platforms so users
don't need to read schematics or datasheets.

To add a new platform:
1. Find the INA3221 I2C bus and address: i2cdetect -y -r 1
2. Find rail names: check carrier board schematic or /sys/bus/i2c/devices/
3. Find shunt resistor values: carrier board schematic
4. Add a config dict below
5. Submit a PR!
"""

from powerlens.sensors.ina3221 import INA3221, ChannelConfig

# ============================================================
# Jetson Orin Nano Developer Kit
# ============================================================
# Carrier board: NVIDIA reference carrier (P3768)
# INA3221 at address 0x40 on I2C bus 1
#
# Verify on your board:
#   sudo i2cdetect -y -r 1
#   ls /sys/bus/i2c/devices/1-0040/
#
# Rail mapping (from carrier board schematic):
#   Channel 1: VDD_GPU_SOC — GPU and SoC power
#   Channel 2: VDD_CPU_CV  — CPU and computer vision
#   Channel 3: VIN_SYS_5V0 — System 5V input
#
# Shunt resistors: 5 milliohm (0.005 ohm)
# NOTE: Verify shunt values on YOUR specific board revision.
#       These values are for the reference developer kit.

ORIN_NANO_CONFIG = {
    "name": "Jetson Orin Nano Developer Kit",
    "bus_number": 1,
    "address": 0x40,
    "channels": [
        ChannelConfig(
            channel=1,
            rail_name="VDD_GPU_SOC",
            shunt_resistor_ohm=0.005,
        ),
        ChannelConfig(
            channel=2,
            rail_name="VDD_CPU_CV",
            shunt_resistor_ohm=0.005,
        ),
        ChannelConfig(
            channel=3,
            rail_name="VIN_SYS_5V0",
            shunt_resistor_ohm=0.005,
        ),
    ],
}

# ============================================================
# Add more platforms here as they are tested
# ============================================================
# ORIN_AGX_CONFIG = { ... }
# XAVIER_NX_CONFIG = { ... }

# Map of known platforms
PLATFORM_CONFIGS = {
    "orin-nano": ORIN_NANO_CONFIG,
}


def create_jetson_sensor(platform: str = "orin-nano") -> INA3221:
    """Create an INA3221 sensor for a known Jetson platform.

    Args:
        platform: Platform identifier. Currently supported:
                  "orin-nano" — Jetson Orin Nano Developer Kit

    Returns:
        Configured INA3221 sensor. Caller must call .open() or use
        as context manager.

    Raises:
        ValueError: If platform is not recognized.
    """
    if platform not in PLATFORM_CONFIGS:
        available = ", ".join(sorted(PLATFORM_CONFIGS.keys()))
        raise ValueError(
            f"Unknown platform '{platform}'. "
            f"Available: {available}"
        )

    config = PLATFORM_CONFIGS[platform]
    return INA3221(
        bus_number=config["bus_number"],
        address=config["address"],
        channels=config["channels"],
    )
