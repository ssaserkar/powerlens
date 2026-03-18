"""
Jetson power mode analysis.

Profiles the same model across different NVPMode settings
to find the most energy-efficient configuration.

Jetson power modes are controlled via:
    sudo nvpmodel -m <mode_id>
    nvpmodel -q  (query current mode)

Common modes on Orin Nano:
    0: MAXN (max performance, max power)
    1: 15W
    2: 7W
"""

import subprocess
import logging
import time
from dataclasses import dataclass
from typing import List, Optional, Dict

logger = logging.getLogger(__name__)


@dataclass
class PowerModeResult:
    """Profiling result for one power mode."""
    mode_id: int
    mode_name: str
    energy_per_inference_j: float
    avg_power_w: float
    peak_power_w: float
    latency_ms: float
    efficiency_inf_per_j: float
    num_inferences: int


@dataclass
class PowerModeReport:
    """Comparison across power modes."""
    results: List[PowerModeResult]
    best_efficiency: Optional[PowerModeResult] = None
    best_latency: Optional[PowerModeResult] = None

    def summary(self) -> str:
        lines = [
            "",
            "Power Mode Comparison",
            "=" * 70,
            f"{'Mode':15s} {'Latency':>10s} {'Energy/inf':>12s} {'Avg Power':>12s} {'Efficiency':>12s}",
            "-" * 70,
        ]

        for r in self.results:
            lines.append(
                f"{r.mode_name:15s} "
                f"{r.latency_ms:>9.1f}ms "
                f"{r.energy_per_inference_j:>11.4f}J "
                f"{r.avg_power_w:>11.2f}W "
                f"{r.efficiency_inf_per_j:>10.1f} inf/J"
            )

        lines.append("-" * 70)

        if self.best_efficiency:
            lines.append(f"Most efficient: {self.best_efficiency.mode_name} "
                        f"({self.best_efficiency.efficiency_inf_per_j:.1f} inf/J)")
        if self.best_latency:
            lines.append(f"Lowest latency: {self.best_latency.mode_name} "
                        f"({self.best_latency.latency_ms:.1f} ms)")

        lines.append("")
        return "\n".join(lines)


def get_current_power_mode() -> Dict[str, str]:
    """Get current NVPModel mode.

    Returns:
        Dict with 'id' and 'name' keys.
    """
    try:
        result = subprocess.run(
            ["nvpmodel", "-q"],
            capture_output=True, text=True, timeout=5,
        )
        output = result.stdout.strip()

        mode_name = "unknown"
        mode_id = -1

        for line in output.split("\n"):
            line = line.strip()
            if "Power Mode:" in line or "NV Power Mode:" in line:
                mode_name = line.split(":")[-1].strip()
            if "MODE_ID:" in line or "MODE_ID :" in line:
                try:
                    mode_id = int(line.split(":")[-1].strip())
                except ValueError:
                    pass

        # If MODE_ID not found, try to match name against available modes
        if mode_id == -1 and mode_name != "unknown":
            modes = get_available_modes()
            for m in modes:
                if m["name"].lower() == mode_name.lower():
                    mode_id = m["id"]
                    break

        return {"id": mode_id, "name": mode_name}
    except Exception as e:
        logger.warning("Could not query power mode: %s", e)
        return {"id": -1, "name": "unknown"}


def set_power_mode(mode_id: int) -> bool:
    """Set NVPModel power mode.

    Some modes require a reboot and cannot be switched on-the-fly.
    Returns False for those modes with a clear message.
    """
    import os

    if os.geteuid() == 0:
        cmd = ["nvpmodel", "-m", str(mode_id)]
    else:
        cmd = ["sudo", "nvpmodel", "-m", str(mode_id)]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True, text=True, timeout=30,
            input="no\n",  # Auto-answer "no" to reboot prompt
        )

        combined_output = result.stdout + result.stderr

        if "Reboot required" in combined_output or "REBOOT" in combined_output:
            logger.warning(
                "Power mode %d requires a reboot and cannot be tested on-the-fly",
                mode_id,
            )
            print(f"  NOTE: Mode {mode_id} requires reboot — skipping")
            return False

        if result.returncode == 0:
            time.sleep(3.0)
            logger.info("Power mode set to %d", mode_id)
            return True
        else:
            logger.error("Failed to set power mode: %s", combined_output.strip())
            return False
    except subprocess.TimeoutExpired:
        logger.error("Timeout setting power mode %d", mode_id)
        return False
    except Exception as e:
        logger.error("Could not set power mode: %s", e)
        return False


def get_available_modes() -> List[Dict[str, str]]:
    """List available power modes.

    Returns:
        List of dicts with 'id' and 'name'.
    """
    try:
        result = subprocess.run(
            ["nvpmodel", "-p", "--verbose"],
            capture_output=True, text=True, timeout=5,
        )

        modes = []
        for line in result.stdout.split("\n"):
            # Parse lines like "POWER_MODEL ID=0 NAME=MAXN"
            if "POWER_MODEL" in line and "ID=" in line:
                parts = line.strip().split()
                mode_id = -1
                mode_name = "unknown"
                for part in parts:
                    if part.startswith("ID="):
                        mode_id = int(part.split("=")[1])
                    if part.startswith("NAME="):
                        mode_name = part.split("=")[1]
                modes.append({"id": mode_id, "name": mode_name})

        return modes if modes else [{"id": 0, "name": "MAXN"}]
    except Exception as e:
        logger.warning("Could not list power modes: %s", e)
        return []
