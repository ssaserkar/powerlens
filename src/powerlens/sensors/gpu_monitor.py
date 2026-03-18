"""
GPU utilization monitor for NVIDIA Jetson.

Reads GPU utilization, memory usage, and clock frequencies
from Jetson sysfs to correlate with power measurements.

On Jetson, GPU stats are available at:
    /sys/devices/gpu.0/load              — GPU utilization (0-1000 = 0-100%)
    /sys/devices/gpu.0/railgate_enable   — power gating status
    /sys/devices/17000000.ga10b/devfreq/17000000.ga10b/cur_freq — current clock
"""

import os
import time
import threading
import logging
from dataclasses import dataclass
from typing import List, Optional, Dict

logger = logging.getLogger(__name__)


@dataclass
class GpuSample:
    """A single GPU utilization reading."""
    timestamp: float
    gpu_util_pct: float      # 0-100%
    gpu_freq_mhz: float      # Current clock in MHz
    emc_util_pct: float      # Memory controller utilization
    emc_freq_mhz: float      # Memory clock in MHz


# Known sysfs paths for Jetson GPU stats
_GPU_LOAD_PATHS = [
    "/sys/devices/gpu.0/load",
    "/sys/devices/platform/gpu.0/load",
]

_GPU_FREQ_PATHS = [
    "/sys/devices/platform/bus@0/17000000.gpu/devfreq/17000000.gpu/cur_freq",
    "/sys/class/devfreq/17000000.gpu/cur_freq",
    "/sys/devices/17000000.ga10b/devfreq/17000000.ga10b/cur_freq",
    "/sys/devices/gpu.0/devfreq/gpu.0/cur_freq",
]

_EMC_FREQ_PATHS = [
    "/sys/kernel/actmon_avg_activity/mc_all",
    "/sys/devices/platform/17000000.ga10b/devfreq/17000000.ga10b/device/of_node/../emc/cur_freq",
]


def _find_readable_path(paths: List[str]) -> Optional[str]:
    """Find the first readable path from a list of candidates."""
    for path in paths:
        if os.path.exists(path) and os.access(path, os.R_OK):
            return path
    return None


def _read_int(path: str) -> Optional[int]:
    """Read an integer from a sysfs file."""
    try:
        with open(path, "r") as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


class GpuMonitor:
    """Monitor GPU utilization and clock frequencies.

    Usage:
        monitor = GpuMonitor()
        if monitor.available:
            monitor.start()
            # ... run workload ...
            monitor.stop()
            for sample in monitor.samples:
                print(f"GPU: {sample.gpu_util_pct:.0f}% at {sample.gpu_freq_mhz:.0f}MHz")
    """

    def __init__(self, sample_interval_s: float = 0.1):
        self._interval = sample_interval_s
        self._gpu_load_path = _find_readable_path(_GPU_LOAD_PATHS)
        self._gpu_freq_path = _find_readable_path(_GPU_FREQ_PATHS)
        self._samples: List[GpuSample] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None

    @property
    def available(self) -> bool:
        """Whether GPU monitoring is available."""
        return self._gpu_load_path is not None

    def read_once(self) -> Optional[GpuSample]:
        """Take a single GPU reading."""
        timestamp = time.monotonic()

        gpu_util = 0.0
        if self._gpu_load_path:
            val = _read_int(self._gpu_load_path)
            if val is not None:
                gpu_util = val / 10.0  # 0-1000 → 0-100%

        gpu_freq = 0.0
        if self._gpu_freq_path:
            val = _read_int(self._gpu_freq_path)
            if val is not None:
                gpu_freq = val / 1_000_000.0  # Hz → MHz

        return GpuSample(
            timestamp=timestamp,
            gpu_util_pct=gpu_util,
            gpu_freq_mhz=gpu_freq,
            emc_util_pct=0.0,
            emc_freq_mhz=0.0,
        )

    def start(self):
        """Start background GPU monitoring."""
        if not self.available:
            logger.warning("GPU monitoring not available")
            return

        self._samples = []
        self._running = True
        self._thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="powerlens-gpu-monitor",
        )
        self._thread.start()
        logger.info("GPU monitor started")

    def stop(self):
        """Stop GPU monitoring."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _monitor_loop(self):
        while self._running:
            sample = self.read_once()
            if sample:
                self._samples.append(sample)
            time.sleep(self._interval)

    @property
    def samples(self) -> List[GpuSample]:
        return list(self._samples)

    def get_summary(self) -> Dict[str, float]:
        """Get summary statistics."""
        if not self._samples:
            return {}

        utils = [s.gpu_util_pct for s in self._samples]
        freqs = [s.gpu_freq_mhz for s in self._samples if s.gpu_freq_mhz > 0]

        summary = {
            "gpu_util_avg_pct": sum(utils) / len(utils),
            "gpu_util_max_pct": max(utils),
            "gpu_util_min_pct": min(utils),
        }

        if freqs:
            summary["gpu_freq_avg_mhz"] = sum(freqs) / len(freqs)
            summary["gpu_freq_max_mhz"] = max(freqs)

        return summary

    def format_summary(self) -> str:
        """Format summary as readable string."""
        s = self.get_summary()
        if not s:
            return "GPU monitoring: no data"

        lines = [
            "",
            "GPU Utilization",
            "=" * 42,
            f"  GPU util:  avg={s.get('gpu_util_avg_pct', 0):.0f}%  "
            f"max={s.get('gpu_util_max_pct', 0):.0f}%  "
            f"min={s.get('gpu_util_min_pct', 0):.0f}%",
        ]

        if "gpu_freq_avg_mhz" in s:
            lines.append(
                f"  GPU freq:  avg={s['gpu_freq_avg_mhz']:.0f}MHz  "
                f"max={s.get('gpu_freq_max_mhz', 0):.0f}MHz"
            )

        lines.append("")
        return "\n".join(lines)