"""
Thermal throttling detection.

Monitors CPU/GPU temperature during profiling and detects
when thermal throttling causes:
1. Clock frequency reduction
2. Increased inference latency
3. Increased energy per inference

On Jetson, thermal data is available via sysfs:
    /sys/devices/virtual/thermal/thermal_zone*/temp
    /sys/devices/virtual/thermal/thermal_zone*/type
"""

import glob
import os
import time
import logging
from dataclasses import dataclass
from typing import List, Dict

logger = logging.getLogger(__name__)


@dataclass
class ThermalSample:
    """A single thermal reading."""
    timestamp: float
    zone_name: str
    temperature_c: float


@dataclass
class ThrottleEvent:
    """A detected thermal throttling event."""
    timestamp: float
    zone_name: str
    temperature_c: float
    inference_index: int
    energy_before_j: float
    energy_after_j: float
    energy_increase_pct: float


@dataclass
class ThermalReport:
    """Thermal analysis results."""
    samples: List[ThermalSample]
    throttle_events: List[ThrottleEvent]
    max_temperatures: Dict[str, float]
    avg_temperatures: Dict[str, float]
    throttling_detected: bool
    
    def summary(self) -> str:
        lines = [
            "",
            "Thermal Analysis",
            "=" * 42,
        ]
        
        for zone, temp in sorted(self.max_temperatures.items()):
            avg = self.avg_temperatures.get(zone, 0)
            lines.append(f"  {zone:20s} avg={avg:.1f}°C  max={temp:.1f}°C")
        
        lines.append("")
        
        if self.throttling_detected:
            lines.append(f"⚠ THROTTLING DETECTED: {len(self.throttle_events)} events")
            for event in self.throttle_events[:5]:
                lines.append(
                    f"  Inference #{event.inference_index}: "
                    f"{event.zone_name}={event.temperature_c:.1f}°C, "
                    f"energy +{event.energy_increase_pct:.1f}%"
                )
        else:
            lines.append("✓ No thermal throttling detected")
        
        lines.append("")
        return "\n".join(lines)


def discover_thermal_zones() -> Dict[str, str]:
    """Find all thermal zones and their names.
    
    Returns:
        Dict mapping zone name to sysfs path.
        e.g., {"cpu": "/sys/devices/virtual/thermal/thermal_zone0/temp"}
    """
    zones = {}
    pattern = "/sys/devices/virtual/thermal/thermal_zone*"
    
    for zone_path in sorted(glob.glob(pattern)):
        type_file = os.path.join(zone_path, "type")
        temp_file = os.path.join(zone_path, "temp")
        
        if not os.path.exists(temp_file):
            continue
            
        try:
            with open(type_file, "r") as f:
                zone_name = f.read().strip()
            zones[zone_name] = temp_file
        except OSError:
            continue
    
    return zones


def read_temperatures(zones: Dict[str, str]) -> List[ThermalSample]:
    """Read current temperature from all zones.

    Args:
        zones: Dict from discover_thermal_zones().

    Returns:
        List of ThermalSample with current readings.
    """
    timestamp = time.monotonic()
    samples = []

    for zone_name, temp_file in zones.items():
        try:
            with open(temp_file, "r", errors="replace") as f:
                content = f.read().strip()
            if not content:
                continue
            temp_mc = int(content)
            temp_c = temp_mc / 1000.0
            samples.append(ThermalSample(
                timestamp=timestamp,
                zone_name=zone_name,
                temperature_c=temp_c,
            ))
        except (OSError, ValueError, TypeError):
            continue

    return samples


class ThermalMonitor:
    """Background thermal monitoring during profiling.
    
    Usage:
        monitor = ThermalMonitor()
        monitor.start()
        # ... run workload ...
        monitor.stop()
        report = monitor.analyze(energy_report)
    """
    
    def __init__(self, sample_interval_s: float = 0.5):
        self._interval = sample_interval_s
        self._zones = discover_thermal_zones()
        self._samples: List[ThermalSample] = []
        self._running = False
        self._thread = None
    
    @property
    def available(self) -> bool:
        """Whether thermal monitoring is available on this platform."""
        return len(self._zones) > 0
    
    @property
    def zone_names(self) -> List[str]:
        return list(self._zones.keys())
    
    def start(self):
        """Start background thermal monitoring."""
        if not self._zones:
            logger.warning("No thermal zones found, monitoring disabled")
            return
        
        import threading
        
        self._samples = []
        self._running = True
        self._thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="powerlens-thermal",
        )
        self._thread.start()
        logger.info("Thermal monitor started (%d zones)", len(self._zones))
    
    def stop(self):
        """Stop thermal monitoring."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
    
    def _monitor_loop(self):
        while self._running:
            samples = read_temperatures(self._zones)
            self._samples.extend(samples)
            time.sleep(self._interval)
    
    def read_once(self) -> List[ThermalSample]:
        """Take a single thermal reading (no background thread needed)."""
        return read_temperatures(self._zones)
    
    def analyze(self, energy_report=None, throttle_temp_c: float = 85.0) -> ThermalReport:
        """Analyze thermal data and detect throttling.
        
        Args:
            energy_report: Optional EnergyReport to correlate with.
            throttle_temp_c: Temperature threshold for throttle detection.
        
        Returns:
            ThermalReport with analysis results.
        """
        if not self._samples:
            return ThermalReport(
                samples=[],
                throttle_events=[],
                max_temperatures={},
                avg_temperatures={},
                throttling_detected=False,
            )
        
        # Compute per-zone statistics
        zone_temps: Dict[str, List[float]] = {}
        for sample in self._samples:
            if sample.zone_name not in zone_temps:
                zone_temps[sample.zone_name] = []
            zone_temps[sample.zone_name].append(sample.temperature_c)
        
        max_temps = {name: max(temps) for name, temps in zone_temps.items()}
        avg_temps = {name: sum(temps) / len(temps) for name, temps in zone_temps.items()}
        
        # Detect throttling events
        throttle_events = []
        
        if energy_report and energy_report.inferences:
            inferences = energy_report.inferences
            
            # Check if any zone exceeded throttle threshold
            for sample in self._samples:
                if sample.temperature_c >= throttle_temp_c:
                    # Find the closest inference
                    closest_inf = None
                    min_dist = float("inf")
                    for inf in inferences:
                        dist = abs(sample.timestamp - inf.start_time)
                        if dist < min_dist:
                            min_dist = dist
                            closest_inf = inf
                    
                    if closest_inf and closest_inf.index > 0:
                        # Compare energy with earlier inferences
                        early_infs = [i for i in inferences if i.index < 5]
                        if early_infs:
                            early_avg = sum(i.energy_j for i in early_infs) / len(early_infs)
                            if early_avg > 0:
                                increase_pct = ((closest_inf.energy_j - early_avg) / early_avg) * 100
                                
                                throttle_events.append(ThrottleEvent(
                                    timestamp=sample.timestamp,
                                    zone_name=sample.zone_name,
                                    temperature_c=sample.temperature_c,
                                    inference_index=closest_inf.index,
                                    energy_before_j=early_avg,
                                    energy_after_j=closest_inf.energy_j,
                                    energy_increase_pct=increase_pct,
                                ))
        
        # Deduplicate events (one per inference)
        seen_inferences = set()
        unique_events = []
        for event in throttle_events:
            if event.inference_index not in seen_inferences:
                seen_inferences.add(event.inference_index)
                unique_events.append(event)
        
        return ThermalReport(
            samples=self._samples,
            throttle_events=unique_events,
            max_temperatures=max_temps,
            avg_temperatures=avg_temps,
            throttling_detected=len(unique_events) > 0,
        )