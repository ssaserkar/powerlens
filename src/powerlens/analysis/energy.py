"""
Energy computation engine.

Takes timestamped power samples and inference timestamps,
aligns them, and computes energy (joules) per inference
using numerical integration (trapezoidal rule).

Energy = integral of Power over Time
1 Joule = 1 Watt * 1 Second

This module is the core intellectual contribution of PowerLens.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional

from powerlens.sensors.types import PowerSample


@dataclass
class InferenceEnergy:
    """Energy measurement for a single inference."""

    index: int
    start_time: float
    end_time: float
    duration_s: float
    energy_j: float
    avg_power_w: float
    peak_power_w: float
    rail_energy_j: Dict[str, float] = field(default_factory=dict)


@dataclass
class EnergyReport:
    """Complete energy profiling report."""

    inferences: List[InferenceEnergy]
    num_inferences: int = 0
    mean_energy_j: float = 0.0
    std_energy_j: float = 0.0
    min_energy_j: float = 0.0
    max_energy_j: float = 0.0
    mean_power_w: float = 0.0
    peak_power_w: float = 0.0
    idle_power_w: float = 0.0
    total_energy_j: float = 0.0
    total_duration_s: float = 0.0
    actual_sample_rate_hz: float = 0.0
    rail_breakdown: Dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        """Return a human-readable summary string."""
        lines = [
            "",
            "PowerLens Inference Energy Report",
            "=" * 42,
            f"Inferences:         {self.num_inferences}",
            f"Sample rate:        {self.actual_sample_rate_hz:.1f} Hz",
            "",
            f"Energy/inference:   {self.mean_energy_j:.4f} +/- {self.std_energy_j:.4f} J",
            f"  Min:              {self.min_energy_j:.4f} J",
            f"  Max:              {self.max_energy_j:.4f} J",
            "",
            f"Power (avg):        {self.mean_power_w:.2f} W",
            f"Power (peak):       {self.peak_power_w:.2f} W",
            f"Power (idle):       {self.idle_power_w:.2f} W",
            "",
            f"Total energy:       {self.total_energy_j:.4f} J",
            f"Total duration:     {self.total_duration_s:.2f} s",
            "",
            f"Efficiency:         {self.num_inferences / self.total_energy_j:.1f} inferences/J"
            if self.total_energy_j > 0 else "Efficiency:         N/A",
            "",
        ]

        if self.rail_breakdown:
            lines.append("Rail breakdown (avg power):")
            total_rail = sum(self.rail_breakdown.values())
            for rail_name, power_w in sorted(
                self.rail_breakdown.items(), key=lambda x: x[1], reverse=True
            ):
                pct = (power_w / total_rail * 100) if total_rail > 0 else 0
                lines.append(f"  {rail_name:20s} {power_w:.2f} W ({pct:.0f}%)")
            lines.append("")

        return "\n".join(lines)


def _flatten_samples(sample_cycles: List[List[PowerSample]]) -> Dict[str, List]:
    """Convert list of sample cycles into per-rail arrays.

    Args:
        sample_cycles: List where each element is a list of PowerSample
                       (one per channel) from a single read cycle.

    Returns:
        Dict with keys:
            "timestamps": numpy array of timestamps
            "total_power": numpy array of total power per cycle
            "rails": dict of rail_name -> {"power": array, "voltage": array}
    """
    if not sample_cycles:
        return {"timestamps": np.array([]), "total_power": np.array([]), "rails": {}}

    timestamps = []
    total_power = []
    rails: Dict[str, Dict[str, list]] = {}

    for cycle in sample_cycles:
        # Use timestamp from first channel in cycle
        timestamps.append(cycle[0].timestamp)
        cycle_total = 0.0

        for sample in cycle:
            if sample.rail_name not in rails:
                rails[sample.rail_name] = {"power": [], "voltage": []}
            rails[sample.rail_name]["power"].append(sample.power_w)
            rails[sample.rail_name]["voltage"].append(sample.voltage_v)
            cycle_total += sample.power_w

        total_power.append(cycle_total)

    result_rails = {}
    for name, data in rails.items():
        result_rails[name] = {
            "power": np.array(data["power"]),
            "voltage": np.array(data["voltage"]),
        }

    return {
        "timestamps": np.array(timestamps),
        "total_power": np.array(total_power),
        "rails": result_rails,
    }


def _compute_inference_energy(
    timestamps: np.ndarray,
    total_power: np.ndarray,
    rails: Dict[str, Dict[str, np.ndarray]],
    inf_start: float,
    inf_end: float,
    index: int,
) -> Optional[InferenceEnergy]:
    """Compute energy for a single inference by integrating power over time.

    Uses the trapezoidal rule for numerical integration.
    Only uses power samples that fall within the inference time window.
    """
    # Find samples within this inference window
    mask = (timestamps >= inf_start) & (timestamps <= inf_end)
    window_times = timestamps[mask]
    window_power = total_power[mask]

    if len(window_times) < 2:
        # Not enough samples to integrate — inference was too fast
        # for our sample rate. Estimate using nearest samples.
        if len(window_times) == 1:
            duration = inf_end - inf_start
            energy = window_power[0] * duration
            return InferenceEnergy(
                index=index,
                start_time=inf_start,
                end_time=inf_end,
                duration_s=duration,
                energy_j=energy,
                avg_power_w=window_power[0],
                peak_power_w=window_power[0],
            )
        return None

    # Trapezoidal integration: energy = integral(power, dt)
    duration = inf_end - inf_start
    energy = float(np.trapz(window_power, window_times))

    # Per-rail energy breakdown
    rail_energy = {}
    for rail_name, rail_data in rails.items():
        rail_power = rail_data["power"][mask]
        if len(rail_power) >= 2:
            rail_energy[rail_name] = float(np.trapz(rail_power, window_times))

    return InferenceEnergy(
        index=index,
        start_time=inf_start,
        end_time=inf_end,
        duration_s=duration,
        energy_j=energy,
        avg_power_w=float(np.mean(window_power)),
        peak_power_w=float(np.max(window_power)),
        rail_energy_j=rail_energy,
    )


def compute_energy_report(
    samples: List[List[PowerSample]],
    inference_timestamps: List[tuple],
    idle_samples: Optional[List[List[PowerSample]]] = None,
) -> EnergyReport:
    """Compute a complete energy report from samples and inference timestamps.

    Args:
        samples: Power samples collected during inference.
                 Each element is a list of PowerSample from one read cycle.
        inference_timestamps: List of (start_time, end_time) tuples,
                              one per inference, using time.monotonic().
        idle_samples: Optional power samples collected during idle.
                      Used to compute idle baseline power.

    Returns:
        EnergyReport with per-inference and aggregate statistics.
    """
    flat = _flatten_samples(samples)
    timestamps = flat["timestamps"]
    total_power = flat["total_power"]
    rails = flat["rails"]

    # Compute idle baseline
    idle_power = 0.0
    if idle_samples:
        idle_flat = _flatten_samples(idle_samples)
        if len(idle_flat["total_power"]) > 0:
            idle_power = float(np.mean(idle_flat["total_power"]))

    # Compute energy for each inference
    inferences = []
    for i, (start, end) in enumerate(inference_timestamps):
        result = _compute_inference_energy(
            timestamps, total_power, rails, start, end, i
        )
        if result is not None:
            inferences.append(result)

    # Aggregate statistics
    if not inferences:
        return EnergyReport(inferences=[], idle_power_w=idle_power)

    energies = np.array([inf.energy_j for inf in inferences])
    powers = np.array([inf.avg_power_w for inf in inferences])
    peak_powers = np.array([inf.peak_power_w for inf in inferences])

    # Compute actual sample rate
    actual_rate = 0.0
    if len(timestamps) > 1:
        total_time = timestamps[-1] - timestamps[0]
        if total_time > 0:
            actual_rate = len(timestamps) / total_time

    # Per-rail average power breakdown
    rail_breakdown = {}
    for rail_name, rail_data in rails.items():
        rail_breakdown[rail_name] = float(np.mean(rail_data["power"]))

    return EnergyReport(
        inferences=inferences,
        num_inferences=len(inferences),
        mean_energy_j=float(np.mean(energies)),
        std_energy_j=float(np.std(energies)),
        min_energy_j=float(np.min(energies)),
        max_energy_j=float(np.max(energies)),
        mean_power_w=float(np.mean(powers)),
        peak_power_w=float(np.max(peak_powers)),
        idle_power_w=idle_power,
        total_energy_j=float(np.sum(energies)),
        total_duration_s=float(np.sum([inf.duration_s for inf in inferences])),
        actual_sample_rate_hz=actual_rate,
        rail_breakdown=rail_breakdown,
    )