"""
Export energy reports to CSV files.

Two CSV files are generated:
1. Summary CSV: one row per inference with energy stats
2. Raw CSV: all power samples with timestamps
"""

import csv
from pathlib import Path
from typing import List

from powerlens.analysis.energy import EnergyReport
from powerlens.sensors.types import PowerSample


def export_summary_csv(report: EnergyReport, filepath: str) -> str:
    """Export per-inference energy summary to CSV.

    Args:
        report: EnergyReport from profiling session.
        filepath: Output file path.

    Returns:
        Absolute path to the written file.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)

        # Header
        writer.writerow([
            "inference_index",
            "start_time_s",
            "end_time_s",
            "duration_s",
            "energy_j",
            "avg_power_w",
            "peak_power_w",
        ])

        # Data rows
        for inf in report.inferences:
            writer.writerow([
                inf.index,
                f"{inf.start_time:.6f}",
                f"{inf.end_time:.6f}",
                f"{inf.duration_s:.6f}",
                f"{inf.energy_j:.6f}",
                f"{inf.avg_power_w:.4f}",
                f"{inf.peak_power_w:.4f}",
            ])

    return str(path.resolve())


def export_raw_csv(
    samples: List[List[PowerSample]], filepath: str
) -> str:
    """Export raw power samples to CSV.

    Args:
        samples: List of sample cycles from PowerSampler.get_samples().
        filepath: Output file path.

    Returns:
        Absolute path to the written file.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)

        # Header
        writer.writerow([
            "timestamp_s",
            "channel",
            "rail_name",
            "voltage_v",
            "current_a",
            "power_w",
        ])

        # Data rows
        for cycle in samples:
            for sample in cycle:
                writer.writerow([
                    f"{sample.timestamp:.6f}",
                    sample.channel,
                    sample.rail_name,
                    f"{sample.voltage_v:.4f}",
                    f"{sample.current_a:.4f}",
                    f"{sample.power_w:.4f}",
                ])

    return str(path.resolve())
