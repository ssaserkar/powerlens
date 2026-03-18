"""
Batch size energy scaling analysis.

Profiles the same model at different batch sizes to find
the optimal batch size for energy efficiency vs latency.

Key insight: larger batches amortize fixed overhead (memory transfers,
kernel launch) across more inferences, improving energy per inference.
But latency increases linearly.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class BatchResult:
    """Profiling result for one batch size."""
    batch_size: int
    latency_ms: float
    energy_per_batch_j: float
    energy_per_inference_j: float
    avg_power_w: float
    peak_power_w: float
    throughput_inf_per_s: float
    efficiency_inf_per_j: float


@dataclass
class BatchScalingReport:
    """Batch size scaling analysis results."""
    model_name: str
    results: List[BatchResult]
    best_efficiency: Optional[BatchResult] = None
    best_latency: Optional[BatchResult] = None
    sweet_spot: Optional[BatchResult] = None

    def summary(self) -> str:
        lines = [
            "",
            f"Batch Size Energy Scaling — {self.model_name}",
            "=" * 75,
            f"{'Batch':>5s} {'Latency':>10s} {'E/batch':>10s} {'E/inf':>10s} "
            f"{'Throughput':>12s} {'Efficiency':>12s}",
            "-" * 75,
        ]

        for r in self.results:
            lines.append(
                f"{r.batch_size:>5d} "
                f"{r.latency_ms:>9.1f}ms "
                f"{r.energy_per_batch_j:>9.4f}J "
                f"{r.energy_per_inference_j:>9.4f}J "
                f"{r.throughput_inf_per_s:>10.0f} inf/s "
                f"{r.efficiency_inf_per_j:>10.1f} inf/J"
            )

        lines.append("-" * 75)

        if self.best_efficiency and self.results:
            baseline = self.results[0]  # batch=1 or smallest
            speedup = self.best_efficiency.efficiency_inf_per_j / baseline.efficiency_inf_per_j
            lines.append(
                f"Best efficiency:  batch={self.best_efficiency.batch_size} "
                f"({self.best_efficiency.efficiency_inf_per_j:.1f} inf/J, "
                f"{speedup:.1f}x vs batch={baseline.batch_size})"
            )

        if self.best_latency:
            lines.append(
                f"Best latency:     batch={self.best_latency.batch_size} "
                f"({self.best_latency.latency_ms:.1f}ms)"
            )

        if self.sweet_spot and self.sweet_spot != self.best_efficiency:
            lines.append(
                f"Sweet spot (90%): batch={self.sweet_spot.batch_size} "
                f"({self.sweet_spot.efficiency_inf_per_j:.1f} inf/J "
                f"at {self.sweet_spot.latency_ms:.1f}ms)"
            )

        lines.append("")
        return "\n".join(lines)


def find_sweet_spot(results: List[BatchResult]) -> Optional[BatchResult]:
    """Find the batch size that achieves 90% of max efficiency.

    The "sweet spot" balances efficiency and latency — it gets
    most of the efficiency benefit without the full latency cost.
    """
    if not results:
        return None

    max_eff = max(r.efficiency_inf_per_j for r in results)
    threshold = max_eff * 0.9

    # Find smallest batch that exceeds 90% efficiency
    for r in sorted(results, key=lambda x: x.batch_size):
        if r.efficiency_inf_per_j >= threshold:
            return r

    return results[-1]
