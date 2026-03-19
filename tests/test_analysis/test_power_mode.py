"""Tests for power mode analysis."""

from powerlens.analysis.power_mode import (
    PowerModeResult,
    PowerModeReport,
)


def test_power_mode_result_dataclass():
    """PowerModeResult should store all fields."""
    result = PowerModeResult(
        mode_id=0,
        mode_name="15W",
        energy_per_inference_j=0.015,
        avg_power_w=11.6,
        peak_power_w=12.0,
        latency_ms=2.9,
        efficiency_inf_per_j=66.7,
        num_inferences=30,
    )
    assert result.mode_name == "15W"
    assert result.energy_per_inference_j == 0.015


def test_power_mode_report_summary():
    """PowerModeReport should produce readable summary."""
    results = [
        PowerModeResult(0, "15W", 0.015, 11.6, 12.0, 2.9, 66.7, 30),
        PowerModeResult(1, "25W", 0.014, 11.8, 12.5, 2.9, 71.4, 30),
        PowerModeResult(2, "MAXN", 0.015, 12.1, 13.0, 2.8, 66.7, 30),
    ]
    report = PowerModeReport(
        results=results,
        best_efficiency=results[1],
        best_latency=results[2],
    )
    summary = report.summary()
    assert "15W" in summary
    assert "25W" in summary
    assert "MAXN" in summary
    assert "Most efficient" in summary


def test_power_mode_report_empty():
    """Empty report should not crash."""
    report = PowerModeReport(results=[])
    summary = report.summary()
    assert isinstance(summary, str)
