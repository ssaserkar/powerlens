"""Tests for batch scaling analysis."""

from powerlens.analysis.batch_scaling import (
    BatchResult,
    BatchScalingReport,
    find_sweet_spot,
)


def _make_result(batch, energy, power, latency):
    """Helper to create a BatchResult."""
    return BatchResult(
        batch_size=batch,
        latency_ms=latency,
        energy_per_batch_j=energy * batch,
        energy_per_inference_j=energy,
        avg_power_w=power,
        peak_power_w=power + 2.0,
        throughput_inf_per_s=batch / (latency / 1000),
        efficiency_inf_per_j=1.0 / energy,
    )


def test_batch_result_dataclass():
    """BatchResult should store all fields."""
    result = _make_result(batch=4, energy=0.01, power=15.0, latency=5.0)
    assert result.batch_size == 4
    assert result.energy_per_inference_j == 0.01
    assert result.avg_power_w == 15.0


def test_batch_scaling_report_summary():
    """BatchScalingReport should produce readable summary."""
    results = [
        _make_result(1, 0.02, 12.0, 2.0),
        _make_result(10, 0.015, 15.0, 20.0),
        _make_result(100, 0.012, 18.0, 200.0),
    ]
    report = BatchScalingReport(
        model_name="test_model",
        results=results,
        best_efficiency=results[2],
        best_latency=results[0],
    )
    summary = report.summary()
    assert "test_model" in summary
    assert "Batch" in summary


def test_find_sweet_spot():
    """Sweet spot should find smallest batch at 90% efficiency."""
    results = [
        _make_result(1, 0.020, 12.0, 2.0),    # 50 inf/J
        _make_result(4, 0.012, 15.0, 8.0),     # 83 inf/J
        _make_result(8, 0.011, 17.0, 16.0),    # 91 inf/J
        _make_result(16, 0.010, 18.0, 32.0),   # 100 inf/J
    ]
    sweet = find_sweet_spot(results)
    assert sweet is not None
    # 90% of 100 = 90 inf/J, batch=8 has 91 inf/J
    assert sweet.batch_size == 8


def test_find_sweet_spot_empty():
    """Empty results should return None."""
    assert find_sweet_spot([]) is None


def test_find_sweet_spot_single():
    """Single result should return that result."""
    results = [_make_result(1, 0.02, 12.0, 2.0)]
    sweet = find_sweet_spot(results)
    assert sweet.batch_size == 1
