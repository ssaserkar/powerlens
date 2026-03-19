"""Tests for text report generation."""

import os
import tempfile

from powerlens.analysis.energy import EnergyReport, InferenceEnergy
from powerlens.export.report import generate_text_report


def _make_report():
    """Create a minimal EnergyReport for testing."""
    inferences = [
        InferenceEnergy(
            index=0, start_time=0.0, end_time=0.05,
            duration_s=0.05, energy_j=0.5,
            avg_power_w=10.0, peak_power_w=12.0,
        ),
        InferenceEnergy(
            index=1, start_time=0.1, end_time=0.15,
            duration_s=0.05, energy_j=0.6,
            avg_power_w=12.0, peak_power_w=14.0,
        ),
    ]
    return EnergyReport(
        inferences=inferences,
        num_inferences=2,
        mean_energy_j=0.55,
        std_energy_j=0.05,
        min_energy_j=0.5,
        max_energy_j=0.6,
        mean_power_w=11.0,
        peak_power_w=14.0,
        idle_power_w=7.0,
        total_energy_j=1.1,
        total_duration_s=0.1,
        actual_sample_rate_hz=100.0,
        rail_breakdown={"VDD_IN": 7.0, "VDD_CPU_GPU_CV": 2.5, "VDD_SOC": 1.5},
    )


def test_generate_text_report():
    """Should generate a text file with report content."""
    report = _make_report()

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_report.txt")
        result = generate_text_report(
            report=report,
            model_name="test_model",
            platform="Test Platform",
            output_path=path,
        )
        assert os.path.exists(result)

        with open(result, "r") as f:
            content = f.read()

        assert "PowerLens" in content
        assert "test_model" in content
        assert "Test Platform" in content
        assert "Energy/inference" in content


def test_generate_text_report_with_thermal():
    """Should include thermal data when provided."""
    from powerlens.analysis.thermal import ThermalReport

    report = _make_report()
    thermal = ThermalReport(
        samples=[],
        throttle_events=[],
        max_temperatures={"gpu-thermal": 45.0},
        avg_temperatures={"gpu-thermal": 42.0},
        throttling_detected=False,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_report.txt")
        result = generate_text_report(
            report=report,
            thermal_report=thermal,
            model_name="test_model",
            output_path=path,
        )

        with open(result, "r") as f:
            content = f.read()

        assert "Thermal" in content or "thermal" in content


def test_generate_text_report_with_gpu():
    """Should include GPU data when provided."""
    report = _make_report()
    gpu_summary = {
        "gpu_util_avg_pct": 55.0,
        "gpu_util_max_pct": 94.0,
        "gpu_util_min_pct": 0.0,
        "gpu_freq_avg_mhz": 1020.0,
        "gpu_freq_max_mhz": 1020.0,
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_report.txt")
        result = generate_text_report(
            report=report,
            gpu_summary=gpu_summary,
            model_name="test_model",
            output_path=path,
        )

        with open(result, "r") as f:
            content = f.read()

        assert "GPU" in content
        assert "55" in content or "util" in content.lower()
