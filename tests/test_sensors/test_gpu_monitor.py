"""Tests for GPU utilization monitor."""

import platform

from powerlens.sensors.gpu_monitor import GpuMonitor, GpuSample


def test_gpu_monitor_creation():
    """GpuMonitor should create without errors."""
    monitor = GpuMonitor(sample_interval_s=0.1)
    assert monitor is not None


def test_gpu_monitor_available_on_non_linux():
    """On non-Linux, GPU monitor should not be available."""
    if platform.system() != "Linux":
        monitor = GpuMonitor()
        assert monitor.available is False


def test_gpu_sample_dataclass():
    """GpuSample should store all fields."""
    sample = GpuSample(
        timestamp=1.0,
        gpu_util_pct=50.0,
        gpu_freq_mhz=1020.0,
        emc_util_pct=30.0,
        emc_freq_mhz=2133.0,
    )
    assert sample.gpu_util_pct == 50.0
    assert sample.gpu_freq_mhz == 1020.0
    assert sample.timestamp == 1.0


def test_gpu_monitor_summary_empty():
    """Empty monitor should return empty summary."""
    monitor = GpuMonitor()
    summary = monitor.get_summary()
    assert summary == {}


def test_gpu_monitor_format_summary_empty():
    """Empty monitor should return a string."""
    monitor = GpuMonitor()
    text = monitor.format_summary()
    assert isinstance(text, str)
    assert "no data" in text.lower()
