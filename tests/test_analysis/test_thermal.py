"""Tests for thermal monitoring."""


from powerlens.analysis.thermal import (
    ThermalMonitor,
    ThermalSample,
    ThrottleEvent,
    ThermalReport,
)


def test_thermal_sample_dataclass():
    """ThermalSample should store all fields."""
    sample = ThermalSample(
        timestamp=1.0,
        zone_name="gpu-thermal",
        temperature_c=45.5,
    )
    assert sample.zone_name == "gpu-thermal"
    assert sample.temperature_c == 45.5


def test_throttle_event_dataclass():
    """ThrottleEvent should store all fields."""
    event = ThrottleEvent(
        timestamp=1.0,
        zone_name="gpu-thermal",
        temperature_c=90.0,
        inference_index=42,
        energy_before_j=0.5,
        energy_after_j=0.8,
        energy_increase_pct=60.0,
    )
    assert event.temperature_c == 90.0
    assert event.energy_increase_pct == 60.0


def test_thermal_report_no_throttling():
    """ThermalReport with no events should report no throttling."""
    report = ThermalReport(
        samples=[],
        throttle_events=[],
        max_temperatures={"gpu-thermal": 45.0},
        avg_temperatures={"gpu-thermal": 42.0},
        throttling_detected=False,
    )
    assert report.throttling_detected is False
    summary = report.summary()
    assert "No thermal throttling" in summary


def test_thermal_report_with_throttling():
    """ThermalReport with events should report throttling."""
    event = ThrottleEvent(
        timestamp=1.0,
        zone_name="gpu-thermal",
        temperature_c=92.0,
        inference_index=50,
        energy_before_j=0.5,
        energy_after_j=0.8,
        energy_increase_pct=60.0,
    )
    report = ThermalReport(
        samples=[],
        throttle_events=[event],
        max_temperatures={"gpu-thermal": 92.0},
        avg_temperatures={"gpu-thermal": 85.0},
        throttling_detected=True,
    )
    assert report.throttling_detected is True
    summary = report.summary()
    assert "THROTTLING DETECTED" in summary


def test_thermal_monitor_creation():
    """ThermalMonitor should create without errors."""
    monitor = ThermalMonitor(sample_interval_s=1.0)
    assert monitor is not None


def test_thermal_monitor_analyze_empty():
    """Analyzing with no samples should return empty report."""
    monitor = ThermalMonitor()
    report = monitor.analyze()
    assert report.throttling_detected is False
    assert len(report.samples) == 0
    assert len(report.throttle_events) == 0
