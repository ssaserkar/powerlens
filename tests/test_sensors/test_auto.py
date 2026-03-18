"""Tests for sensor auto-detection."""

from powerlens.sensors.auto import detect_sensor, get_sensor_info


def test_detect_sensor_returns_mock_on_windows():
    """On non-Linux, detect_sensor should return MockSensor."""
    sensor = detect_sensor(use_mock_fallback=True)
    # On Windows (where tests run), should get MockSensor
    assert sensor is not None
    # Should be usable
    sensor.open()
    samples = sensor.read_all()
    assert len(samples) > 0
    sensor.close()


def test_detect_sensor_no_fallback_on_windows():
    """On non-Linux without fallback, should raise or return mock."""
    import platform
    if platform.system() != "Linux":
        raised = False
        try:
            detect_sensor(use_mock_fallback=False)
        except RuntimeError:
            raised = True
        assert raised


def test_get_sensor_info_returns_dict():
    """get_sensor_info should return a complete info dict."""
    info = get_sensor_info()
    assert isinstance(info, dict)
    assert "platform" in info
    assert "i2c_available" in info
    assert "sysfs_available" in info
    assert "mock_available" in info
    assert "recommended" in info
    assert info["mock_available"] is True


def test_get_sensor_info_recommends_mock_on_windows():
    """On non-Linux, should recommend mock sensor."""
    import platform
    if platform.system() != "Linux":
        info = get_sensor_info()
        assert info["recommended"] == "mock"
