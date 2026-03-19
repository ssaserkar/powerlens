"""Tests for Jetson board detection."""

import platform

from powerlens.sensors.auto import detect_jetson_board


def test_detect_jetson_board_returns_dict():
    """Should always return a dict with expected keys."""
    info = detect_jetson_board()
    assert isinstance(info, dict)
    assert "model" in info
    assert "chip" in info
    assert "is_jetson" in info


def test_detect_jetson_board_non_jetson():
    """On non-Jetson hardware, is_jetson should be False."""
    if platform.system() != "Linux":
        info = detect_jetson_board()
        assert info["is_jetson"] is False
