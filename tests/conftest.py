from __future__ import annotations

import importlib.util

import pytest

HAS_ROS = importlib.util.find_spec("rclpy") is not None


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "ros: needs a ROS 2 Jazzy environment (rclpy, ros2 bag)")
    config.addinivalue_line("markers", "slow: takes more than ~30 s")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if HAS_ROS:
        return
    skip = pytest.mark.skip(reason="ROS 2 not available")
    for item in items:
        if "ros" in item.keywords:
            item.add_marker(skip)
