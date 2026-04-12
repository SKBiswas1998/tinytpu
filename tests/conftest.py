"""Shared fixtures for TinyTPU test suite."""

import pytest
import numpy as np
from dataclasses import dataclass


@dataclass
class FakeDetection:
    """Lightweight detection for testing (no model needed)."""
    class_id: int = 0
    class_name: str = "person"
    confidence: float = 0.9
    x1: float = 100.0
    y1: float = 100.0
    x2: float = 200.0
    y2: float = 300.0

    @property
    def center(self):
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def area(self):
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)


@pytest.fixture
def sample_frame():
    """640x480 RGB uint8 frame."""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def small_frame():
    """100x100 RGB uint8 frame."""
    return np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)


@pytest.fixture
def sample_detections():
    """List of 3 fake detections at different positions."""
    return [
        FakeDetection(0, "person", 0.95, 100, 50, 200, 350),
        FakeDetection(2, "car", 0.80, 400, 200, 600, 400),
        FakeDetection(15, "cat", 0.60, 50, 300, 150, 400),
    ]


@pytest.fixture
def single_person_detection():
    """One person detection centered in frame."""
    return [FakeDetection(0, "person", 0.9, 270, 100, 370, 380)]


@pytest.fixture
def large_detection():
    """Detection covering >15% of 640x480 frame (proximity trigger)."""
    return [FakeDetection(0, "person", 0.95, 0, 0, 500, 400)]


@pytest.fixture
def safety_controller():
    """SafetyController with no startup delay for testing."""
    from tinytpu.control.safety import SafetyController
    import time
    sc = SafetyController(
        max_linear=0.3, max_angular=0.5,
        watchdog_timeout=2.0, ramp_rate=0.5,
        min_proximity=0.15, startup_delay=0.0,
    )
    sc.feed_watchdog()
    time.sleep(0.01)  # ensure past startup
    return sc


@pytest.fixture
def tracker():
    """ObjectTracker with default settings."""
    from tinytpu.perception.tracker import ObjectTracker
    return ObjectTracker(iou_threshold=0.3, max_age=15, min_hits=2)


@pytest.fixture
def kalman_filter():
    """KalmanFilter2D initialized at (100, 100, 50, 50)."""
    from tinytpu.perception.tracker import KalmanFilter2D
    return KalmanFilter2D(initial_bbox=(100, 100, 50, 50))
