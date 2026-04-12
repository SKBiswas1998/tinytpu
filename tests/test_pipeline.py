"""Tests for Pipeline — async perception-to-action loop."""

import time
import threading
import pytest
import numpy as np
from collections import deque
from tinytpu.control.pipeline import Pipeline, PipelineConfig, PipelineState


class TestPipelineInit:
    """Pipeline creation and configuration."""

    def test_default_config(self):
        p = Pipeline(model="yolov8n")
        assert p.config.model == "yolov8n"
        assert p.config.mode == "detect"
        assert p.config.target == "person"
        assert p.config.control_hz == 30.0

    def test_custom_config(self):
        p = Pipeline(model="yolov5n", mode="follow", target="cat",
                     camera_id=1, control_hz=60.0, max_linear=0.5)
        assert p.config.model == "yolov5n"
        assert p.config.mode == "follow"
        assert p.config.target == "cat"
        assert p.config.camera_id == 1
        assert p.config.control_hz == 60.0
        assert p.config.max_linear == 0.5

    def test_timing_lists_are_deques(self):
        """Phase 1 fix: thread-safe deques instead of plain lists."""
        p = Pipeline(model="yolov8n")
        assert isinstance(p._capture_times, deque)
        assert isinstance(p._inference_times, deque)
        assert isinstance(p._control_times, deque)
        assert p._capture_times.maxlen == 100
        assert p._inference_times.maxlen == 100
        assert p._control_times.maxlen == 100

    def test_deque_auto_evicts(self):
        """Deque should auto-evict old entries at maxlen."""
        p = Pipeline(model="yolov8n")
        for i in range(200):
            p._capture_times.append(float(i))
        assert len(p._capture_times) == 100
        assert p._capture_times[0] == 100.0  # oldest kept

    def test_initial_state(self):
        p = Pipeline(model="yolov8n")
        assert p._running is False
        assert p._latest_frame is None
        assert p._latest_detections == []
        assert p._latest_tracks == []


class TestPipelineState:
    """State tracking and reporting."""

    def test_state_defaults(self):
        s = PipelineState()
        assert s.running is False
        assert s.fps_capture == 0.0
        assert s.fps_inference == 0.0
        assert s.num_detections == 0
        assert s.target_acquired is False

    def test_get_state_returns_dict(self):
        p = Pipeline(model="yolov8n")
        state = p.get_state()
        assert isinstance(state, dict)
        assert "running" in state
        assert "fps_capture" in state
        assert "fps_inference" in state
        assert "safety_state" in state
        assert "thermal_temp" in state

    def test_get_state_thread_safe(self):
        """get_state should not raise under concurrent access."""
        p = Pipeline(model="yolov8n")
        errors = []

        def reader():
            try:
                for _ in range(100):
                    p.get_state()
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        def writer():
            try:
                for _ in range(100):
                    p._capture_times.append(0.01)
                    p._inference_times.append(0.05)
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=reader), threading.Thread(target=writer)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        assert len(errors) == 0


class TestPipelineFollowCommand:
    """Follow mode command generation."""

    def test_follow_no_target_stops(self):
        p = Pipeline(model="yolov8n", mode="follow", target="person")
        cmd = p._follow_command([], [], None)
        assert cmd["linear_x"] == 0.0
        assert cmd["angular_z"] == 0.0

    def test_follow_centered_target_goes_straight(self):
        p = Pipeline(model="yolov8n", mode="follow", target="person",
                     max_linear=0.3, max_angular=0.5)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        from tests.conftest import FakeDetection
        # Person centered horizontally (cx=320), small (far away)
        det = FakeDetection(0, "person", 0.9, 295, 200, 345, 260)
        cmd = p._follow_command([det], [], frame)
        assert abs(cmd["angular_z"]) < 0.1  # centered → minimal turn
        assert cmd["linear_x"] > 0  # far → move forward

    def test_follow_left_target_turns_right(self):
        p = Pipeline(model="yolov8n", mode="follow", target="person",
                     max_linear=0.3, max_angular=0.5)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        from tests.conftest import FakeDetection
        # Person on LEFT side (cx ≈ 100)
        det = FakeDetection(0, "person", 0.9, 75, 200, 125, 260)
        cmd = p._follow_command([det], [], frame)
        assert cmd["angular_z"] > 0  # negative error → positive angular (turn right/toward)

    def test_follow_close_target_stops_linear(self):
        p = Pipeline(model="yolov8n", mode="follow", target="person",
                     max_linear=0.3, max_angular=0.5)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        from tests.conftest import FakeDetection
        # Person very close (fills >60% of frame height)
        det = FakeDetection(0, "person", 0.9, 270, 50, 370, 430)  # h=380, 380/480=0.79
        cmd = p._follow_command([det], [], frame)
        assert cmd["linear_x"] == 0.0  # too close → stop

    def test_detect_mode_does_not_move(self):
        p = Pipeline(model="yolov8n", mode="detect")
        from tests.conftest import FakeDetection
        det = [FakeDetection(0, "person", 0.9, 100, 100, 200, 300)]
        cmd = p._generate_command(det, [], None)
        assert cmd["linear_x"] == 0.0
        assert cmd["angular_z"] == 0.0

    def test_patrol_mode_wanders(self):
        p = Pipeline(model="yolov8n", mode="patrol")
        p._start_time = time.monotonic() - 5.0  # pretend running for 5s
        cmd = p._generate_command([], [], None)
        assert cmd["linear_x"] == 0.0
        assert cmd["angular_z"] != 0.0  # should be turning


class TestPipelineUpdateState:
    """_update_state correctly computes FPS from deques."""

    def test_fps_from_capture_times(self):
        p = Pipeline(model="yolov8n")
        p._start_time = time.monotonic()
        # Simulate 100 FPS captures (10ms each)
        for _ in range(20):
            p._capture_times.append(0.01)
        p._update_state([], [])
        assert 80 < p._state.fps_capture < 120  # ~100 FPS

    def test_fps_from_inference_times(self):
        p = Pipeline(model="yolov8n")
        p._start_time = time.monotonic()
        # Simulate 10 FPS inference (100ms each)
        for _ in range(20):
            p._inference_times.append(0.1)
        p._update_state([], [])
        assert 8 < p._state.fps_inference < 12

    def test_empty_times_no_crash(self):
        p = Pipeline(model="yolov8n")
        p._start_time = time.monotonic()
        p._update_state([], [])  # should not divide by zero
        assert p._state.fps_capture == 0.0


class TestPipelineContextManager:
    """Pipeline as context manager."""

    def test_stop_without_start(self):
        p = Pipeline(model="yolov8n")
        p.stop()  # should not raise
