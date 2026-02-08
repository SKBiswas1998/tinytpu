"""
TinyTPU HAL Backend Tests.
"""

import numpy as np
import pytest


class TestBackendInterface:
    """Test backend registry and auto-selection."""

    def test_all_backends_instantiate(self):
        from tinytpu.hal.backends import ALL_BACKENDS
        for cls in ALL_BACKENDS:
            b = cls()
            assert hasattr(b, "name")
            assert hasattr(b, "priority")
            assert hasattr(b, "available")
            assert hasattr(b, "load")
            assert hasattr(b, "run")

    def test_get_available_backends(self):
        from tinytpu.hal.backends import get_available_backends
        backends = get_available_backends()
        assert len(backends) >= 1  # At least NumPy
        names = [b.name for b in backends]
        assert "numpy" in names

    def test_auto_backend_returns_something(self):
        from tinytpu.hal.backends import auto_backend
        backend = auto_backend()
        assert backend is not None
        assert backend.available()

    def test_auto_backend_prefers_onnxruntime_over_numpy(self):
        from tinytpu.hal.backends import auto_backend
        backend = auto_backend()
        # If ORT is installed, it should be preferred
        try:
            import onnxruntime
            assert backend.name == "onnxruntime"
        except ImportError:
            assert backend.name == "numpy"

    def test_auto_backend_with_preference(self):
        from tinytpu.hal.backends import auto_backend
        backend = auto_backend(prefer="numpy")
        assert backend.name == "numpy"

    def test_backend_priority_order(self):
        from tinytpu.hal.backends import get_backends
        backends = get_backends()
        # Hailo > Coral > ORT > NumPy
        priorities = {b.name: b.priority for b in backends}
        assert priorities["hailo"] > priorities["coral"]
        assert priorities["coral"] > priorities["onnxruntime"]
        assert priorities["onnxruntime"] > priorities["numpy"]

    def test_list_backends_string(self):
        from tinytpu.hal.backends import list_backends
        output = list_backends()
        assert "onnxruntime" in output or "numpy" in output

    def test_numpy_always_available(self):
        from tinytpu.hal.backends import NumPyBackend
        b = NumPyBackend()
        assert b.available() is True

    def test_onnxrt_supports_onnx_format(self):
        from tinytpu.hal.backends import ONNXRuntimeBackend
        b = ONNXRuntimeBackend()
        assert b.supports("model.onnx") is True
        assert b.supports("model.hef") is False

    def test_hailo_supports_hef_format(self):
        from tinytpu.hal.backends import HailoBackend
        b = HailoBackend()
        assert b.supports("model.hef") is True
        assert b.supports("model.onnx") is False

    def test_coral_supports_tflite_format(self):
        from tinytpu.hal.backends import CoralBackend
        b = CoralBackend()
        assert b.supports("model.tflite") is True
        assert b.supports("model.onnx") is False


class TestONNXRuntimeBackend:
    """Test ONNX Runtime backend when available."""

    @pytest.fixture
    def backend(self):
        from tinytpu.hal.backends import ONNXRuntimeBackend
        b = ONNXRuntimeBackend()
        if not b.available():
            pytest.skip("ONNX Runtime not installed")
        return b

    def test_get_providers(self, backend):
        providers = backend._get_providers()
        assert len(providers) >= 1
        assert "CPUExecutionProvider" in providers


class TestPipelineUnit:
    """Unit tests for Pipeline (no camera needed)."""

    def test_pipeline_creates(self):
        from tinytpu.control.pipeline import Pipeline
        p = Pipeline(model="yolov8n", mode="detect")
        assert p.config.model == "yolov8n"
        assert p.config.mode == "detect"

    def test_pipeline_config_defaults(self):
        from tinytpu.control.pipeline import PipelineConfig
        cfg = PipelineConfig()
        assert cfg.model == "yolov8n"
        assert cfg.control_hz == 30.0
        assert cfg.enable_safety is True

    def test_pipeline_state_defaults(self):
        from tinytpu.control.pipeline import PipelineState
        state = PipelineState()
        assert state.running is False
        assert state.fps_inference == 0.0

    def test_pipeline_follow_command(self):
        """Test follow command generation."""
        from tinytpu.control.pipeline import Pipeline
        from tinytpu.inference.model_zoo import Detection

        p = Pipeline(model="yolov8n", mode="follow", target="person")
        p._state.running = True
        p._start_time = 0

        # Simulate a detection at center-right, small enough to move toward
        det = Detection(0, "person", 0.9, 400, 200, 460, 340)  # 60x140 in 640x480
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cmd = p._follow_command([det], [], frame)

        # Should turn left (negative angular since target is right of center)
        assert cmd["angular_z"] < 0
        assert cmd["linear_x"] > 0  # Should move forward (target is small/far)

    def test_pipeline_follow_no_target(self):
        from tinytpu.control.pipeline import Pipeline

        p = Pipeline(mode="follow", target="person")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cmd = p._follow_command([], [], frame)
        assert cmd["linear_x"] == 0.0
        assert cmd["angular_z"] == 0.0

    def test_pipeline_get_state(self):
        from tinytpu.control.pipeline import Pipeline
        p = Pipeline()
        state = p.get_state()
        assert isinstance(state, dict)
        assert "running" in state
        assert "fps_inference" in state
        assert "safety_state" in state


class TestSafetyIntegration:
    """Test Safety + Pipeline integration."""

    def test_safety_estop(self):
        from tinytpu.control.safety import SafetyController
        sc = SafetyController()
        sc.estop("test")
        cmd = sc.filter_command({"linear_x": 1.0, "angular_z": 0.5})
        assert cmd.linear_x == 0.0
        assert cmd.angular_z == 0.0
        assert cmd.safe is False

    def test_safety_reset(self):
        from tinytpu.control.safety import SafetyController
        sc = SafetyController(startup_delay=0.0)
        sc.estop("test")
        sc.reset()
        sc.feed_watchdog()
        import time; time.sleep(0.01)
        cmd = sc.filter_command({"linear_x": 0.2, "angular_z": 0.1})
        assert cmd.safe is True

    def test_safety_velocity_limits(self):
        from tinytpu.control.safety import SafetyController
        sc = SafetyController(max_linear=0.3, max_angular=0.5, startup_delay=0.0)
        sc.feed_watchdog()
        import time; time.sleep(0.01)
        cmd = sc.filter_command({"linear_x": 10.0, "angular_z": 10.0})
        # Should be clamped (might also be ramped)
        assert abs(cmd.linear_x) <= 0.3 + 0.01
        assert abs(cmd.angular_z) <= 0.5 + 0.01

    def test_safety_watchdog_timeout(self):
        from tinytpu.control.safety import SafetyController
        sc = SafetyController(watchdog_timeout=0.01, startup_delay=0.0)
        import time; time.sleep(0.02)
        cmd = sc.filter_command({"linear_x": 0.5})
        assert cmd.linear_x == 0.0
        assert "watchdog" in cmd.reason


class TestTrackerIntegration:
    """Test tracker with pipeline."""

    def test_kalman_predict_ahead(self):
        from tinytpu.perception.tracker import KalmanFilter2D
        kf = KalmanFilter2D(initial_bbox=(100, 100, 50, 80))

        # Update with moving object (rightward motion)
        for x in range(110, 160, 10):
            kf.predict()
            kf.update(np.array([x, 100, 50, 80]))

        # Predict should extrapolate rightward
        pred = kf.predict()
        assert pred[0] > 150  # x should be beyond last measurement

    def test_tracker_assigns_ids(self):
        from tinytpu.perception.tracker import ObjectTracker
        from tinytpu.inference.model_zoo import Detection

        tracker = ObjectTracker()

        # First update creates tentative tracks
        dets1 = [Detection(0, "person", 0.9, 100, 100, 200, 300)]
        tracks1 = tracker.update(dets1)

        # Second update at similar position confirms the track (min_hits=2)
        dets2 = [Detection(0, "person", 0.9, 105, 100, 205, 300)]
        tracks2 = tracker.update(dets2)
        assert len(tracks2) >= 1

        # Third update should keep same track
        dets3 = [Detection(0, "person", 0.9, 110, 100, 210, 300)]
        tracks3 = tracker.update(dets3)
        assert len(tracks3) >= 1
        assert tracks3[0].track_id == tracks2[0].track_id
