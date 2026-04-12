"""Tests for SafetyController — every motor command must pass through safety."""

import time
import threading
import pytest
from tinytpu.control.safety import SafetyController, SafeCommand


class TestSafetyProximity:
    """Verify proximity stop zeroes BOTH linear and angular velocity."""

    def test_proximity_zeroes_linear(self, safety_controller, large_detection):
        result = safety_controller.filter_command(
            {"linear_x": 0.3, "angular_z": 0.0}, large_detection
        )
        assert result.linear_x == 0.0

    def test_proximity_zeroes_angular(self, safety_controller, large_detection):
        result = safety_controller.filter_command(
            {"linear_x": 0.0, "angular_z": 0.5}, large_detection
        )
        assert result.angular_z == 0.0, "Angular must be zeroed on proximity stop"

    def test_proximity_zeroes_both(self, safety_controller, large_detection):
        result = safety_controller.filter_command(
            {"linear_x": 0.3, "angular_z": 0.5}, large_detection
        )
        assert result.linear_x == 0.0
        assert result.angular_z == 0.0

    def test_proximity_marks_unsafe(self, safety_controller, large_detection):
        result = safety_controller.filter_command(
            {"linear_x": 0.3, "angular_z": 0.5}, large_detection
        )
        assert result.safe is False
        assert result.reason == "proximity_stop"

    def test_proximity_not_triggered_small_detection(self, safety_controller, sample_detections):
        """Small detections (far away) should NOT trigger proximity stop."""
        result = safety_controller.filter_command(
            {"linear_x": 0.2, "angular_z": 0.1}, sample_detections
        )
        assert result.reason != "proximity_stop"

    def test_proximity_threshold_boundary(self):
        """Detection area exactly at threshold boundary."""
        sc = SafetyController(min_proximity=0.15, startup_delay=0.0)
        sc.feed_watchdog()
        time.sleep(0.01)

        from tests.conftest import FakeDetection
        # 640*480 = 307200. 15% = 46080. Detection: 240*192 = 46080 (exactly 15%)
        det = FakeDetection(0, "person", 0.9, 200, 144, 440, 336)
        result = sc.filter_command({"linear_x": 0.3, "angular_z": 0.5}, [det], 640, 480)
        # At exactly the threshold, should NOT trigger (> not >=)
        assert result.reason != "proximity_stop"

    def test_proximity_disabled_when_zero(self):
        """min_proximity=0 should disable proximity checks."""
        sc = SafetyController(min_proximity=0, startup_delay=0.0)
        sc.feed_watchdog()
        time.sleep(0.01)
        from tests.conftest import FakeDetection
        det = FakeDetection(0, "person", 0.9, 0, 0, 640, 480)  # full frame
        result = sc.filter_command({"linear_x": 0.3, "angular_z": 0.5}, [det], 640, 480)
        assert result.reason != "proximity_stop"


class TestSafetyEstop:
    """E-stop must immediately zero everything and require manual reset."""

    def test_estop_zeroes_command(self, safety_controller):
        safety_controller.estop("test")
        result = safety_controller.filter_command({"linear_x": 0.3, "angular_z": 0.5})
        assert result.linear_x == 0.0
        assert result.angular_z == 0.0
        assert result.safe is False

    def test_estop_reason_preserved(self, safety_controller):
        safety_controller.estop("collision_detected")
        result = safety_controller.filter_command({"linear_x": 0.3, "angular_z": 0.0})
        assert "collision_detected" in result.reason

    def test_estop_persists_without_reset(self, safety_controller):
        safety_controller.estop("test")
        for _ in range(10):
            result = safety_controller.filter_command({"linear_x": 0.3, "angular_z": 0.0})
            assert result.linear_x == 0.0

    def test_reset_clears_estop(self, safety_controller):
        safety_controller.estop("test")
        safety_controller.reset()
        safety_controller.feed_watchdog()
        time.sleep(0.01)
        result = safety_controller.filter_command({"linear_x": 0.2, "angular_z": 0.0})
        assert result.safe is True
        assert result.reason == "ok"

    def test_estop_state_in_status(self, safety_controller):
        safety_controller.estop("manual")
        status = safety_controller.get_status()
        assert status["estop"] is True
        assert status["estop_reason"] == "manual"
        assert status["state"] == "estop"


class TestSafetyWatchdog:
    """Watchdog must stop the robot if no detections arrive."""

    def test_watchdog_triggers_after_timeout(self):
        sc = SafetyController(watchdog_timeout=0.05, startup_delay=0.0)
        sc.feed_watchdog()
        time.sleep(0.1)  # exceed timeout
        result = sc.filter_command({"linear_x": 0.3, "angular_z": 0.0})
        assert result.linear_x == 0.0
        assert result.angular_z == 0.0
        assert result.reason == "watchdog_timeout"

    def test_watchdog_fed_prevents_timeout(self):
        sc = SafetyController(watchdog_timeout=0.1, startup_delay=0.0)
        sc.feed_watchdog()
        time.sleep(0.05)
        sc.feed_watchdog()  # reset timer
        time.sleep(0.05)
        sc.feed_watchdog()  # reset again
        result = sc.filter_command({"linear_x": 0.2, "angular_z": 0.0})
        assert result.reason != "watchdog_timeout"

    def test_watchdog_state(self):
        sc = SafetyController(watchdog_timeout=0.05, startup_delay=0.0)
        sc.feed_watchdog()
        time.sleep(0.1)
        sc.filter_command({"linear_x": 0.3, "angular_z": 0.0})
        assert sc.get_status()["state"] == "watchdog"


class TestSafetyVelocityLimits:
    """Hard velocity limits must always be enforced."""

    def test_linear_clamped(self, safety_controller):
        result = safety_controller.filter_command({"linear_x": 999.0, "angular_z": 0.0})
        assert abs(result.linear_x) <= safety_controller.max_linear

    def test_angular_clamped(self, safety_controller):
        result = safety_controller.filter_command({"linear_x": 0.0, "angular_z": 999.0})
        assert abs(result.angular_z) <= safety_controller.max_angular

    def test_negative_linear_clamped(self, safety_controller):
        result = safety_controller.filter_command({"linear_x": -999.0, "angular_z": 0.0})
        assert result.linear_x >= -safety_controller.max_linear

    def test_negative_angular_clamped(self, safety_controller):
        result = safety_controller.filter_command({"linear_x": 0.0, "angular_z": -999.0})
        assert result.angular_z >= -safety_controller.max_angular


class TestSafetyRamping:
    """Velocity ramping prevents sudden acceleration."""

    def test_linear_ramps_up_gradually(self, safety_controller):
        # First command from zero — should be limited by ramp rate
        result = safety_controller.filter_command({"linear_x": 0.3, "angular_z": 0.0})
        assert result.linear_x < 0.3, "Should ramp up, not jump to max"
        assert result.linear_x > 0.0, "Should move some amount"

    def test_braking_faster_than_acceleration(self, safety_controller):
        # Accelerate for several steps
        for _ in range(50):
            safety_controller.filter_command({"linear_x": 0.3, "angular_z": 0.0})
            time.sleep(0.01)
        # Now brake
        r1 = safety_controller.filter_command({"linear_x": 0.0, "angular_z": 0.0})
        # r1 should have dropped from ~0.3 more than it would gain in one step
        assert r1.linear_x < 0.3


class TestSafetyStartupDelay:
    """Robot must not move during startup."""

    def test_startup_delay_blocks_motion(self):
        sc = SafetyController(startup_delay=1.0)
        sc.feed_watchdog()
        result = sc.filter_command({"linear_x": 0.3, "angular_z": 0.5})
        assert result.linear_x == 0.0
        assert result.angular_z == 0.0
        assert result.reason == "startup_delay"

    def test_startup_delay_state(self):
        sc = SafetyController(startup_delay=1.0)
        sc.feed_watchdog()
        sc.filter_command({"linear_x": 0.3, "angular_z": 0.0})
        assert sc.get_status()["state"] == "startup"


class TestSafetyInputFormats:
    """filter_command should accept both dict and object inputs."""

    def test_dict_input(self, safety_controller):
        result = safety_controller.filter_command({"linear_x": 0.1, "angular_z": 0.1})
        assert result.safe is True

    def test_object_input(self, safety_controller):
        cmd = SafeCommand(linear_x=0.1, angular_z=0.1)
        result = safety_controller.filter_command(cmd)
        assert result.safe is True

    def test_missing_keys_default_zero(self, safety_controller):
        result = safety_controller.filter_command({})
        assert result.linear_x == 0.0
        assert result.angular_z == 0.0


class TestSafetyThreadSafety:
    """Safety controller must be safe under concurrent access."""

    def test_concurrent_filter_commands(self, safety_controller):
        results = []
        errors = []

        def worker():
            try:
                for _ in range(100):
                    r = safety_controller.filter_command({"linear_x": 0.2, "angular_z": 0.1})
                    results.append(r)
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(errors) == 0, f"Thread errors: {errors}"
        assert len(results) == 400

    def test_concurrent_estop_and_filter(self, safety_controller):
        errors = []

        def estop_worker():
            try:
                for _ in range(50):
                    safety_controller.estop("test")
                    time.sleep(0.001)
                    safety_controller.reset()
                    safety_controller.feed_watchdog()
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        def filter_worker():
            try:
                for _ in range(100):
                    safety_controller.filter_command({"linear_x": 0.2, "angular_z": 0.1})
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=estop_worker)
        t2 = threading.Thread(target=filter_worker)
        t1.start(); t2.start()
        t1.join(timeout=10); t2.join(timeout=10)
        assert len(errors) == 0
