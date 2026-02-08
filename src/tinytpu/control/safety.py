"""
TinyTPU Safety Controller - E-stop, watchdog, velocity ramping, proximity.
Every motor command passes through safety before reaching the robot.
"""

import time
import threading
import logging
from dataclasses import dataclass

logger = logging.getLogger("tinytpu.control.safety")


@dataclass
class SafeCommand:
    """Filtered motor command."""
    linear_x: float = 0.0
    angular_z: float = 0.0
    safe: bool = True
    reason: str = ""


class SafetyController:
    """
    Production safety layer between perception and motors.

    Features:
        - E-stop (immediate zero, requires manual reset)
        - Watchdog timeout (stops if no detection for N seconds)
        - Velocity ramping (max acceleration limit)
        - Proximity stop (zero velocity when object too close)
        - Hard velocity limits
        - Startup delay
    """

    def __init__(self, max_linear=0.3, max_angular=0.5, watchdog_timeout=2.0,
                 ramp_rate=0.5, min_proximity=0.15, startup_delay=1.0):
        self.max_linear = max_linear
        self.max_angular = max_angular
        self.watchdog_timeout = watchdog_timeout
        self.ramp_rate = ramp_rate
        self.min_proximity = min_proximity
        self.startup_delay = startup_delay
        self.state = "startup"
        self._estop = False
        self._estop_reason = ""
        self._last_detection_time = time.monotonic()
        self._start_time = time.monotonic()
        self._last_cmd = SafeCommand()
        self._last_cmd_time = time.monotonic()
        self._lock = threading.Lock()

    def feed_watchdog(self):
        with self._lock:
            self._last_detection_time = time.monotonic()

    def estop(self, reason: str = "manual"):
        with self._lock:
            self._estop = True
            self._estop_reason = reason
            self.state = "estop"
            logger.warning(f"E-STOP: {reason}")

    def reset(self):
        with self._lock:
            self._estop = False
            self._estop_reason = ""
            self.state = "active"
            self._last_detection_time = time.monotonic()
            self._start_time = time.monotonic()

    def filter_command(self, cmd, detections=None, img_w=640, img_h=480) -> SafeCommand:
        with self._lock:
            now = time.monotonic()
            if isinstance(cmd, dict):
                raw_lin = cmd.get("linear_x", 0.0)
                raw_ang = cmd.get("angular_z", 0.0)
            else:
                raw_lin = getattr(cmd, "linear_x", 0.0)
                raw_ang = getattr(cmd, "angular_z", 0.0)

            if self._estop:
                return SafeCommand(0, 0, False, f"estop: {self._estop_reason}")
            if now - self._start_time < self.startup_delay:
                self.state = "startup"
                return SafeCommand(0, 0, True, "startup_delay")
            if now - self._last_detection_time > self.watchdog_timeout:
                self.state = "watchdog"
                return SafeCommand(0, 0, False, "watchdog_timeout")

            if detections and self.min_proximity > 0:
                frame_area = img_w * img_h
                for det in detections:
                    det_w = getattr(det, "x2", 0) - getattr(det, "x1", 0)
                    det_h = getattr(det, "y2", 0) - getattr(det, "y1", 0)
                    det_area = max(0, det_w) * max(0, det_h)
                    if det_area / frame_area > self.min_proximity:
                        return SafeCommand(0, raw_ang, True, "proximity_stop")

            lin = max(-self.max_linear, min(self.max_linear, raw_lin))
            ang = max(-self.max_angular, min(self.max_angular, raw_ang))

            dt = max(0.001, now - self._last_cmd_time)
            max_delta = self.ramp_rate * dt
            brake_delta = max_delta * 2

            prev_lin = self._last_cmd.linear_x
            if lin > prev_lin:
                lin = min(lin, prev_lin + max_delta)
            elif lin < prev_lin:
                lin = max(lin, prev_lin - brake_delta)

            prev_ang = self._last_cmd.angular_z
            if ang > prev_ang:
                ang = min(ang, prev_ang + max_delta)
            elif ang < prev_ang:
                ang = max(ang, prev_ang - brake_delta)

            self.state = "active"
            result = SafeCommand(lin, ang, True, "ok")
            self._last_cmd = result
            self._last_cmd_time = now
            return result

    def get_status(self) -> dict:
        with self._lock:
            return {
                "state": self.state, "estop": self._estop,
                "estop_reason": self._estop_reason,
                "watchdog_age": time.monotonic() - self._last_detection_time,
            }
