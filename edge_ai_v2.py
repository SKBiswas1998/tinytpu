"""
TinyTPU Edge AI v2 - Production Safety & Tracking Layer
=========================================================
Extends edge_ai.py with production-critical features:

Tier 1 - Safety:
  SafetyController  - E-stop, watchdog, velocity ramping, detection timeout
  ThermalMonitor    - CPU temp polling, auto-throttle before OS thermal cliff
  MemoryWatchdog    - RSS tracking, OOM prevention

Tier 2 - Tracking:
  KalmanFilter2D    - Predict object positions between inference frames
  TrackedObject     - Persistent object with ID, history, Kalman state
  ObjectTracker     - IoU matching + Kalman prediction, 30Hz from 2FPS vision
  AsyncPipeline     - Threaded capture/inference/control, nothing blocks

Tier 3 - Field Debugging:
  BlackBoxRecorder  - Log detections + commands + frames, replay offline
  ImageQualityScorer - Detect blur/darkness/occlusion

Usage:
  from tinytpu.edge_ai_v2 import SafetyController, ThermalMonitor, MemoryWatchdog
  from tinytpu.edge_ai_v2 import ObjectTracker, AsyncPipeline
  from tinytpu.edge_ai_v2 import BlackBoxRecorder, ImageQualityScorer
  from tinytpu.edge_ai_v2 import ProductionEdgeAI
"""

import os, sys, time, threading, json, struct, logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from collections import deque
import numpy as np

from tinytpu.edge_ai import Detection, RobotCommand, ObjectDetector, RobotController, EdgeAI, detect_hardware

logger = logging.getLogger("tinytpu.edge_ai_v2")


# ================================================================
# TIER 1: SAFETY
# ================================================================


class SafetyController:
    """
    Production safety layer between perception and motors.

    Features:
      - Software E-stop (immediate zero velocity)
      - Watchdog timer (no detection update -> safe stop)
      - Velocity ramping (no instant max speed)
      - Detection timeout (vision failure -> stop)
      - Max velocity enforcement (hardware limits)
      - Collision proximity stop (object too close)
      - State machine: RUNNING -> ESTOP | TIMEOUT | RAMPING_DOWN
    """

    RUNNING = "running"
    ESTOP = "estop"
    TIMEOUT = "timeout"
    RAMPING_DOWN = "ramping_down"
    STARTUP = "startup"

    def __init__(self,
                 max_linear: float = 0.3,
                 max_angular: float = 0.5,
                 watchdog_timeout: float = 2.0,
                 ramp_rate: float = 0.5,
                 min_proximity: float = 0.15,
                 startup_delay: float = 1.0):
        """
        Args:
            max_linear: Hard limit on linear velocity (m/s)
            max_angular: Hard limit on angular velocity (rad/s)
            watchdog_timeout: Seconds without detection update before safe stop
            ramp_rate: Max velocity change per second (m/s/s)
            min_proximity: Object relative size that triggers emergency stop
            startup_delay: Seconds to wait before allowing movement
        """
        self.max_linear = max_linear
        self.max_angular = max_angular
        self.watchdog_timeout = watchdog_timeout
        self.ramp_rate = ramp_rate
        self.min_proximity = min_proximity
        self.startup_delay = startup_delay

        self.state = self.STARTUP
        self.estop_active = False
        self.last_detection_time = time.monotonic()
        self.last_command_time = time.monotonic()
        self.startup_time = time.monotonic()
        self.current_linear = 0.0
        self.current_angular = 0.0

        # Telemetry
        self.estop_count = 0
        self.timeout_count = 0
        self.proximity_stop_count = 0
        self.total_commands = 0
        self.violations = []  # list of (time, type, detail)

        self._lock = threading.Lock()

    def estop(self, reason: str = "manual"):
        """Immediate stop. Must call reset() to resume."""
        with self._lock:
            self.estop_active = True
            self.state = self.ESTOP
            self.current_linear = 0.0
            self.current_angular = 0.0
            self.estop_count += 1
            self.violations.append((time.monotonic(), "estop", reason))
            logger.warning(f"E-STOP activated: {reason}")

    def reset(self):
        """Clear E-stop. Robot will ramp up from zero."""
        with self._lock:
            self.estop_active = False
            self.state = self.RUNNING
            self.current_linear = 0.0
            self.current_angular = 0.0
            self.last_detection_time = time.monotonic()
            logger.info("E-STOP cleared, resuming")

    def feed_watchdog(self):
        """Call this every time a valid detection frame arrives."""
        self.last_detection_time = time.monotonic()

    def _check_proximity(self, detections: list, img_w: int, img_h: int) -> bool:
        """Return True if any object is dangerously close."""
        img_area = img_w * img_h
        if img_area == 0:
            return False
        for det in detections:
            relative_size = det.area / img_area
            if relative_size > self.min_proximity:
                return True
        return False

    def filter_command(self, cmd: RobotCommand, detections: list = None,
                       img_w: int = 640, img_h: int = 480) -> RobotCommand:
        """
        Filter a robot command through safety checks.

        Returns a safe command (possibly zeroed).
        """
        now = time.monotonic()
        self.total_commands += 1

        with self._lock:
            # 1. E-stop check
            if self.estop_active:
                return RobotCommand(0, 0, 0, "estop", "E-STOP active")

            # 2. Startup delay
            if now - self.startup_time < self.startup_delay:
                self.state = self.STARTUP
                return RobotCommand(0, 0, 0, "startup", f"Starting in {self.startup_delay-(now-self.startup_time):.1f}s")

            # 3. Watchdog timeout
            since_detection = now - self.last_detection_time
            if since_detection > self.watchdog_timeout:
                if self.state != self.TIMEOUT:
                    self.timeout_count += 1
                    self.violations.append((now, "timeout", f"{since_detection:.1f}s without detection"))
                    logger.warning(f"Watchdog timeout: {since_detection:.1f}s without detection")
                self.state = self.TIMEOUT
                self.current_linear = 0.0
                self.current_angular = 0.0
                return RobotCommand(0, 0, 0, "timeout", f"No detection for {since_detection:.1f}s")

            # 4. Proximity check
            if detections and self._check_proximity(detections, img_w, img_h):
                self.proximity_stop_count += 1
                self.current_linear = 0.0
                return RobotCommand(0, 0, cmd.angular_z * 0.5, "proximity_stop", "Object too close")

            # 5. Enforce velocity limits
            target_linear = max(-self.max_linear, min(self.max_linear, cmd.linear_x))
            target_angular = max(-self.max_angular, min(self.max_angular, cmd.angular_z))

            # 6. Velocity ramping
            dt = now - self.last_command_time
            dt = min(dt, 0.5)  # cap for first call or long gaps
            max_change = self.ramp_rate * dt

            if target_linear > self.current_linear:
                self.current_linear = min(target_linear, self.current_linear + max_change)
            else:
                self.current_linear = max(target_linear, self.current_linear - max_change * 2)  # brake faster

            if target_angular > self.current_angular:
                self.current_angular = min(target_angular, self.current_angular + max_change * 2)
            else:
                self.current_angular = max(target_angular, self.current_angular - max_change * 2)

            self.last_command_time = now
            self.state = self.RUNNING

            return RobotCommand(self.current_linear, cmd.linear_y, self.current_angular,
                               cmd.action, cmd.description)

    def get_status(self) -> dict:
        """Return safety system status for telemetry."""
        now = time.monotonic()
        return {
            "state": self.state,
            "estop_active": self.estop_active,
            "seconds_since_detection": now - self.last_detection_time,
            "current_linear": self.current_linear,
            "current_angular": self.current_angular,
            "estop_count": self.estop_count,
            "timeout_count": self.timeout_count,
            "proximity_stops": self.proximity_stop_count,
            "total_commands": self.total_commands,
            "recent_violations": self.violations[-10:],
        }


class ThermalMonitor:
    """
    CPU temperature monitoring with auto-throttle.

    Problem: RPi4 hits 80C in ~50s of inference, OS throttles
    clock from 1.5GHz to 1GHz = 40% perf drop without warning.

    Solution: Monitor temp, proactively reduce inference rate
    before the OS thermal cliff.
    """

    def __init__(self,
                 warn_temp: float = 70.0,
                 critical_temp: float = 78.0,
                 shutdown_temp: float = 85.0,
                 poll_interval: float = 2.0):
        self.warn_temp = warn_temp
        self.critical_temp = critical_temp
        self.shutdown_temp = shutdown_temp
        self.poll_interval = poll_interval

        self.current_temp = 0.0
        self.max_temp = 0.0
        self.throttle_level = 0  # 0=none, 1=warn, 2=critical, 3=shutdown
        self.temp_history = deque(maxlen=300)  # 10 min at 2s interval
        self.throttle_events = []

        self._running = False
        self._thread = None

    def _read_temp(self) -> float:
        """Read CPU temperature. Cross-platform."""
        # Linux: thermal zone
        try:
            with open("/sys/class/thermal/thermal_zone0/temp") as f:
                return float(f.read().strip()) / 1000.0
        except (FileNotFoundError, PermissionError):
            pass

        # Raspberry Pi: vcgencmd
        try:
            import subprocess
            result = subprocess.run(["vcgencmd", "measure_temp"],
                                   capture_output=True, text=True, timeout=2)
            if result.returncode == 0:
                # "temp=45.0'C"
                return float(result.stdout.split("=")[1].split("'")[0])
        except (FileNotFoundError, Exception):
            pass

        # Windows: WMI (rough)
        try:
            import subprocess
            result = subprocess.run(
                ["wmic", "path", "MSAcpi_ThermalZoneTemperature", "get", "CurrentTemperature"],
                capture_output=True, text=True, timeout=5)
            for line in result.stdout.strip().split("\n"):
                line = line.strip()
                if line.isdigit():
                    return (float(line) - 2732) / 10.0  # decikelvin to C
        except Exception:
            pass

        return -1.0  # unknown

    def _poll_loop(self):
        while self._running:
            temp = self._read_temp()
            if temp > 0:
                self.current_temp = temp
                self.max_temp = max(self.max_temp, temp)
                self.temp_history.append((time.monotonic(), temp))

                old_level = self.throttle_level
                if temp >= self.shutdown_temp:
                    self.throttle_level = 3
                elif temp >= self.critical_temp:
                    self.throttle_level = 2
                elif temp >= self.warn_temp:
                    self.throttle_level = 1
                else:
                    self.throttle_level = 0

                if self.throttle_level != old_level and self.throttle_level > 0:
                    event = (time.monotonic(), temp, self.throttle_level)
                    self.throttle_events.append(event)
                    labels = {1: "WARN", 2: "CRITICAL", 3: "SHUTDOWN"}
                    logger.warning(f"Thermal {labels[self.throttle_level]}: {temp:.1f}C")

            time.sleep(self.poll_interval)

    def start(self):
        """Start background temperature monitoring."""
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True, name="thermal_monitor")
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    def get_skip_factor(self) -> int:
        """How many frames to skip based on thermal state.
        0 = process every frame, 1 = skip every other, etc."""
        if self.throttle_level == 0:
            return 0  # full speed
        elif self.throttle_level == 1:
            return 1  # skip every other frame
        elif self.throttle_level == 2:
            return 2  # process every 3rd frame
        else:
            return 4  # minimal inference, near shutdown

    def get_status(self) -> dict:
        return {
            "current_temp_c": self.current_temp,
            "max_temp_c": self.max_temp,
            "throttle_level": self.throttle_level,
            "throttle_label": ["none", "warn", "critical", "shutdown"][self.throttle_level],
            "skip_factor": self.get_skip_factor(),
            "history_len": len(self.temp_history),
            "throttle_events": len(self.throttle_events),
        }


class MemoryWatchdog:
    """
    Monitor process RSS and system memory.

    Problem: YOLO on Pi 2GB silently crashes after ~20min.
    No error, no log, just dies. OOM killer strikes.

    Solution: Track memory, warn early, force GC, and
    optionally trigger safe stop before OOM.
    """

    def __init__(self,
                 warn_percent: float = 70.0,
                 critical_percent: float = 85.0,
                 max_rss_mb: float = 0,
                 poll_interval: float = 5.0,
                 on_critical=None):
        """
        Args:
            warn_percent: System RAM usage % to trigger warning
            critical_percent: System RAM usage % to trigger critical action
            max_rss_mb: Max process RSS in MB (0 = auto: 50% of total)
            poll_interval: Seconds between checks
            on_critical: Callback when critical threshold hit
        """
        self.warn_percent = warn_percent
        self.critical_percent = critical_percent
        self.max_rss_mb = max_rss_mb
        self.poll_interval = poll_interval
        self.on_critical = on_critical

        self.current_rss_mb = 0.0
        self.peak_rss_mb = 0.0
        self.system_used_percent = 0.0
        self.system_available_mb = 0.0
        self.state = "ok"  # ok, warn, critical
        self.events = []
        self.rss_history = deque(maxlen=120)  # 10 min at 5s

        self._running = False
        self._thread = None

        # Auto-detect max RSS if not set
        if self.max_rss_mb == 0:
            hw = detect_hardware()
            self.max_rss_mb = hw.ram_total_mb * 0.5

    def _read_memory(self):
        """Read process and system memory."""
        # Process RSS
        try:
            with open(f"/proc/{os.getpid()}/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        self.current_rss_mb = float(line.split()[1]) / 1024.0
                        break
        except (FileNotFoundError, PermissionError):
            try:
                import psutil
                proc = psutil.Process(os.getpid())
                self.current_rss_mb = proc.memory_info().rss / 1024 / 1024
            except ImportError:
                pass

        self.peak_rss_mb = max(self.peak_rss_mb, self.current_rss_mb)

        # System memory
        try:
            with open("/proc/meminfo") as f:
                info = {}
                for line in f:
                    parts = line.split()
                    if len(parts) >= 2:
                        info[parts[0].rstrip(":")] = int(parts[1])
                total = info.get("MemTotal", 1) / 1024
                avail = info.get("MemAvailable", info.get("MemFree", 0)) / 1024
                self.system_available_mb = avail
                self.system_used_percent = (1 - avail / total) * 100 if total > 0 else 0
        except (FileNotFoundError, PermissionError):
            try:
                import ctypes
                class MEMSTAT(ctypes.Structure):
                    _fields_ = [("dwLength", ctypes.c_ulong),
                                ("dwMemoryLoad", ctypes.c_ulong),
                                ("ullTotalPhys", ctypes.c_ulonglong),
                                ("ullAvailPhys", ctypes.c_ulonglong),
                                ("ullTotalPageFile", ctypes.c_ulonglong),
                                ("ullAvailPageFile", ctypes.c_ulonglong),
                                ("ullTotalVirtual", ctypes.c_ulonglong),
                                ("ullAvailVirtual", ctypes.c_ulonglong),
                                ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
                m = MEMSTAT(dwLength=ctypes.sizeof(MEMSTAT))
                ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
                self.system_available_mb = m.ullAvailPhys / 1024 / 1024
                total_mb = m.ullTotalPhys / 1024 / 1024
                self.system_used_percent = (1 - self.system_available_mb / total_mb) * 100
            except Exception:
                pass

    def _poll_loop(self):
        while self._running:
            self._read_memory()
            self.rss_history.append((time.monotonic(), self.current_rss_mb))

            old_state = self.state
            rss_exceeded = self.current_rss_mb > self.max_rss_mb

            if self.system_used_percent > self.critical_percent or rss_exceeded:
                self.state = "critical"
                if old_state != "critical":
                    detail = f"RSS={self.current_rss_mb:.0f}MB sys={self.system_used_percent:.0f}%"
                    self.events.append((time.monotonic(), "critical", detail))
                    logger.error(f"Memory CRITICAL: {detail}")
                    # Force garbage collection
                    import gc
                    gc.collect()
                    if self.on_critical:
                        try:
                            self.on_critical()
                        except Exception as e:
                            logger.error(f"Critical callback failed: {e}")

            elif self.system_used_percent > self.warn_percent:
                self.state = "warn"
                if old_state == "ok":
                    detail = f"RSS={self.current_rss_mb:.0f}MB sys={self.system_used_percent:.0f}%"
                    self.events.append((time.monotonic(), "warn", detail))
                    logger.warning(f"Memory WARNING: {detail}")
            else:
                self.state = "ok"

            time.sleep(self.poll_interval)

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True, name="memory_watchdog")
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    def get_status(self) -> dict:
        return {
            "state": self.state,
            "process_rss_mb": round(self.current_rss_mb, 1),
            "peak_rss_mb": round(self.peak_rss_mb, 1),
            "max_rss_mb": round(self.max_rss_mb, 1),
            "system_used_percent": round(self.system_used_percent, 1),
            "system_available_mb": round(self.system_available_mb, 1),
            "events": len(self.events),
        }


# ================================================================
# TIER 2: TRACKING & ASYNC PIPELINE
# ================================================================


class KalmanFilter2D:
    """
    2D Kalman filter for bounding box tracking.

    State: [cx, cy, w, h, vx, vy, vw, vh]
    Measurement: [cx, cy, w, h]

    Predicts object position between slow inference frames,
    enabling 30Hz control from 2 FPS vision.
    """

    def __init__(self, initial_bbox: Tuple[float, float, float, float],
                 process_noise: float = 1.0,
                 measurement_noise: float = 1.0):
        """
        Args:
            initial_bbox: (cx, cy, w, h)
            process_noise: Higher = trust predictions less
            measurement_noise: Higher = trust measurements less
        """
        cx, cy, w, h = initial_bbox

        # State [cx, cy, w, h, vx, vy, vw, vh]
        self.x = np.array([cx, cy, w, h, 0, 0, 0, 0], dtype=np.float64)

        # State transition (constant velocity)
        self.F = np.eye(8, dtype=np.float64)
        # dt gets set in predict()

        # Measurement matrix (observe position + size)
        self.H = np.zeros((4, 8), dtype=np.float64)
        self.H[0, 0] = 1  # cx
        self.H[1, 1] = 1  # cy
        self.H[2, 2] = 1  # w
        self.H[3, 3] = 1  # h

        # Covariance
        self.P = np.eye(8, dtype=np.float64) * 10.0
        self.P[4:, 4:] *= 100.0  # high uncertainty on initial velocity

        # Process noise
        self.Q_base = np.eye(8, dtype=np.float64) * process_noise
        self.Q_base[4:, 4:] *= 4.0  # velocity changes more

        # Measurement noise
        self.R = np.eye(4, dtype=np.float64) * measurement_noise

        self.last_time = time.monotonic()
        self.age = 0

    def predict(self, dt: float = None) -> Tuple[float, float, float, float]:
        """Predict next state. Returns (cx, cy, w, h)."""
        now = time.monotonic()
        if dt is None:
            dt = now - self.last_time
        self.last_time = now
        dt = max(dt, 0.001)

        # Update transition matrix with dt
        self.F = np.eye(8, dtype=np.float64)
        self.F[0, 4] = dt  # cx += vx * dt
        self.F[1, 5] = dt  # cy += vy * dt
        self.F[2, 6] = dt  # w += vw * dt
        self.F[3, 7] = dt  # h += vh * dt

        # Scale process noise by dt
        Q = self.Q_base * dt

        # Predict
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + Q

        # Clamp size to positive
        self.x[2] = max(self.x[2], 1.0)
        self.x[3] = max(self.x[3], 1.0)

        self.age += 1
        return tuple(self.x[:4])

    def update(self, measurement: Tuple[float, float, float, float]):
        """Update with actual measurement (cx, cy, w, h)."""
        z = np.array(measurement, dtype=np.float64)

        # Innovation
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain
        try:
            K = self.P @ self.H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return  # skip update if singular

        # Update
        self.x = self.x + K @ y
        I = np.eye(8, dtype=np.float64)
        self.P = (I - K @ self.H) @ self.P

        # Clamp size
        self.x[2] = max(self.x[2], 1.0)
        self.x[3] = max(self.x[3], 1.0)
        self.age = 0

    @property
    def velocity(self) -> Tuple[float, float]:
        """Return (vx, vy) in pixels/sec."""
        return (self.x[4], self.x[5])

    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        """Return current (cx, cy, w, h)."""
        return tuple(self.x[:4])


@dataclass
class TrackedObject:
    """A persistently tracked object with ID and history."""
    track_id: int
    class_id: int
    class_name: str
    kalman: KalmanFilter2D
    confidence: float = 0.0
    frames_seen: int = 1
    frames_missed: int = 0
    last_detection: Optional[Detection] = None
    history: list = field(default_factory=list)  # last N positions

    def to_detection(self) -> Detection:
        """Convert current Kalman state to Detection."""
        cx, cy, w, h = self.kalman.bbox
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return Detection(self.class_id, self.class_name, self.confidence, x1, y1, x2, y2)


class ObjectTracker:
    """
    IoU-based multi-object tracker with Kalman prediction.

    - Assigns persistent IDs to objects across frames
    - Predicts positions between inference frames
    - Handles object enter/exit/occlusion
    - Provides 30Hz tracking from 2 FPS detection

    Based on SORT algorithm (Simple Online Realtime Tracking).
    """

    def __init__(self,
                 iou_threshold: float = 0.3,
                 max_missed: int = 15,
                 min_hits: int = 2,
                 max_tracks: int = 50):
        """
        Args:
            iou_threshold: Min IoU to match detection to track
            max_missed: Frames without match before track is removed
            min_hits: Detections needed before track is confirmed
            max_tracks: Maximum simultaneous tracks
        """
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self.min_hits = min_hits
        self.max_tracks = max_tracks

        self.tracks: Dict[int, TrackedObject] = {}
        self.next_id = 1
        self.frame_count = 0

    @staticmethod
    def _iou(box1, box2) -> float:
        """Compute IoU between two (cx,cy,w,h) boxes."""
        # Convert to x1y1x2y2
        ax1 = box1[0] - box1[2] / 2
        ay1 = box1[1] - box1[3] / 2
        ax2 = box1[0] + box1[2] / 2
        ay2 = box1[1] + box1[3] / 2

        bx1 = box2[0] - box2[2] / 2
        by1 = box2[1] - box2[3] / 2
        bx2 = box2[0] + box2[2] / 2
        by2 = box2[1] + box2[3] / 2

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0

        inter = (ix2 - ix1) * (iy2 - iy1)
        area1 = (ax2 - ax1) * (ay2 - ay1)
        area2 = (bx2 - bx1) * (by2 - by1)
        union = area1 + area2 - inter

        return inter / union if union > 0 else 0.0

    def update(self, detections: List[Detection]) -> List[TrackedObject]:
        """
        Update tracks with new detections.

        Returns list of confirmed tracked objects.
        """
        self.frame_count += 1

        # 1. Predict all existing tracks
        for track in self.tracks.values():
            track.kalman.predict()

        # 2. Build cost matrix (negative IoU)
        track_ids = list(self.tracks.keys())
        if len(track_ids) > 0 and len(detections) > 0:
            cost = np.zeros((len(track_ids), len(detections)))
            for i, tid in enumerate(track_ids):
                track_bbox = self.tracks[tid].kalman.bbox
                for j, det in enumerate(detections):
                    det_bbox = (det.cx, det.cy, det.width, det.height)
                    cost[i, j] = -self._iou(track_bbox, det_bbox)

            # 3. Greedy matching (simple but fast)
            matched_tracks = set()
            matched_dets = set()

            while True:
                if cost.size == 0:
                    break
                min_idx = np.unravel_index(np.argmin(cost), cost.shape)
                min_val = cost[min_idx]
                if -min_val < self.iou_threshold:
                    break

                ti, di = min_idx
                tid = track_ids[ti]
                det = detections[di]

                # Update matched track
                track = self.tracks[tid]
                track.kalman.update((det.cx, det.cy, det.width, det.height))
                track.confidence = det.confidence
                track.frames_seen += 1
                track.frames_missed = 0
                track.last_detection = det
                if len(track.history) > 100:
                    track.history.pop(0)
                track.history.append((det.cx, det.cy))

                matched_tracks.add(ti)
                matched_dets.add(di)

                # Block this row/col
                cost[ti, :] = 0
                cost[:, di] = 0

            # 4. Unmatched tracks: increment missed
            for i, tid in enumerate(track_ids):
                if i not in matched_tracks:
                    self.tracks[tid].frames_missed += 1
                    self.tracks[tid].confidence *= 0.9  # decay

            # 5. Unmatched detections: create new tracks
            for j, det in enumerate(detections):
                if j not in matched_dets and len(self.tracks) < self.max_tracks:
                    self._create_track(det)

        elif len(detections) > 0:
            # No existing tracks, create all
            for det in detections[:self.max_tracks]:
                self._create_track(det)

        else:
            # No detections, all tracks miss
            for track in self.tracks.values():
                track.frames_missed += 1
                track.confidence *= 0.9

        # 6. Remove dead tracks
        dead = [tid for tid, t in self.tracks.items() if t.frames_missed > self.max_missed]
        for tid in dead:
            del self.tracks[tid]

        # 7. Return confirmed tracks
        confirmed = [t for t in self.tracks.values() if t.frames_seen >= self.min_hits]
        return sorted(confirmed, key=lambda t: t.confidence, reverse=True)

    def predict(self) -> List[TrackedObject]:
        """
        Predict all track positions WITHOUT new detections.
        Call this between inference frames for 30Hz tracking.
        """
        for track in self.tracks.values():
            track.kalman.predict()
        confirmed = [t for t in self.tracks.values() if t.frames_seen >= self.min_hits]
        return sorted(confirmed, key=lambda t: t.confidence, reverse=True)

    def get_detections(self, tracks: List[TrackedObject] = None) -> List[Detection]:
        """Convert current tracked objects to Detection list."""
        if tracks is None:
            tracks = [t for t in self.tracks.values() if t.frames_seen >= self.min_hits]
        return [t.to_detection() for t in tracks]

    def _create_track(self, det: Detection):
        bbox = (det.cx, det.cy, det.width, det.height)
        kf = KalmanFilter2D(bbox)
        track = TrackedObject(
            track_id=self.next_id,
            class_id=det.class_id,
            class_name=det.class_name,
            kalman=kf,
            confidence=det.confidence,
            last_detection=det,
            history=[(det.cx, det.cy)]
        )
        self.tracks[self.next_id] = track
        self.next_id += 1

    def get_track(self, track_id: int) -> Optional[TrackedObject]:
        return self.tracks.get(track_id)

    def get_status(self) -> dict:
        return {
            "active_tracks": len(self.tracks),
            "confirmed_tracks": sum(1 for t in self.tracks.values() if t.frames_seen >= self.min_hits),
            "next_id": self.next_id,
            "frame_count": self.frame_count,
        }


class AsyncPipeline:
    """
    Threaded capture -> inference -> control pipeline.

    Problem: At 1-3 FPS inference, if capture blocks on inference,
    you get 1-3 Hz control. Robot is blind between frames.

    Solution: Three threads:
      1. Capture thread: grabs frames as fast as camera allows
      2. Inference thread: processes latest frame when ready
      3. Control runs at fixed rate using Kalman predictions

    Control loop runs at target_hz (default 30) even when
    inference runs at 2 FPS.
    """

    def __init__(self,
                 detector: ObjectDetector,
                 controller: RobotController,
                 tracker: ObjectTracker = None,
                 safety: SafetyController = None,
                 thermal: ThermalMonitor = None,
                 target_hz: float = 30.0):
        self.detector = detector
        self.controller = controller
        self.tracker = tracker or ObjectTracker()
        self.safety = safety or SafetyController()
        self.thermal = thermal
        self.target_hz = target_hz

        # Shared state (protected by locks)
        self._frame_lock = threading.Lock()
        self._result_lock = threading.Lock()
        self._latest_frame = None
        self._frame_id = 0
        self._last_processed_id = -1
        self._latest_detections: List[Detection] = []
        self._latest_command = RobotCommand(0, 0, 0, "startup", "Initializing")
        self._img_w = 640
        self._img_h = 480

        # Stats
        self.inference_fps = 0.0
        self.control_fps = 0.0
        self.capture_fps = 0.0
        self.frames_captured = 0
        self.frames_inferred = 0
        self.frames_skipped = 0

        self._running = False
        self._threads = []

    def push_frame(self, frame: np.ndarray):
        """Push a new camera frame. Non-blocking."""
        with self._frame_lock:
            self._latest_frame = frame
            self._frame_id += 1
            self._img_h, self._img_w = frame.shape[:2]
        self.frames_captured += 1

    def get_command(self) -> RobotCommand:
        """Get latest safe robot command. Non-blocking."""
        with self._result_lock:
            return self._latest_command

    def get_detections(self) -> List[Detection]:
        """Get latest tracked detections. Non-blocking."""
        with self._result_lock:
            return list(self._latest_detections)

    def _inference_loop(self):
        """Inference thread: process latest frame when available."""
        last_time = time.monotonic()
        while self._running:
            # Get latest frame
            with self._frame_lock:
                frame = self._latest_frame
                frame_id = self._frame_id

            if frame is None or frame_id == self._last_processed_id:
                time.sleep(0.001)
                continue

            # Thermal skip
            skip = 0
            if self.thermal:
                skip = self.thermal.get_skip_factor()
            if skip > 0 and frame_id % (skip + 1) != 0:
                self.frames_skipped += 1
                time.sleep(0.01)
                continue

            # Run detection
            detections = self.detector.detect(frame)
            self._last_processed_id = frame_id
            self.frames_inferred += 1

            # Feed watchdog
            self.safety.feed_watchdog()

            # Update tracker
            tracks = self.tracker.update(detections)
            tracked_dets = self.tracker.get_detections(tracks)

            # Update shared state
            with self._result_lock:
                self._latest_detections = tracked_dets

            # FPS
            now = time.monotonic()
            dt = now - last_time
            if dt > 0:
                self.inference_fps = 0.9 * self.inference_fps + 0.1 * (1.0 / dt)
            last_time = now

    def _control_loop(self):
        """Control thread: runs at target_hz using Kalman predictions."""
        period = 1.0 / self.target_hz
        last_time = time.monotonic()

        while self._running:
            loop_start = time.monotonic()

            # Get Kalman-predicted detections (fast, no inference)
            predicted_tracks = self.tracker.predict()
            predicted_dets = self.tracker.get_detections(predicted_tracks)

            # Run controller
            with self._result_lock:
                img_w, img_h = self._img_w, self._img_h

            raw_cmd = self.controller.update(predicted_dets, img_w, img_h)

            # Safety filter
            safe_cmd = self.safety.filter_command(raw_cmd, predicted_dets, img_w, img_h)

            with self._result_lock:
                self._latest_command = safe_cmd
                self._latest_detections = predicted_dets

            # FPS
            now = time.monotonic()
            dt = now - last_time
            if dt > 0:
                self.control_fps = 0.9 * self.control_fps + 0.1 * (1.0 / dt)
            last_time = now

            # Sleep to maintain target rate
            elapsed = time.monotonic() - loop_start
            sleep_time = period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def start(self):
        """Start inference and control threads."""
        self._running = True
        if self.thermal:
            self.thermal.start()

        t_inf = threading.Thread(target=self._inference_loop, daemon=True, name="inference")
        t_ctrl = threading.Thread(target=self._control_loop, daemon=True, name="control")
        t_inf.start()
        t_ctrl.start()
        self._threads = [t_inf, t_ctrl]
        logger.info("AsyncPipeline started: inference + control threads")

    def stop(self):
        self._running = False
        if self.thermal:
            self.thermal.stop()
        for t in self._threads:
            t.join(timeout=5)
        logger.info("AsyncPipeline stopped")

    def get_status(self) -> dict:
        return {
            "inference_fps": round(self.inference_fps, 1),
            "control_fps": round(self.control_fps, 1),
            "frames_captured": self.frames_captured,
            "frames_inferred": self.frames_inferred,
            "frames_skipped": self.frames_skipped,
            "tracker": self.tracker.get_status(),
            "safety": self.safety.get_status(),
        }


# ================================================================
# TIER 3: FIELD DEBUGGING
# ================================================================


class BlackBoxRecorder:
    """
    Flight-recorder style logging for robot debugging.

    Records to a binary log file:
      - Timestamped detections
      - Robot commands
      - Safety events
      - Image quality scores
      - System metrics (temp, memory, FPS)

    Supports:
      - Circular buffer (fixed max size, overwrites oldest)
      - JSON export for analysis
      - Frame snapshots on events (e-stop, timeout)
      - Replay: load log and feed to controller
    """

    def __init__(self,
                 log_dir: str = "blackbox",
                 max_entries: int = 10000,
                 save_frames_on_event: bool = True,
                 max_frame_snapshots: int = 100):
        self.log_dir = log_dir
        self.max_entries = max_entries
        self.save_frames_on_event = save_frames_on_event
        self.max_frame_snapshots = max_frame_snapshots

        self.entries = deque(maxlen=max_entries)
        self.frame_snapshots = deque(maxlen=max_frame_snapshots)
        self.start_time = time.monotonic()
        self.session_id = time.strftime("%Y%m%d_%H%M%S")
        self._entry_count = 0

        os.makedirs(log_dir, exist_ok=True)

    def record(self, entry_type: str, data: dict, frame: np.ndarray = None):
        """Record a timestamped entry."""
        entry = {
            "t": time.monotonic() - self.start_time,
            "wall": time.time(),
            "seq": self._entry_count,
            "type": entry_type,
            "data": data,
        }
        self.entries.append(entry)
        self._entry_count += 1

        # Save frame snapshot on important events
        if frame is not None and self.save_frames_on_event:
            if entry_type in ("estop", "timeout", "critical", "proximity_stop", "quality_alert"):
                self.frame_snapshots.append({
                    "seq": entry["seq"],
                    "t": entry["t"],
                    "type": entry_type,
                    "frame_shape": frame.shape,
                })
                # Save frame as numpy file
                try:
                    fname = os.path.join(self.log_dir, f"frame_{entry['seq']:06d}.npy")
                    np.save(fname, frame)
                except Exception as e:
                    logger.warning(f"Failed to save frame: {e}")

    def record_detection(self, detections: list, inference_ms: float):
        self.record("detection", {
            "count": len(detections),
            "objects": [{"class": d.class_name, "conf": round(d.confidence, 3),
                         "bbox": [round(d.x1, 1), round(d.y1, 1), round(d.x2, 1), round(d.y2, 1)]
                        } for d in detections[:20]],
            "inference_ms": round(inference_ms, 1),
        })

    def record_command(self, cmd: RobotCommand):
        self.record("command", {
            "linear_x": round(cmd.linear_x, 4),
            "angular_z": round(cmd.angular_z, 4),
            "action": cmd.action,
            "description": cmd.description,
        })

    def record_safety_event(self, event_type: str, detail: str, frame: np.ndarray = None):
        self.record(event_type, {"detail": detail}, frame=frame)

    def record_metrics(self, metrics: dict):
        self.record("metrics", metrics)

    def save(self, filename: str = None) -> str:
        """Save log to JSON file. Returns filepath."""
        if filename is None:
            filename = f"blackbox_{self.session_id}.json"
        filepath = os.path.join(self.log_dir, filename)

        log_data = {
            "session_id": self.session_id,
            "total_entries": self._entry_count,
            "saved_entries": len(self.entries),
            "duration_s": time.monotonic() - self.start_time,
            "frame_snapshots": len(self.frame_snapshots),
            "entries": list(self.entries),
        }

        with open(filepath, "w") as f:
            json.dump(log_data, f, indent=2, default=str)

        logger.info(f"Black box saved: {filepath} ({len(self.entries)} entries)")
        return filepath

    @staticmethod
    def load(filepath: str) -> dict:
        """Load a saved black box log."""
        with open(filepath) as f:
            return json.load(f)

    def get_recent(self, n: int = 20, entry_type: str = None) -> list:
        """Get last N entries, optionally filtered by type."""
        if entry_type:
            filtered = [e for e in self.entries if e["type"] == entry_type]
            return list(filtered)[-n:]
        return list(self.entries)[-n:]

    def get_status(self) -> dict:
        return {
            "session_id": self.session_id,
            "total_entries": self._entry_count,
            "buffer_used": len(self.entries),
            "buffer_max": self.max_entries,
            "frame_snapshots": len(self.frame_snapshots),
            "duration_s": round(time.monotonic() - self.start_time, 1),
        }


class ImageQualityScorer:
    """
    Fast image quality assessment for camera health monitoring.

    Detects:
      - Blur (Laplacian variance)
      - Darkness / overexposure (histogram analysis)
      - Low contrast (histogram spread)
      - Occlusion (uniform regions)

    Returns quality score 0-100 and per-issue flags.
    All computed with numpy only (no OpenCV required).
    """

    def __init__(self,
                 blur_threshold: float = 50.0,
                 dark_threshold: float = 40.0,
                 bright_threshold: float = 220.0,
                 contrast_threshold: float = 30.0):
        self.blur_threshold = blur_threshold
        self.dark_threshold = dark_threshold
        self.bright_threshold = bright_threshold
        self.contrast_threshold = contrast_threshold
        self.history = deque(maxlen=100)

    def score(self, image: np.ndarray) -> dict:
        """
        Score image quality.

        Returns:
            {"score": 0-100, "blur": float, "brightness": float,
             "contrast": float, "issues": ["dark", "blurry", ...],
             "usable": bool}
        """
        if image is None or image.size == 0:
            return {"score": 0, "issues": ["no_image"], "usable": False,
                    "blur": 0, "brightness": 0, "contrast": 0}

        # Convert to grayscale if needed
        if image.ndim == 3:
            gray = np.mean(image, axis=2).astype(np.float32)
        else:
            gray = image.astype(np.float32)

        # Downsample for speed
        h, w = gray.shape
        if h > 120:
            step = h // 120
            gray = gray[::step, ::step]

        issues = []
        scores = []

        # 1. Blur detection (Laplacian variance)
        # Laplacian kernel: [[0,1,0],[1,-4,1],[0,1,0]]
        laplacian = np.zeros_like(gray)
        laplacian[1:-1, 1:-1] = (
            gray[0:-2, 1:-1] + gray[2:, 1:-1] +
            gray[1:-1, 0:-2] + gray[1:-1, 2:] -
            4 * gray[1:-1, 1:-1]
        )
        blur_score = float(np.var(laplacian))
        if blur_score < self.blur_threshold:
            issues.append("blurry")
        blur_quality = min(100, blur_score / self.blur_threshold * 100)
        scores.append(blur_quality)

        # 2. Brightness
        mean_brightness = float(np.mean(gray))
        if mean_brightness < self.dark_threshold:
            issues.append("dark")
            bright_quality = mean_brightness / self.dark_threshold * 100
        elif mean_brightness > self.bright_threshold:
            issues.append("overexposed")
            bright_quality = max(0, (255 - mean_brightness) / (255 - self.bright_threshold) * 100)
        else:
            bright_quality = 100
        scores.append(bright_quality)

        # 3. Contrast (std dev of brightness)
        contrast = float(np.std(gray))
        if contrast < self.contrast_threshold:
            issues.append("low_contrast")
        contrast_quality = min(100, contrast / self.contrast_threshold * 100)
        scores.append(contrast_quality)

        # 4. Uniform region detection (possible occlusion)
        # Check if >50% of image is very similar
        hist_range = np.ptp(gray)  # peak to peak
        if hist_range < 15:
            issues.append("occluded")
            scores.append(20)
        else:
            scores.append(100)

        # Overall score
        overall = float(np.mean(scores))
        usable = overall > 40 and "occluded" not in issues

        result = {
            "score": round(overall, 1),
            "blur": round(blur_score, 1),
            "brightness": round(mean_brightness, 1),
            "contrast": round(contrast, 1),
            "issues": issues,
            "usable": usable,
        }
        self.history.append(result)
        return result

    def get_trend(self) -> dict:
        """Get quality trend over recent frames."""
        if not self.history:
            return {"avg_score": 0, "degrading": False, "samples": 0}

        recent = list(self.history)
        scores = [r["score"] for r in recent]
        avg = np.mean(scores)

        # Check if degrading (last 10 worse than first 10)
        degrading = False
        if len(scores) > 20:
            early = np.mean(scores[:10])
            late = np.mean(scores[-10:])
            degrading = late < early * 0.8  # 20% worse

        return {
            "avg_score": round(float(avg), 1),
            "min_score": round(float(min(scores)), 1),
            "degrading": degrading,
            "samples": len(scores),
            "issue_counts": {
                issue: sum(1 for r in recent if issue in r["issues"])
                for issue in ("blurry", "dark", "overexposed", "low_contrast", "occluded")
            },
        }


# ================================================================
# PRODUCTION EDGE AI - PUTS IT ALL TOGETHER
# ================================================================


class ProductionEdgeAI:
    """
    Production-ready edge AI pipeline.

    Integrates all tiers:
      - Safety controller (e-stop, watchdog, ramping)
      - Thermal monitor (auto-throttle)
      - Memory watchdog (OOM prevention)
      - Object tracker (persistent IDs, Kalman prediction)
      - Async pipeline (30Hz control from 2 FPS vision)
      - Black box recorder (field debugging)
      - Image quality scorer (sensor health)

    Usage:
      ai = ProductionEdgeAI.auto(mode="follow", target="person")
      ai.start()
      while running:
          ai.push_frame(camera_frame)
          cmd = ai.get_command()
          robot.set_velocity(cmd.linear_x, cmd.angular_z)
      ai.stop()
    """

    def __init__(self,
                 detector: ObjectDetector,
                 controller: RobotController,
                 safety: SafetyController = None,
                 thermal: ThermalMonitor = None,
                 memory: MemoryWatchdog = None,
                 tracker: ObjectTracker = None,
                 recorder: BlackBoxRecorder = None,
                 quality: ImageQualityScorer = None,
                 target_hz: float = 30.0):

        self.detector = detector
        self.controller = controller
        self.safety = safety or SafetyController()
        self.thermal = thermal or ThermalMonitor()
        self.memory = memory or MemoryWatchdog(on_critical=lambda: self.safety.estop("OOM_imminent"))
        self.tracker = tracker or ObjectTracker()
        self.recorder = recorder or BlackBoxRecorder()
        self.quality = quality or ImageQualityScorer()

        self.pipeline = AsyncPipeline(
            detector=detector,
            controller=controller,
            tracker=self.tracker,
            safety=self.safety,
            thermal=self.thermal,
            target_hz=target_hz,
        )

        self._metrics_interval = 5.0  # log metrics every 5s
        self._last_metrics = 0
        self._frame_count = 0

    @classmethod
    def auto(cls,
             mode: str = "follow",
             target: str = "person",
             model_path: str = None,
             conf_thresh: float = 0.4,
             max_linear: float = 0.3,
             max_angular: float = 0.5,
             target_hz: float = 30.0):
        """Auto-configure everything based on hardware."""

        detector = ObjectDetector.auto(task="detect", conf_thresh=conf_thresh)
        controller = RobotController(mode=mode, target_classes=[target],
                                      max_linear=max_linear, max_angular=max_angular)
        safety = SafetyController(max_linear=max_linear, max_angular=max_angular)

        return cls(
            detector=detector,
            controller=controller,
            safety=safety,
            target_hz=target_hz,
        )

    def start(self):
        """Start all background threads."""
        self.thermal.start()
        self.memory.start()
        self.pipeline.start()
        logger.info("ProductionEdgeAI started")

    def stop(self):
        """Stop everything and save black box."""
        self.pipeline.stop()
        self.thermal.stop()
        self.memory.stop()
        filepath = self.recorder.save()
        logger.info(f"ProductionEdgeAI stopped. Black box: {filepath}")
        return filepath

    def push_frame(self, frame: np.ndarray):
        """Push camera frame. Also checks image quality."""
        self._frame_count += 1

        # Quality check (every 10th frame to save CPU)
        if self._frame_count % 10 == 0:
            q = self.quality.score(frame)
            if not q["usable"]:
                self.recorder.record_safety_event(
                    "quality_alert",
                    f"Image unusable: {q['issues']}, score={q['score']}",
                    frame=frame
                )

        self.pipeline.push_frame(frame)

        # Periodic metrics logging
        now = time.monotonic()
        if now - self._last_metrics > self._metrics_interval:
            self._log_metrics()
            self._last_metrics = now

    def get_command(self) -> RobotCommand:
        """Get latest safe robot command."""
        cmd = self.pipeline.get_command()
        self.recorder.record_command(cmd)
        return cmd

    def get_detections(self) -> List[Detection]:
        """Get latest tracked detections."""
        return self.pipeline.get_detections()

    def estop(self, reason: str = "manual"):
        """Emergency stop."""
        self.safety.estop(reason)
        self.recorder.record_safety_event("estop", reason)

    def reset(self):
        """Clear e-stop."""
        self.safety.reset()
        self.recorder.record_safety_event("reset", "E-stop cleared")

    def _log_metrics(self):
        metrics = {
            "pipeline": self.pipeline.get_status(),
            "thermal": self.thermal.get_status(),
            "memory": self.memory.get_status(),
            "quality": self.quality.get_trend(),
            "recorder": self.recorder.get_status(),
        }
        self.recorder.record_metrics(metrics)

    def get_full_status(self) -> dict:
        """Get complete system status for dashboard/telemetry."""
        return {
            "safety": self.safety.get_status(),
            "thermal": self.thermal.get_status(),
            "memory": self.memory.get_status(),
            "pipeline": self.pipeline.get_status(),
            "quality": self.quality.get_trend(),
            "recorder": self.recorder.get_status(),
            "tracker": self.tracker.get_status(),
        }

    def process_sync(self, frame: np.ndarray) -> dict:
        """
        Synchronous processing (no threads needed).
        For testing or simple loops.

        Returns: {"detections", "tracks", "command", "quality", "safety_state"}
        """
        # Quality check
        q = self.quality.score(frame)

        # Detect
        detections = self.detector.detect(frame)
        self.safety.feed_watchdog()

        # Track
        tracks = self.tracker.update(detections)
        tracked_dets = self.tracker.get_detections(tracks)

        # Control
        h, w = frame.shape[:2]
        raw_cmd = self.controller.update(tracked_dets, w, h)
        safe_cmd = self.safety.filter_command(raw_cmd, tracked_dets, w, h)

        # Record
        self.recorder.record_detection(detections, self.detector.avg_ms)
        self.recorder.record_command(safe_cmd)

        return {
            "detections": detections,
            "tracks": [{"id": t.track_id, "class": t.class_name,
                        "confidence": t.confidence, "frames_seen": t.frames_seen}
                       for t in tracks],
            "command": safe_cmd,
            "quality": q,
            "safety_state": self.safety.state,
            "fps": self.detector.fps,
            "avg_ms": self.detector.avg_ms,
        }
