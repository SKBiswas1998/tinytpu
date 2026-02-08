"""
TinyTPU Pipeline - Full perception-to-action loop.
camera -> detect -> track -> decide -> act, all async with safety.

Usage:
    pipeline = Pipeline(model="yolov8n", mode="follow", target="person")
    pipeline.start()
"""

import time
import threading
import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger("tinytpu.control.pipeline")


@dataclass
class PipelineConfig:
    model: str = "yolov8n"
    conf_threshold: float = 0.4
    backend: str = "auto"
    mode: str = "detect"
    target: str = "person"
    camera_id: int = 0
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: int = 30
    control_hz: float = 30.0
    enable_safety: bool = True
    enable_tracking: bool = True
    enable_thermal: bool = True
    enable_recording: bool = False
    max_linear: float = 0.3
    max_angular: float = 0.5
    on_detection: Optional[Callable] = None
    on_command: Optional[Callable] = None
    on_state_change: Optional[Callable] = None


@dataclass
class PipelineState:
    running: bool = False
    mode: str = "detect"
    fps_capture: float = 0.0
    fps_inference: float = 0.0
    fps_control: float = 0.0
    num_detections: int = 0
    num_tracks: int = 0
    target_acquired: bool = False
    target_center: Tuple[float, float] = (0.0, 0.0)
    target_size: float = 0.0
    command_linear: float = 0.0
    command_angular: float = 0.0
    safety_state: str = "ok"
    thermal_temp: float = 0.0
    uptime_seconds: float = 0.0


class Pipeline:
    """
    Complete perception-to-action pipeline.
    Solves vision-to-control frequency mismatch via Kalman prediction.
    """

    def __init__(self, model="yolov8n", mode="detect", target="person", **kwargs):
        self.config = PipelineConfig(model=model, mode=mode, target=target, **kwargs)
        self._state = PipelineState(mode=mode)
        self._model = None
        self._tracker = None
        self._safety = None
        self._thermal = None
        self._recorder = None
        self._running = False
        self._threads = []
        self._lock = threading.Lock()
        self._latest_frame = None
        self._latest_detections = []
        self._latest_tracks = []
        self._frame_event = threading.Event()
        self._start_time = 0.0
        self._capture_times = []
        self._inference_times = []
        self._control_times = []

    def _init_components(self):
        from tinytpu.inference.model_zoo import Model
        self._model = Model(self.config.model, conf_threshold=self.config.conf_threshold, backend=self.config.backend)
        if self.config.enable_tracking:
            from tinytpu.perception.tracker import ObjectTracker
            self._tracker = ObjectTracker(max_age=15, iou_threshold=0.3)
        if self.config.enable_safety:
            from tinytpu.control.safety import SafetyController
            self._safety = SafetyController(max_linear=self.config.max_linear, max_angular=self.config.max_angular)
        if self.config.enable_thermal:
            from tinytpu.monitoring.thermal import ThermalMonitor
            self._thermal = ThermalMonitor()
        if self.config.enable_recording:
            from tinytpu.monitoring.recorder import BlackBoxRecorder
            self._recorder = BlackBoxRecorder()

    def start(self, blocking=True):
        if self._running:
            return
        self._init_components()
        self._running = True
        self._start_time = time.monotonic()
        self._state.running = True
        self._threads = [
            threading.Thread(target=self._capture_loop, daemon=True, name="capture"),
            threading.Thread(target=self._inference_loop, daemon=True, name="inference"),
            threading.Thread(target=self._control_loop, daemon=True, name="control"),
        ]
        for t in self._threads:
            t.start()
        if self._thermal:
            self._thermal.start()
        if blocking:
            try:
                while self._running:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                pass
            finally:
                self.stop()

    def stop(self):
        if not self._running:
            return
        self._running = False
        self._frame_event.set()
        for t in self._threads:
            t.join(timeout=3.0)
        if self._thermal:
            self._thermal.stop()
        self._state.running = False

    def _capture_loop(self):
        cap = None
        try:
            import cv2
            cap = cv2.VideoCapture(self.config.camera_id)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.camera_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.camera_height)
            cap.set(cv2.CAP_PROP_FPS, self.config.camera_fps)
            if not cap.isOpened():
                self._running = False
                return
            while self._running:
                t0 = time.perf_counter()
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.01)
                    continue
                with self._lock:
                    self._latest_frame = frame
                self._frame_event.set()
                self._capture_times.append(time.perf_counter() - t0)
                if len(self._capture_times) > 100:
                    self._capture_times = self._capture_times[-50:]
        except ImportError:
            self._running = False
        finally:
            if cap:
                cap.release()

    def _inference_loop(self):
        while self._running:
            self._frame_event.wait(timeout=1.0)
            self._frame_event.clear()
            with self._lock:
                frame = self._latest_frame
            if frame is None:
                continue
            t0 = time.perf_counter()
            try:
                results = self._model.predict(frame)
                if self.config.mode in ("follow", "track"):
                    detections = results.filter(class_name=self.config.target)
                else:
                    detections = results.detections
                tracks = self._tracker.update(detections) if self._tracker and detections else []
                with self._lock:
                    self._latest_detections = detections
                    self._latest_tracks = tracks
                if self._safety and detections:
                    self._safety.feed_watchdog()
                if self.config.on_detection:
                    self.config.on_detection(detections, tracks)
            except Exception as e:
                logger.error(f"Inference error: {e}")
            self._inference_times.append(time.perf_counter() - t0)
            if len(self._inference_times) > 100:
                self._inference_times = self._inference_times[-50:]

    def _control_loop(self):
        period = 1.0 / self.config.control_hz
        while self._running:
            t0 = time.perf_counter()
            with self._lock:
                detections = list(self._latest_detections)
                tracks = list(self._latest_tracks)
                frame = self._latest_frame
            cmd = self._generate_command(detections, tracks, frame)
            if self._safety:
                img_h = frame.shape[0] if frame is not None else 480
                img_w = frame.shape[1] if frame is not None else 640
                safe = self._safety.filter_command(cmd, detections, img_w, img_h)
                with self._lock:
                    self._state.command_linear = safe.linear_x
                    self._state.command_angular = safe.angular_z
                    self._state.safety_state = self._safety.state
            else:
                with self._lock:
                    self._state.command_linear = cmd.get("linear_x", 0)
                    self._state.command_angular = cmd.get("angular_z", 0)
            if self.config.on_command:
                self.config.on_command(self._state.command_linear, self._state.command_angular)
            self._control_times.append(time.perf_counter() - t0)
            if len(self._control_times) > 100:
                self._control_times = self._control_times[-50:]
            self._update_state(detections, tracks)
            sleep_time = period - (time.perf_counter() - t0)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _generate_command(self, detections, tracks, frame):
        if self.config.mode == "detect":
            return {"linear_x": 0.0, "angular_z": 0.0}
        if self.config.mode == "follow":
            return self._follow_command(detections, tracks, frame)
        if self.config.mode == "patrol":
            if detections:
                return {"linear_x": 0.0, "angular_z": 0.0}
            t = time.monotonic() - self._start_time
            return {"linear_x": 0.0, "angular_z": float(0.2 * np.sign(np.sin(t * 0.5)))}
        return {"linear_x": 0.0, "angular_z": 0.0}

    def _follow_command(self, detections, tracks, frame):
        target = None
        if tracks:
            target = max(tracks, key=lambda t: getattr(t, "area", 0), default=None)
        elif detections:
            targets = [d for d in detections if d.class_name == self.config.target]
            if targets:
                target = max(targets, key=lambda d: d.area)
        if target is None:
            with self._lock:
                self._state.target_acquired = False
            return {"linear_x": 0.0, "angular_z": 0.0}
        img_w = frame.shape[1] if frame is not None else 640
        img_h = frame.shape[0] if frame is not None else 480
        cx = (getattr(target, "x1", 0) + getattr(target, "x2", 0)) / 2
        cy = (getattr(target, "y1", 0) + getattr(target, "y2", 0)) / 2
        error_x = (cx - img_w / 2) / (img_w / 2)
        angular_z = -error_x * self.config.max_angular
        target_h = getattr(target, "y2", 0) - getattr(target, "y1", 0)
        ratio = target_h / img_h
        if ratio > 0.6:
            linear_x = 0.0
        elif ratio > 0.3:
            linear_x = self.config.max_linear * 0.3
        elif ratio > 0.1:
            linear_x = self.config.max_linear * 0.7
        else:
            linear_x = self.config.max_linear
        with self._lock:
            self._state.target_acquired = True
            self._state.target_center = (cx / img_w, cy / img_h)
            self._state.target_size = ratio
        return {"linear_x": linear_x, "angular_z": angular_z}

    def _update_state(self, detections, tracks):
        with self._lock:
            self._state.num_detections = len(detections)
            self._state.num_tracks = len(tracks)
            self._state.uptime_seconds = time.monotonic() - self._start_time
            if self._capture_times:
                self._state.fps_capture = 1.0 / max(0.001, np.mean(self._capture_times[-10:]))
            if self._inference_times:
                self._state.fps_inference = 1.0 / max(0.001, np.mean(self._inference_times[-10:]))
            if self._control_times:
                self._state.fps_control = 1.0 / max(0.001, np.mean(self._control_times[-10:]))
            if self._thermal:
                self._state.thermal_temp = self._thermal.get_temp()

    def get_state(self) -> dict:
        with self._lock:
            return {
                "running": self._state.running, "mode": self._state.mode,
                "fps_capture": round(self._state.fps_capture, 1),
                "fps_inference": round(self._state.fps_inference, 1),
                "fps_control": round(self._state.fps_control, 1),
                "num_detections": self._state.num_detections,
                "num_tracks": self._state.num_tracks,
                "target_acquired": self._state.target_acquired,
                "target_center": self._state.target_center,
                "target_size": round(self._state.target_size, 3),
                "command_linear": round(self._state.command_linear, 3),
                "command_angular": round(self._state.command_angular, 3),
                "safety_state": self._state.safety_state,
                "thermal_temp": round(self._state.thermal_temp, 1),
                "uptime_seconds": round(self._state.uptime_seconds, 1),
            }

    def __enter__(self):
        self.start(blocking=False)
        return self

    def __exit__(self, *args):
        self.stop()
