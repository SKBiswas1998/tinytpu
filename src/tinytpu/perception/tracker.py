"""
TinyTPU Object Tracking - Kalman filter + IoU-based SORT tracker.
Enables 30 Hz control from 2 FPS vision via Kalman prediction.
"""

import time
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np

logger = logging.getLogger("tinytpu.perception.tracker")


class KalmanFilter2D:
    """
    2D Kalman filter for bounding box tracking.
    State: [cx, cy, w, h, vx, vy, vw, vh]
    Measurement: [cx, cy, w, h]
    """

    def __init__(self, initial_bbox: Tuple[float, float, float, float] = None):
        self.x = np.zeros(8, dtype=np.float64)
        self.F = np.eye(8, dtype=np.float64)
        self.dt = 1.0 / 30.0
        for i in range(4):
            self.F[i, i + 4] = self.dt
        self.H = np.zeros((4, 8), dtype=np.float64)
        self.H[:4, :4] = np.eye(4)
        self.P = np.eye(8, dtype=np.float64) * 100.0
        self.P[4:, 4:] *= 10.0
        self.Q = np.eye(8, dtype=np.float64)
        self.Q[:4, :4] *= 1.0
        self.Q[4:, 4:] *= 0.1
        self.R = np.eye(4, dtype=np.float64) * 5.0
        self._initialized = False
        if initial_bbox is not None:
            self.x[:4] = np.array(initial_bbox, dtype=np.float64)
            self._initialized = True

    def predict(self, dt: float = None) -> np.ndarray:
        if dt is not None:
            for i in range(4):
                self.F[i, i + 4] = dt
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.x[2] = max(self.x[2], 1.0)
        self.x[3] = max(self.x[3], 1.0)
        return self.x[:4].copy()

    def update(self, measurement) -> np.ndarray:
        z = np.array(measurement, dtype=np.float64)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = np.linalg.solve(S.T, (self.P @ self.H.T).T).T
        self.x = self.x + K @ y
        self.P = (np.eye(8) - K @ self.H) @ self.P
        self.x[2] = max(self.x[2], 1.0)
        self.x[3] = max(self.x[3], 1.0)
        self._initialized = True
        return self.x[:4].copy()

    @property
    def bbox(self):
        return tuple(self.x[:4])

    @property
    def velocity(self):
        return (self.x[4], self.x[5])

    @property
    def bbox_xyxy(self):
        cx, cy, w, h = self.x[:4]
        return (cx - w/2, cy - h/2, cx + w/2, cy + h/2)


@dataclass
class TrackedObject:
    track_id: int
    class_name: str
    confidence: float
    kalman: KalmanFilter2D
    frames_seen: int = 0
    frames_missed: int = 0
    hits: int = 0
    state: str = "tentative"

    @property
    def bbox_xyxy(self):
        return self.kalman.bbox_xyxy

    @property
    def center(self):
        return self.kalman.bbox[:2]

    @property
    def x1(self):
        return self.bbox_xyxy[0]

    @property
    def y1(self):
        return self.bbox_xyxy[1]

    @property
    def x2(self):
        return self.bbox_xyxy[2]

    @property
    def y2(self):
        return self.bbox_xyxy[3]

    @property
    def area(self):
        x1, y1, x2, y2 = self.bbox_xyxy
        return max(0, x2 - x1) * max(0, y2 - y1)


class ObjectTracker:
    """SORT-style multi-object tracker with Kalman prediction."""

    def __init__(self, iou_threshold=0.3, max_age=15, max_missed=None, min_hits=2, max_tracks=50):
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed if max_missed is not None else max_age
        self.min_hits = min_hits
        self.max_tracks = max_tracks
        self._tracks: List[TrackedObject] = []
        self._next_id = 1

    def update(self, detections) -> List[TrackedObject]:
        for track in self._tracks:
            track.kalman.predict()
        if not detections:
            for track in self._tracks:
                track.frames_missed += 1
            self._cleanup()
            return self._get_confirmed()

        det_boxes = np.array([[d.x1, d.y1, d.x2, d.y2] for d in detections], dtype=np.float64)

        if not self._tracks:
            for det in detections:
                self._create_track(det)
            return self._get_confirmed()

        track_boxes = np.array([list(t.bbox_xyxy) for t in self._tracks], dtype=np.float64)
        iou_matrix = self._iou_batch(track_boxes, det_boxes)
        mt, md, umt, umd = self._match(iou_matrix)

        for ti, di in zip(mt, md):
            t = self._tracks[ti]
            d = detections[di]
            cx, cy = (d.x1+d.x2)/2, (d.y1+d.y2)/2
            w, h = d.x2-d.x1, d.y2-d.y1
            t.kalman.update((cx, cy, w, h))
            t.class_name = d.class_name
            t.confidence = d.confidence
            t.frames_seen += 1
            t.hits += 1
            t.frames_missed = 0
            if t.hits >= self.min_hits:
                t.state = "confirmed"

        for ti in umt:
            self._tracks[ti].frames_missed += 1
        for di in umd:
            if len(self._tracks) < self.max_tracks:
                self._create_track(detections[di])

        self._cleanup()
        return self._get_confirmed()

    def predict(self) -> List[TrackedObject]:
        for t in self._tracks:
            t.kalman.predict()
        return self._get_confirmed()

    def _create_track(self, det):
        cx, cy = (det.x1+det.x2)/2, (det.y1+det.y2)/2
        w, h = det.x2-det.x1, det.y2-det.y1
        self._tracks.append(TrackedObject(
            track_id=self._next_id, class_name=det.class_name,
            confidence=det.confidence,
            kalman=KalmanFilter2D(initial_bbox=(cx, cy, w, h)),
            frames_seen=1, hits=1,
        ))
        self._next_id += 1

    def _match(self, iou_matrix):
        nt, nd = iou_matrix.shape
        mt, md, ut, ud = [], [], set(range(nt)), set(range(nd))
        indices = np.argsort(-iou_matrix.ravel())
        for idx in indices:
            t, d = idx // nd, idx % nd
            if t not in ut or d not in ud:
                continue
            if iou_matrix[t, d] < self.iou_threshold:
                break
            mt.append(t); md.append(d)
            ut.discard(t); ud.discard(d)
        return mt, md, list(ut), list(ud)

    def _cleanup(self):
        self._tracks = [t for t in self._tracks if t.frames_missed <= self.max_missed]

    def _get_confirmed(self):
        return [t for t in self._tracks if t.state == "confirmed"]

    @staticmethod
    def _iou_batch(a, b):
        x1 = np.maximum(a[:, 0:1], b[:, 0:1].T)
        y1 = np.maximum(a[:, 1:2], b[:, 1:2].T)
        x2 = np.minimum(a[:, 2:3], b[:, 2:3].T)
        y2 = np.minimum(a[:, 3:4], b[:, 3:4].T)
        inter = np.maximum(0, x2-x1) * np.maximum(0, y2-y1)
        aa = (a[:,2]-a[:,0]) * (a[:,3]-a[:,1])
        ab = (b[:,2]-b[:,0]) * (b[:,3]-b[:,1])
        return inter / (aa[:, None] + ab[None, :] - inter + 1e-6)
