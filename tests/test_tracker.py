"""Tests for KalmanFilter2D and ObjectTracker — the vision-to-control bridge."""

import pytest
import numpy as np
from tinytpu.perception.tracker import KalmanFilter2D, ObjectTracker, TrackedObject


class TestKalmanFilter:
    """Kalman filter correctness and numerical stability."""

    def test_initial_state(self, kalman_filter):
        assert kalman_filter._initialized is True
        cx, cy, w, h = kalman_filter.bbox
        assert cx == 100.0
        assert cy == 100.0
        assert w == 50.0
        assert h == 50.0

    def test_predict_moves_state(self, kalman_filter):
        # Set velocity
        kalman_filter.x[4] = 10.0  # vx = 10 px/frame
        kalman_filter.x[5] = 5.0   # vy = 5 px/frame
        pred = kalman_filter.predict()
        # Position should shift by velocity * dt
        assert pred[0] > 100.0
        assert pred[1] > 100.0

    def test_predict_custom_dt(self, kalman_filter):
        kalman_filter.x[4] = 30.0  # 30 px/sec
        pred = kalman_filter.predict(dt=1.0)
        assert abs(pred[0] - 130.0) < 1.0  # should move ~30px

    def test_update_corrects_state(self, kalman_filter):
        kalman_filter.predict()
        updated = kalman_filter.update((110, 110, 50, 50))
        # After update, state should be closer to measurement
        assert abs(updated[0] - 110) < 20
        assert abs(updated[1] - 110) < 20

    def test_width_height_never_negative(self, kalman_filter):
        # Force negative width/height via state manipulation
        kalman_filter.x[2] = -100.0
        kalman_filter.x[3] = -100.0
        pred = kalman_filter.predict()
        assert pred[2] >= 1.0
        assert pred[3] >= 1.0

    def test_update_width_height_never_negative(self, kalman_filter):
        kalman_filter.predict()
        result = kalman_filter.update((100, 100, -50, -50))
        assert result[2] >= 1.0
        assert result[3] >= 1.0

    def test_velocity_property(self, kalman_filter):
        kalman_filter.x[4] = 5.0
        kalman_filter.x[5] = -3.0
        vx, vy = kalman_filter.velocity
        assert vx == 5.0
        assert vy == -3.0

    def test_bbox_xyxy_property(self, kalman_filter):
        x1, y1, x2, y2 = kalman_filter.bbox_xyxy
        assert x1 == 75.0   # 100 - 50/2
        assert y1 == 75.0
        assert x2 == 125.0  # 100 + 50/2
        assert y2 == 125.0

    # --- Numerical stability tests ---

    def test_near_singular_covariance_survives(self):
        """The fix: np.linalg.solve instead of np.linalg.inv."""
        kf = KalmanFilter2D(initial_bbox=(100, 100, 50, 50))
        kf.P = np.eye(8) * 1e-12  # near-singular
        kf.predict()
        result = kf.update((101, 101, 50, 50))
        assert np.all(np.isfinite(result)), "Update must not produce NaN/Inf"

    def test_very_large_covariance(self):
        """Extremely uncertain state."""
        kf = KalmanFilter2D(initial_bbox=(100, 100, 50, 50))
        kf.P = np.eye(8) * 1e10
        kf.predict()
        result = kf.update((200, 200, 60, 60))
        assert np.all(np.isfinite(result))
        # With huge uncertainty, update should snap close to measurement
        assert abs(result[0] - 200) < 10

    def test_repeated_predict_without_update(self):
        """Covariance grows but stays finite."""
        kf = KalmanFilter2D(initial_bbox=(100, 100, 50, 50))
        for _ in range(1000):
            kf.predict()
        assert np.all(np.isfinite(kf.x))
        assert np.all(np.isfinite(kf.P))

    def test_rapid_predict_update_cycles(self):
        """Many cycles should remain stable."""
        kf = KalmanFilter2D(initial_bbox=(100, 100, 50, 50))
        for i in range(500):
            kf.predict()
            kf.update((100 + i * 0.1, 100, 50, 50))
        assert np.all(np.isfinite(kf.x))
        assert np.all(np.isfinite(kf.P))

    def test_zero_measurement_noise(self):
        """R = 0 means measurements are perfect — should still work."""
        kf = KalmanFilter2D(initial_bbox=(100, 100, 50, 50))
        kf.R = np.eye(4) * 1e-15  # nearly zero noise
        kf.predict()
        result = kf.update((105, 105, 50, 50))
        assert np.all(np.isfinite(result))

    def test_uninitialized_filter(self):
        """Filter created without initial bbox."""
        kf = KalmanFilter2D()
        assert kf._initialized is False
        pred = kf.predict()
        assert np.all(np.isfinite(pred))
        result = kf.update((50, 50, 30, 30))
        assert kf._initialized is True


class TestTrackedObject:
    """TrackedObject properties and lifecycle."""

    def test_bbox_xyxy_from_kalman(self):
        kf = KalmanFilter2D(initial_bbox=(100, 100, 50, 50))
        t = TrackedObject(track_id=1, class_name="person", confidence=0.9, kalman=kf)
        x1, y1, x2, y2 = t.bbox_xyxy
        assert x1 == 75.0
        assert y2 == 125.0

    def test_area_calculation(self):
        kf = KalmanFilter2D(initial_bbox=(100, 100, 50, 50))
        t = TrackedObject(track_id=1, class_name="person", confidence=0.9, kalman=kf)
        assert t.area == 2500.0  # 50 * 50

    def test_center_property(self):
        kf = KalmanFilter2D(initial_bbox=(200, 300, 50, 50))
        t = TrackedObject(track_id=1, class_name="car", confidence=0.8, kalman=kf)
        cx, cy = t.center
        assert cx == 200.0
        assert cy == 300.0

    def test_x1_y1_x2_y2_properties(self):
        kf = KalmanFilter2D(initial_bbox=(100, 100, 40, 60))
        t = TrackedObject(track_id=1, class_name="person", confidence=0.9, kalman=kf)
        assert t.x1 == 80.0   # 100 - 40/2
        assert t.y1 == 70.0   # 100 - 60/2
        assert t.x2 == 120.0  # 100 + 40/2
        assert t.y2 == 130.0  # 100 + 60/2


class TestObjectTracker:
    """SORT-style multi-object tracker."""

    def test_empty_update_returns_empty(self, tracker):
        result = tracker.update([])
        assert result == []

    def test_single_detection_creates_track(self, tracker, single_person_detection):
        # First frame: track created but tentative (min_hits=2)
        result = tracker.update(single_person_detection)
        assert len(result) == 0  # not yet confirmed

        # Second frame: same detection → confirmed
        result = tracker.update(single_person_detection)
        assert len(result) == 1
        assert result[0].class_name == "person"
        assert result[0].track_id == 1

    def test_track_ids_are_unique(self, tracker, sample_detections):
        tracker.update(sample_detections)
        tracker.update(sample_detections)
        confirmed = tracker.update(sample_detections)
        ids = [t.track_id for t in confirmed]
        assert len(ids) == len(set(ids)), "Track IDs must be unique"

    def test_track_persists_across_frames(self, tracker, single_person_detection):
        for _ in range(5):
            tracker.update(single_person_detection)
        result = tracker.update(single_person_detection)
        assert len(result) == 1
        assert result[0].track_id == 1  # same ID across frames

    def test_track_deleted_after_max_missed(self):
        tracker = ObjectTracker(max_age=3, min_hits=1)
        from tests.conftest import FakeDetection
        det = [FakeDetection(0, "person", 0.9, 100, 100, 200, 200)]

        # Create and confirm track
        tracker.update(det)
        result = tracker.update(det)
        assert len(result) == 1

        # Miss detections for max_age frames
        for _ in range(4):
            result = tracker.update([])
        assert len(result) == 0  # track should be deleted

    def test_multiple_objects_tracked(self, tracker):
        from tests.conftest import FakeDetection
        dets = [
            FakeDetection(0, "person", 0.9, 100, 100, 200, 200),
            FakeDetection(2, "car", 0.8, 400, 300, 550, 450),
        ]
        for _ in range(3):
            tracker.update(dets)
        result = tracker.update(dets)
        assert len(result) == 2
        classes = {t.class_name for t in result}
        assert classes == {"person", "car"}

    def test_predict_without_update(self, tracker, single_person_detection):
        # Build up tracks
        for _ in range(3):
            tracker.update(single_person_detection)
        # Predict only (no new detections)
        result = tracker.predict()
        assert len(result) == 1  # confirmed track still exists

    def test_max_tracks_limit(self):
        """max_tracks limits new track creation for unmatched detections."""
        tracker = ObjectTracker(max_tracks=3, min_hits=1)
        from tests.conftest import FakeDetection
        # First frame: creates initial tracks (no limit on first batch since no existing tracks)
        dets1 = [FakeDetection(0, "a", 0.9, 0, 0, 50, 50)]
        tracker.update(dets1)
        assert len(tracker._tracks) == 1
        # Add more unmatched detections — should be limited
        dets2 = [FakeDetection(i, f"obj{i}", 0.9, i*200, 0, i*200+50, 50) for i in range(10)]
        tracker.update(dets2)
        assert len(tracker._tracks) <= 4  # 1 existing + up to max_tracks new

    def test_iou_matching_accuracy(self):
        """Overlapping boxes should be matched, distant boxes should not."""
        tracker = ObjectTracker(iou_threshold=0.3, min_hits=1)
        from tests.conftest import FakeDetection

        # Frame 1: person at (100,100)-(200,200)
        det1 = [FakeDetection(0, "person", 0.9, 100, 100, 200, 200)]
        tracker.update(det1)

        # Frame 2: person shifted slightly (high IoU → same track)
        det2 = [FakeDetection(0, "person", 0.9, 110, 110, 210, 210)]
        result = tracker.update(det2)
        assert len(result) == 1
        assert result[0].track_id == 1

    def test_no_match_for_distant_boxes(self):
        """Very different positions should create new tracks."""
        tracker = ObjectTracker(iou_threshold=0.3, min_hits=1)
        from tests.conftest import FakeDetection

        det1 = [FakeDetection(0, "person", 0.9, 0, 0, 50, 50)]
        tracker.update(det1)

        # Completely different location
        det2 = [FakeDetection(0, "person", 0.9, 500, 500, 600, 600)]
        tracker.update(det2)
        # Should have 2 tracks now (original missed, new one created)
        assert len(tracker._tracks) == 2


class TestIoUBatch:
    """Vectorized IoU computation."""

    def test_identical_boxes(self):
        a = np.array([[0, 0, 100, 100]], dtype=np.float64)
        iou = ObjectTracker._iou_batch(a, a)
        assert abs(iou[0, 0] - 1.0) < 1e-5

    def test_no_overlap(self):
        a = np.array([[0, 0, 50, 50]], dtype=np.float64)
        b = np.array([[100, 100, 200, 200]], dtype=np.float64)
        iou = ObjectTracker._iou_batch(a, b)
        assert iou[0, 0] < 1e-5

    def test_partial_overlap(self):
        a = np.array([[0, 0, 100, 100]], dtype=np.float64)
        b = np.array([[50, 50, 150, 150]], dtype=np.float64)
        iou = ObjectTracker._iou_batch(a, b)
        # intersection = 50*50 = 2500, union = 10000 + 10000 - 2500 = 17500
        expected = 2500.0 / 17500.0
        assert abs(iou[0, 0] - expected) < 0.01

    def test_batch_shape(self):
        a = np.array([[0, 0, 50, 50], [100, 100, 200, 200]], dtype=np.float64)
        b = np.array([[25, 25, 75, 75], [150, 150, 250, 250], [300, 300, 400, 400]], dtype=np.float64)
        iou = ObjectTracker._iou_batch(a, b)
        assert iou.shape == (2, 3)

    def test_zero_area_box(self):
        a = np.array([[50, 50, 50, 50]], dtype=np.float64)  # zero area
        b = np.array([[0, 0, 100, 100]], dtype=np.float64)
        iou = ObjectTracker._iou_batch(a, b)
        assert iou[0, 0] < 1e-5  # zero area → zero IoU
