"""Tests for Model Zoo — download, cache, load, and inference pipeline."""

import pytest
import numpy as np
from pathlib import Path
from tinytpu.inference.model_zoo import (
    MODEL_REGISTRY, COCO_CLASSES, Detection, PredictionResult, ModelZoo, Model,
)


class TestModelRegistry:
    """Model registry structure and content."""

    def test_has_direct_onnx_models(self):
        """Phase 1 fix: at least some models must be direct ONNX (no ultralytics)."""
        direct_onnx = [name for name, info in MODEL_REGISTRY.items()
                       if info["url"].endswith(".onnx")]
        assert len(direct_onnx) >= 2, f"Need direct ONNX models, got: {direct_onnx}"

    def test_yolov5n_is_direct_onnx(self):
        assert "yolov5n" in MODEL_REGISTRY
        assert MODEL_REGISTRY["yolov5n"]["url"].endswith(".onnx")

    def test_yolov5s_is_direct_onnx(self):
        assert "yolov5s" in MODEL_REGISTRY
        assert MODEL_REGISTRY["yolov5s"]["url"].endswith(".onnx")

    def test_mobilenetv2_is_direct_onnx(self):
        assert "mobilenetv2" in MODEL_REGISTRY
        assert MODEL_REGISTRY["mobilenetv2"]["url"].endswith(".onnx")

    def test_yolov8_models_present(self):
        for name in ["yolov8n", "yolov8s", "yolov8m"]:
            assert name in MODEL_REGISTRY

    def test_all_models_have_required_fields(self):
        required = {"task", "format", "size", "url", "input_size", "classes", "description"}
        for name, info in MODEL_REGISTRY.items():
            missing = required - set(info.keys())
            assert not missing, f"{name} missing fields: {missing}"

    def test_all_models_have_valid_task(self):
        valid_tasks = {"detection", "classification", "pose", "segmentation"}
        for name, info in MODEL_REGISTRY.items():
            assert info["task"] in valid_tasks, f"{name} has invalid task: {info['task']}"

    def test_all_models_have_input_size_tuple(self):
        for name, info in MODEL_REGISTRY.items():
            assert isinstance(info["input_size"], tuple)
            assert len(info["input_size"]) == 4
            assert info["input_size"][0] == 1  # batch size
            assert info["input_size"][1] == 3  # channels

    def test_model_count(self):
        assert len(MODEL_REGISTRY) >= 5


class TestCOCOClasses:

    def test_count(self):
        assert len(COCO_CLASSES) == 80

    def test_person_is_first(self):
        assert COCO_CLASSES[0] == "person"

    def test_common_classes_present(self):
        for cls in ["person", "car", "dog", "cat", "bicycle"]:
            assert cls in COCO_CLASSES


class TestDetection:

    def test_creation(self):
        d = Detection(0, "person", 0.95, 10, 20, 110, 220)
        assert d.class_id == 0
        assert d.class_name == "person"
        assert d.confidence == 0.95

    def test_center(self):
        d = Detection(0, "person", 0.9, 100, 100, 200, 300)
        cx, cy = d.center
        assert cx == 150.0
        assert cy == 200.0

    def test_area(self):
        d = Detection(0, "person", 0.9, 0, 0, 100, 200)
        assert d.area == 20000.0

    def test_area_zero_for_invalid_box(self):
        d = Detection(0, "person", 0.9, 200, 200, 100, 100)  # inverted
        assert d.area == 0.0


class TestPredictionResult:

    def test_empty_result(self):
        r = PredictionResult()
        assert len(r) == 0
        assert r.detections == []

    def test_filter_by_class(self):
        dets = [
            Detection(0, "person", 0.9, 0, 0, 50, 50),
            Detection(2, "car", 0.8, 100, 100, 200, 200),
            Detection(0, "person", 0.7, 300, 300, 400, 400),
        ]
        r = PredictionResult(detections=dets)
        persons = r.filter(class_name="person")
        assert len(persons) == 2
        cars = r.filter(class_name="car")
        assert len(cars) == 1

    def test_filter_by_confidence(self):
        dets = [
            Detection(0, "person", 0.9, 0, 0, 50, 50),
            Detection(0, "person", 0.3, 100, 100, 200, 200),
        ]
        r = PredictionResult(detections=dets)
        high_conf = r.filter(min_confidence=0.5)
        assert len(high_conf) == 1
        assert high_conf[0].confidence == 0.9

    def test_filter_combined(self):
        dets = [
            Detection(0, "person", 0.9, 0, 0, 50, 50),
            Detection(0, "person", 0.3, 100, 100, 200, 200),
            Detection(2, "car", 0.8, 300, 300, 400, 400),
        ]
        r = PredictionResult(detections=dets)
        result = r.filter(class_name="person", min_confidence=0.5)
        assert len(result) == 1

    def test_save_json(self, tmp_path):
        dets = [Detection(0, "person", 0.95, 10, 20, 110, 220)]
        r = PredictionResult(detections=dets, elapsed_ms=42.5, model_name="yolov8n",
                             image_shape=(480, 640))
        path = str(tmp_path / "results.json")
        r.save(path)
        import json
        with open(path) as f:
            data = json.load(f)
        assert data["model"] == "yolov8n"
        assert data["elapsed_ms"] == 42.5
        assert len(data["detections"]) == 1
        assert data["detections"][0]["class"] == "person"


class TestModelZoo:

    def test_cache_dir_created(self, tmp_path):
        zoo = ModelZoo(cache_dir=str(tmp_path / "cache"))
        assert zoo.cache_dir.exists()

    def test_unknown_model_raises(self, tmp_path):
        zoo = ModelZoo(cache_dir=str(tmp_path / "cache"))
        with pytest.raises(ValueError, match="Unknown model"):
            zoo.get_model_path("nonexistent_model_xyz")

    def test_uncached_model_returns_none(self, tmp_path):
        zoo = ModelZoo(cache_dir=str(tmp_path / "cache"))
        path = zoo.get_model_path("yolov5n")
        assert path is None

    def test_list_downloaded_empty(self, tmp_path):
        zoo = ModelZoo(cache_dir=str(tmp_path / "cache"))
        assert zoo.list_downloaded() == []

    def test_list_downloaded_with_cached(self, tmp_path):
        zoo = ModelZoo(cache_dir=str(tmp_path / "cache"))
        # Fake a cached model
        model_dir = zoo.cache_dir / "yolov5n"
        model_dir.mkdir(parents=True)
        (model_dir / "yolov5n.onnx").write_bytes(b"fake")
        assert "yolov5n" in zoo.list_downloaded()

    def test_clear_cache(self, tmp_path):
        zoo = ModelZoo(cache_dir=str(tmp_path / "cache"))
        model_dir = zoo.cache_dir / "yolov5n"
        model_dir.mkdir(parents=True)
        (model_dir / "yolov5n.onnx").write_bytes(b"fake")
        zoo.clear_cache()
        assert zoo.list_downloaded() == []
        assert zoo.cache_dir.exists()  # dir recreated


class TestNMS:
    """Non-maximum suppression."""

    def test_no_boxes(self):
        result = Model._nms(
            np.array([]), np.array([]), np.array([]), np.array([]),
            np.array([]), 0.5,
        )
        assert result == []

    def test_single_box(self):
        result = Model._nms(
            np.array([0]), np.array([0]), np.array([100]), np.array([100]),
            np.array([0.9]), 0.5,
        )
        assert result == [0]

    def test_non_overlapping_boxes_kept(self):
        result = Model._nms(
            np.array([0, 200]), np.array([0, 200]),
            np.array([50, 250]), np.array([50, 250]),
            np.array([0.9, 0.8]), 0.5,
        )
        assert len(result) == 2

    def test_overlapping_boxes_suppressed(self):
        result = Model._nms(
            np.array([0, 10]), np.array([0, 10]),
            np.array([100, 110]), np.array([100, 110]),
            np.array([0.9, 0.8]), 0.3,
        )
        assert len(result) == 1
        assert result[0] == 0  # higher confidence kept

    def test_keeps_highest_confidence(self):
        # Three overlapping boxes
        result = Model._nms(
            np.array([0, 5, 10]), np.array([0, 5, 10]),
            np.array([100, 105, 110]), np.array([100, 105, 110]),
            np.array([0.7, 0.9, 0.5]), 0.3,
        )
        assert result[0] == 1  # 0.9 confidence first


class TestPreprocessing:
    """Image preprocessing for model input."""

    def test_output_shape(self, sample_frame):
        model = Model.__new__(Model)
        model._info = {"input_size": (1, 3, 640, 640)}
        blob = model._preprocess(sample_frame, 640, 640)
        assert blob.shape == (1, 3, 640, 640)
        assert blob.dtype == np.float32

    def test_values_normalized(self, sample_frame):
        model = Model.__new__(Model)
        model._info = {"input_size": (1, 3, 640, 640)}
        blob = model._preprocess(sample_frame, 640, 640)
        assert blob.min() >= 0.0
        assert blob.max() <= 1.0

    def test_small_image_upscaled(self, small_frame):
        model = Model.__new__(Model)
        model._info = {"input_size": (1, 3, 640, 640)}
        blob = model._preprocess(small_frame, 640, 640)
        assert blob.shape == (1, 3, 640, 640)

    def test_letterbox_padding(self):
        """Non-square images should be letterboxed with pad value 114."""
        model = Model.__new__(Model)
        model._info = {"input_size": (1, 3, 640, 640)}
        # Wide image: 640x240
        img = np.full((240, 640, 3), 200, dtype=np.uint8)
        blob = model._preprocess(img, 640, 640)
        # Top/bottom should have pad (114/255 ≈ 0.447)
        assert abs(blob[0, 0, 0, 0] - 114/255) < 0.01  # top-left is padding


class TestPostprocessing:
    """YOLO output postprocessing."""

    def test_empty_output(self):
        model = Model.__new__(Model)
        model.conf_threshold = 0.4
        model.iou_threshold = 0.45
        output = np.zeros((1, 84, 8400), dtype=np.float32)
        dets = model._postprocess_yolo([output], (480, 640), 640, 640)
        assert len(dets) == 0

    def test_classification_postprocess(self):
        model = Model.__new__(Model)
        # Softmax-like output with class 5 being highest
        output = np.zeros((1, 1000), dtype=np.float32)
        output[0, 5] = 10.0
        dets = model._postprocess_classification([output])
        assert len(dets) == 5  # top-5
        assert dets[0].class_id == 5
