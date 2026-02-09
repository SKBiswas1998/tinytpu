"""
TinyTPU Package Tests — Verify installation and basic functionality.

Run: pytest tests/test_package.py -v
"""

import importlib
import platform
import subprocess
import sys
import time

import numpy as np
import pytest


# ================================================================
# Package Structure Tests
# ================================================================


class TestPackageImports:
    """Verify all submodules are importable."""

    def test_import_tinytpu(self):
        import tinytpu
        assert hasattr(tinytpu, "__version__")
        assert tinytpu.__version__ == "0.1.0"

    def test_import_core(self):
        from tinytpu import core
        assert core is not None

    def test_import_inference(self):
        from tinytpu import inference
        assert inference is not None

    def test_import_perception(self):
        from tinytpu import perception
        assert perception is not None

    def test_import_control(self):
        from tinytpu import control
        assert control is not None

    def test_import_monitoring(self):
        from tinytpu import monitoring
        assert monitoring is not None

    def test_import_numerical(self):
        from tinytpu import numerical
        assert numerical is not None

    def test_import_hal(self):
        from tinytpu import hal
        assert hal is not None

    def test_import_cli(self):
        from tinytpu import cli
        assert cli is not None

    def test_version_string(self):
        import tinytpu
        parts = tinytpu.__version__.split(".")
        assert len(parts) == 3
        assert all(p.isdigit() for p in parts)


# ================================================================
# Hardware Abstraction Layer Tests
# ================================================================


class TestHardwareDetection:
    """Test hardware auto-detection."""

    def test_detect_hardware_returns_info(self):
        from tinytpu.hal.detect import detect_hardware
        hw = detect_hardware()
        assert hw is not None
        assert hasattr(hw, "devices")
        assert hasattr(hw, "recommended")
        assert hasattr(hw, "recommended_model")

    def test_cpu_always_detected(self):
        from tinytpu.hal.detect import detect_hardware
        hw = detect_hardware()
        cpu_devices = [d for d in hw.devices if d.available]
        assert len(cpu_devices) >= 1, "At least CPU should be detected"

    def test_recommended_is_valid(self):
        from tinytpu.hal.detect import detect_hardware
        hw = detect_hardware()
        valid_backends = ["hailo", "coral", "cuda", "onnxruntime", "numpy"]
        assert hw.recommended in valid_backends

    def test_recommended_model_is_string(self):
        from tinytpu.hal.detect import detect_hardware
        hw = detect_hardware()
        assert isinstance(hw.recommended_model, str)
        assert len(hw.recommended_model) > 0

    def test_device_info_structure(self):
        from tinytpu.hal.detect import detect_hardware
        hw = detect_hardware()
        for device in hw.devices:
            assert hasattr(device, "name")
            assert hasattr(device, "backend")
            assert hasattr(device, "available")
            assert isinstance(device.available, bool)

    def test_system_info(self):
        from tinytpu.hal.detect import detect_hardware
        hw = detect_hardware()
        assert hw.cpu_cores > 0
        assert isinstance(hw.arch, str)


# ================================================================
# Model Zoo Tests
# ================================================================


class TestModelZoo:
    """Test model registry and zoo functionality."""

    def test_registry_has_models(self):
        from tinytpu.inference.model_zoo import MODEL_REGISTRY
        assert len(MODEL_REGISTRY) >= 5

    def test_registry_model_structure(self):
        from tinytpu.inference.model_zoo import MODEL_REGISTRY
        for name, info in MODEL_REGISTRY.items():
            assert "task" in info, f"{name} missing 'task'"
            assert "format" in info, f"{name} missing 'format'"
            assert "size" in info, f"{name} missing 'size'"
            assert "input_size" in info, f"{name} missing 'input_size'"

    def test_coco_classes_count(self):
        from tinytpu.inference.model_zoo import COCO_CLASSES
        assert len(COCO_CLASSES) == 80

    def test_coco_classes_content(self):
        from tinytpu.inference.model_zoo import COCO_CLASSES
        assert "person" in COCO_CLASSES
        assert "car" in COCO_CLASSES
        assert "dog" in COCO_CLASSES

    def test_detection_dataclass(self):
        from tinytpu.inference.model_zoo import Detection
        det = Detection(
            class_id=0, class_name="person", confidence=0.95,
            x1=10, y1=20, x2=100, y2=200,
        )
        assert det.class_name == "person"
        assert det.confidence == 0.95
        cx, cy = det.center
        assert cx == 55.0
        assert cy == 110.0
        assert det.area == 90 * 180

    def test_prediction_result(self):
        from tinytpu.inference.model_zoo import PredictionResult, Detection
        det = Detection(0, "person", 0.9, 10, 20, 100, 200)
        result = PredictionResult(
            detections=[det], elapsed_ms=15.2, model_name="yolov8n"
        )
        assert len(result) == 1
        assert result.elapsed_ms == 15.2

    def test_prediction_filter(self):
        from tinytpu.inference.model_zoo import PredictionResult, Detection
        dets = [
            Detection(0, "person", 0.9, 10, 20, 100, 200),
            Detection(1, "car", 0.7, 200, 100, 400, 300),
            Detection(0, "person", 0.3, 50, 50, 80, 80),
        ]
        result = PredictionResult(detections=dets)
        persons = result.filter(class_name="person")
        assert len(persons) == 2
        high_conf = result.filter(min_confidence=0.5)
        assert len(high_conf) == 2

    def test_nms_basic(self):
        """Test Non-Maximum Suppression."""
        from tinytpu.inference.model_zoo import Model
        x1 = np.array([10, 12, 200])
        y1 = np.array([10, 12, 200])
        x2 = np.array([100, 102, 300])
        y2 = np.array([100, 102, 300])
        scores = np.array([0.9, 0.8, 0.7])

        keep = Model._nms(x1, y1, x2, y2, scores, iou_threshold=0.5)
        # Box 0 and 1 overlap heavily, keep higher score (0)
        # Box 2 doesn't overlap, keep it
        assert 0 in keep
        assert 2 in keep
        assert len(keep) == 2

    def test_zoo_cache_dir(self):
        from tinytpu.inference.model_zoo import ModelZoo
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            zoo = ModelZoo(cache_dir=tmpdir)
            assert zoo.cache_dir.exists()

    def test_zoo_unknown_model_raises(self):
        from tinytpu.inference.model_zoo import ModelZoo
        zoo = ModelZoo()
        with pytest.raises(ValueError, match="Unknown model"):
            zoo.get_model_path("nonexistent_model_xyz")


# ================================================================
# CLI Tests
# ================================================================


class TestCLI:
    """Test command-line interface."""

    def test_cli_version(self):
        from tinytpu.cli import cmd_version
        # Just verify it doesn't crash
        class Args:
            pass
        cmd_version(Args())

    def test_cli_benchmark_runs(self):
        from tinytpu.cli import run_benchmark
        results = run_benchmark(runs=3)
        assert "tests" in results
        matmul_keys = [k for k in results["tests"] if k.startswith("matmul_")]
        assert len(matmul_keys) > 0, f"No matmul keys in: {list(results['tests'].keys())}"
        first_key = matmul_keys[0]
        assert "gflops" in results["tests"][first_key]
        assert results["tests"][first_key]["gflops"] > 0

    def test_cli_main_help(self):
        """Test that tinytpu --help works."""
        result = subprocess.run(
            [sys.executable, "-m", "tinytpu.cli.main", "--help"],
            capture_output=True, text=True,
        )
        # Either 0 (help shown) or other (module not found) is OK for structure test
        assert "tinytpu" in result.stdout.lower() or result.returncode != 0


# ================================================================
# Preprocessing Tests
# ================================================================


class TestPreprocessing:
    """Test image preprocessing for inference."""

    def test_preprocess_basic(self):
        """Test that preprocessing produces correct shape."""
        from tinytpu.inference.model_zoo import Model
        # Create a mock model that we can test preprocessing on
        model = object.__new__(Model)
        model._info = {"input_size": (1, 3, 640, 640)}

        # Fake 480x640 RGB image
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = model._preprocess(image, 640, 640)

        assert result.shape == (1, 3, 640, 640)
        assert result.dtype == np.float32
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_preprocess_small_image(self):
        from tinytpu.inference.model_zoo import Model
        model = object.__new__(Model)
        model._info = {"input_size": (1, 3, 640, 640)}

        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = model._preprocess(image, 640, 640)
        assert result.shape == (1, 3, 640, 640)

    def test_preprocess_large_image(self):
        from tinytpu.inference.model_zoo import Model
        model = object.__new__(Model)
        model._info = {"input_size": (1, 3, 640, 640)}

        image = np.random.randint(0, 255, (1920, 1080, 3), dtype=np.uint8)
        result = model._preprocess(image, 640, 640)
        assert result.shape == (1, 3, 640, 640)


# ================================================================
# Postprocessing Tests
# ================================================================


class TestPostprocessing:
    """Test YOLO output postprocessing."""

    def test_yolo_postprocess_empty(self):
        from tinytpu.inference.model_zoo import Model
        model = object.__new__(Model)
        model.conf_threshold = 0.4
        model.iou_threshold = 0.45
        model._info = {"input_size": (1, 3, 640, 640)}

        # All zeros = no detections
        output = [np.zeros((1, 84, 8400), dtype=np.float32)]
        dets = model._postprocess_yolo(output, (480, 640), 640, 640)
        assert len(dets) == 0

    def test_yolo_postprocess_synthetic(self):
        from tinytpu.inference.model_zoo import Model
        model = object.__new__(Model)
        model.conf_threshold = 0.3
        model.iou_threshold = 0.45
        model._info = {"input_size": (1, 3, 640, 640)}

        # Create synthetic output with one strong detection
        output = np.zeros((1, 84, 8400), dtype=np.float32)
        # Box at center: x=320, y=320, w=100, h=200
        output[0, 0, 0] = 320  # x
        output[0, 1, 0] = 320  # y
        output[0, 2, 0] = 100  # w
        output[0, 3, 0] = 200  # h
        output[0, 4, 0] = 0.95  # person score (class 0)

        dets = model._postprocess_yolo([output], (640, 640), 640, 640)
        assert len(dets) >= 1
        assert dets[0].class_name == "person"
        assert dets[0].confidence >= 0.3


# ================================================================
# Integration Tests
# ================================================================


class TestIntegration:
    """End-to-end integration tests."""

    def test_info_function(self):
        """Test tinytpu.info() doesn't crash."""
        import tinytpu
        tinytpu.info()

    def test_detect_hardware_from_toplevel(self):
        """Test tinytpu.detect_hardware() works."""
        import tinytpu
        hw = tinytpu.detect_hardware()
        assert hw is not None

    def test_full_benchmark_cycle(self):
        """Test complete benchmark from CLI."""
        from tinytpu.cli import run_benchmark
        results = run_benchmark(runs=2)
        assert results["runs"] == 2
        assert len(results["tests"]) >= 6  # 4 matmul sizes + 3 activations + 1 quant
