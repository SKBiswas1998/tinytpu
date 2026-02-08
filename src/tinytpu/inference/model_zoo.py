"""
TinyTPU Model Zoo - Download, manage, and load pre-trained models.

Usage:
    from tinytpu.inference.model_zoo import Model
    model = Model("yolov8n")
    results = model.predict(frame)
"""

import hashlib
import json
import logging
import os
import platform
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("tinytpu.inference.model_zoo")

MODEL_REGISTRY: Dict[str, dict] = {
    "yolov8n": {
        "task": "detection", "format": "onnx", "size": "6.2 MB",
        "url": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt",
        "input_size": (1, 3, 640, 640), "classes": 80,
        "description": "YOLOv8 Nano - fastest, best for Pi",
    },
    "yolov8s": {
        "task": "detection", "format": "onnx", "size": "22 MB",
        "url": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s.pt",
        "input_size": (1, 3, 640, 640), "classes": 80,
        "description": "YOLOv8 Small - balanced speed/accuracy",
    },
    "yolov8m": {
        "task": "detection", "format": "onnx", "size": "52 MB",
        "url": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m.pt",
        "input_size": (1, 3, 640, 640), "classes": 80,
        "description": "YOLOv8 Medium - higher accuracy, needs GPU/NPU",
    },
    "mobilenetv2": {
        "task": "classification", "format": "onnx", "size": "14 MB",
        "url": "https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-12.onnx",
        "input_size": (1, 3, 224, 224), "classes": 1000,
        "description": "MobileNetV2 - lightweight classification",
    },
    "yolov8n-pose": {
        "task": "pose", "format": "onnx", "size": "6.7 MB",
        "url": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n-pose.pt",
        "input_size": (1, 3, 640, 640), "classes": 1,
        "description": "YOLOv8 Nano Pose - human keypoints",
    },
    "yolov8n-seg": {
        "task": "segmentation", "format": "onnx", "size": "6.8 MB",
        "url": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n-seg.pt",
        "input_size": (1, 3, 640, 640), "classes": 80,
        "description": "YOLOv8 Nano Seg - instance segmentation",
    },
}

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]


@dataclass
class Detection:
    """Single detection result."""
    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def center(self):
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def area(self):
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)


@dataclass
class PredictionResult:
    """Result from model.predict()."""
    detections: List[Detection] = field(default_factory=list)
    raw_output: Any = None
    elapsed_ms: float = 0.0
    model_name: str = ""
    image_shape: tuple = ()

    def __len__(self):
        return len(self.detections)

    def filter(self, class_name: str = None, min_confidence: float = 0.0):
        filtered = self.detections
        if class_name:
            filtered = [d for d in filtered if d.class_name == class_name]
        if min_confidence > 0:
            filtered = [d for d in filtered if d.confidence >= min_confidence]
        return filtered

    def save(self, path: str):
        data = {
            "model": self.model_name, "elapsed_ms": self.elapsed_ms,
            "image_shape": list(self.image_shape),
            "detections": [
                {"class": d.class_name, "confidence": round(d.confidence, 3),
                 "box": [round(d.x1), round(d.y1), round(d.x2), round(d.y2)]}
                for d in self.detections
            ],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)


class ModelZoo:
    """Manage model downloads and caching."""

    def __init__(self, cache_dir: str = None):
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            if platform.system() == "Windows":
                base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
            elif platform.system() == "Darwin":
                base = Path.home() / "Library" / "Caches"
            else:
                base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
            self.cache_dir = base / "tinytpu" / "models"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_model_path(self, name: str) -> Optional[Path]:
        if name not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model: {name}. Available: {list(MODEL_REGISTRY.keys())}")
        model_dir = self.cache_dir / name
        onnx_path = model_dir / f"{name}.onnx"
        if onnx_path.exists():
            return onnx_path
        return None

    def download(self, name: str, force: bool = False) -> Path:
        if name not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model: {name}. Available: {list(MODEL_REGISTRY.keys())}")
        model_dir = self.cache_dir / name
        onnx_path = model_dir / f"{name}.onnx"
        if onnx_path.exists() and not force:
            return onnx_path
        model_dir.mkdir(parents=True, exist_ok=True)
        info = MODEL_REGISTRY[name]
        url = info["url"]
        logger.info(f"Downloading {name} ({info['size']})...")
        if url.endswith(".onnx"):
            self._download_file(url, onnx_path)
        elif url.endswith(".pt"):
            self._download_and_export_yolo(name, url, onnx_path, info)
        else:
            self._download_file(url, onnx_path)
        return onnx_path

    def _download_file(self, url: str, dest: Path):
        import urllib.request
        def progress(count, block_size, total_size):
            if total_size > 0:
                pct = count * block_size * 100 / total_size
                print(f"\r  Downloading: {pct:.0f}%", end="", flush=True)
        try:
            urllib.request.urlretrieve(url, str(dest), reporthook=progress)
            print()
        except Exception as e:
            if dest.exists():
                dest.unlink()
            raise RuntimeError(f"Download failed: {e}") from e

    def _download_and_export_yolo(self, name, url, onnx_path, info):
        pt_path = onnx_path.parent / f"{name}.pt"
        self._download_file(url, pt_path)
        try:
            from ultralytics import YOLO
            model = YOLO(str(pt_path))
            model.export(format="onnx", imgsz=info["input_size"][2])
            exported = pt_path.with_suffix(".onnx")
            if exported.exists():
                shutil.move(str(exported), str(onnx_path))
        except ImportError:
            raise RuntimeError(f"Cannot convert {name}.pt to ONNX without ultralytics. Install: pip install ultralytics")
        finally:
            if pt_path.exists():
                pt_path.unlink()

    def list_downloaded(self) -> List[str]:
        return [name for name in MODEL_REGISTRY if self.get_model_path(name)]

    def clear_cache(self):
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)


class Model:
    """
    High-level model interface - the PyCoral replacement.

    Usage:
        model = Model("yolov8n")
        results = model.predict(frame)
        for det in results.detections:
            print(f"{det.class_name}: {det.confidence:.0%}")
    """

    def __init__(self, name: str = "yolov8n", conf_threshold: float = 0.4,
                 iou_threshold: float = 0.45, backend: str = "auto",
                 device: str = "auto", cache_dir: str = None):
        self.name = name
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self._backend_name = backend
        self._device = device

        if name not in MODEL_REGISTRY:
            if os.path.isfile(name):
                self._model_path = Path(name)
                self._info = {"task": "detection", "input_size": (1, 3, 640, 640), "classes": 80}
            else:
                raise ValueError(f"Unknown model: {name}. Available: {list(MODEL_REGISTRY.keys())}")
        else:
            self._info = MODEL_REGISTRY[name]
            zoo = ModelZoo(cache_dir=cache_dir)
            self._model_path = zoo.get_model_path(name)
            if self._model_path is None:
                self._model_path = zoo.download(name)

        self._session = None
        self._load_model()

    def _load_model(self):
        model_path = str(self._model_path)
        try:
            import onnxruntime as ort
            providers = []
            if self._device == "auto":
                avail = ort.get_available_providers()
                if "CUDAExecutionProvider" in avail:
                    providers.append("CUDAExecutionProvider")
                providers.append("CPUExecutionProvider")
            elif self._device == "cuda":
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            else:
                providers = ["CPUExecutionProvider"]
            self._session = ort.InferenceSession(model_path, providers=providers)
            self._backend_name = "onnxruntime"
            return
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"ONNX Runtime failed: {e}")
        try:
            from tinytpu.inference.engine import TinyTPUEngine
            self._session = TinyTPUEngine(model_path)
            self._backend_name = "tinytpu"
            return
        except Exception as e:
            logger.warning(f"TinyTPU engine failed: {e}")
        raise RuntimeError(f"Cannot load {self.name}. Install onnxruntime: pip install tinytpu[inference]")

    def predict(self, image: np.ndarray) -> PredictionResult:
        if image is None or image.size == 0:
            return PredictionResult(model_name=self.name)
        original_shape = image.shape[:2]
        input_size = self._info.get("input_size", (1, 3, 640, 640))
        target_h, target_w = input_size[2], input_size[3]
        preprocessed = self._preprocess(image, target_h, target_w)
        t0 = time.perf_counter()
        if self._backend_name == "onnxruntime":
            input_name = self._session.get_inputs()[0].name
            outputs = self._session.run(None, {input_name: preprocessed})
        elif self._backend_name == "tinytpu":
            input_name = self._session.input_names[0]
            outputs, _ = self._session.run({input_name: preprocessed})
            if not isinstance(outputs, list):
                outputs = [outputs]
        else:
            raise RuntimeError(f"Unknown backend: {self._backend_name}")
        elapsed_ms = (time.perf_counter() - t0) * 1000
        task = self._info.get("task", "detection")
        if task == "detection":
            detections = self._postprocess_yolo(outputs, original_shape, target_h, target_w)
        elif task == "classification":
            detections = self._postprocess_classification(outputs)
        else:
            detections = []
        return PredictionResult(
            detections=detections, raw_output=outputs, elapsed_ms=elapsed_ms,
            model_name=self.name, image_shape=original_shape,
        )

    def _preprocess(self, image: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
        h, w = image.shape[:2]
        scale = min(target_h / h, target_w / w)
        new_h, new_w = int(h * scale), int(w * scale)
        try:
            import cv2
            resized = cv2.resize(image, (new_w, new_h))
        except ImportError:
            from PIL import Image
            pil_img = Image.fromarray(image)
            pil_img = pil_img.resize((new_w, new_h), Image.BILINEAR)
            resized = np.array(pil_img)
        padded = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        pad_h = (target_h - new_h) // 2
        pad_w = (target_w - new_w) // 2
        padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized
        blob = padded.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))
        blob = np.expand_dims(blob, 0)
        return blob

    def _postprocess_yolo(self, outputs, original_shape, target_h, target_w) -> List[Detection]:
        output = outputs[0]
        if output.ndim == 3:
            output = output[0]
        if output.shape[0] < output.shape[1]:
            output = output.T
        boxes = output[:, :4]
        scores = output[:, 4:]
        class_ids = np.argmax(scores, axis=1)
        confidences = scores[np.arange(len(scores)), class_ids]
        mask = confidences >= self.conf_threshold
        boxes = boxes[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]
        if len(boxes) == 0:
            return []
        x1 = boxes[:, 0] - boxes[:, 2] / 2
        y1 = boxes[:, 1] - boxes[:, 3] / 2
        x2 = boxes[:, 0] + boxes[:, 2] / 2
        y2 = boxes[:, 1] + boxes[:, 3] / 2
        indices = self._nms(x1, y1, x2, y2, confidences, self.iou_threshold)
        h_orig, w_orig = original_shape
        scale = min(target_h / h_orig, target_w / w_orig)
        pad_h = (target_h - h_orig * scale) / 2
        pad_w = (target_w - w_orig * scale) / 2
        detections = []
        for i in indices:
            det = Detection(
                class_id=int(class_ids[i]),
                class_name=COCO_CLASSES[int(class_ids[i])] if int(class_ids[i]) < len(COCO_CLASSES) else f"class_{class_ids[i]}",
                confidence=float(confidences[i]),
                x1=float((x1[i] - pad_w) / scale), y1=float((y1[i] - pad_h) / scale),
                x2=float((x2[i] - pad_w) / scale), y2=float((y2[i] - pad_h) / scale),
            )
            detections.append(det)
        return sorted(detections, key=lambda d: d.confidence, reverse=True)

    def _postprocess_classification(self, outputs) -> List[Detection]:
        output = outputs[0]
        if output.ndim > 1:
            output = output.flatten()
        exp_out = np.exp(output - output.max())
        probs = exp_out / exp_out.sum()
        top_indices = np.argsort(probs)[::-1][:5]
        return [Detection(int(idx), f"class_{idx}", float(probs[idx]), 0, 0, 0, 0) for idx in top_indices]

    @staticmethod
    def _nms(x1, y1, x2, y2, scores, iou_threshold: float) -> List[int]:
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep = []
        while len(order) > 0:
            i = order[0]
            keep.append(i)
            if len(order) == 1:
                break
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]
        return keep
