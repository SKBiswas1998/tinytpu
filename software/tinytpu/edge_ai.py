import numpy as np
import time
import os
import sys
import platform
import subprocess
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path

# ============================================================
# HARDWARE PROFILER
# ============================================================

@dataclass
class HardwareProfile:
    device_name: str = 'unknown'
    cpu_cores: int = 1
    cpu_freq_mhz: int = 0
    ram_total_mb: int = 0
    ram_available_mb: int = 0
    has_gpu: bool = False
    gpu_name: str = ''
    architecture: str = ''
    is_raspberry_pi: bool = False
    pi_model: str = ''
    has_neon: bool = False
    max_model_mb: int = 0
    recommended_quantization: str = 'fp32'
    recommended_batch_size: int = 1
    recommended_image_size: int = 640

    def compute_recommendations(self):
        usable_mb = self.ram_available_mb * 0.5
        self.max_model_mb = int(usable_mb * 0.7)
        if self.ram_total_mb < 512:
            self.recommended_quantization = 'int4'
            self.recommended_image_size = 160
        elif self.ram_total_mb < 2048:
            self.recommended_quantization = 'int8'
            self.recommended_image_size = 320
        elif self.ram_total_mb < 4096:
            self.recommended_quantization = 'int8'
            self.recommended_image_size = 480
        elif self.ram_total_mb < 8192:
            self.recommended_quantization = 'int8'
            self.recommended_image_size = 640
        else:
            self.recommended_quantization = 'fp32'
            self.recommended_image_size = 640

    def __str__(self):
        lines = [
            f'Device: {self.device_name}',
            f'CPU: {self.cpu_cores} cores @ {self.cpu_freq_mhz}MHz ({self.architecture})',
            f'RAM: {self.ram_total_mb}MB total, {self.ram_available_mb}MB available',
        ]
        if self.has_gpu:
            lines.append(f'GPU: {self.gpu_name}')
        if self.is_raspberry_pi:
            lines.append(f'Pi Model: {self.pi_model}')
        lines.extend([
            f'Max model size: {self.max_model_mb}MB',
            f'Recommended: {self.recommended_quantization}, {self.recommended_image_size}px',
        ])
        return chr(10).join(lines)


def detect_hardware() -> HardwareProfile:
    profile = HardwareProfile()
    profile.architecture = platform.machine()
    try:
        profile.cpu_cores = os.cpu_count() or 1
    except:
        profile.cpu_cores = 1

    try:
        if sys.platform == 'linux':
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    if line.startswith('MemTotal'):
                        profile.ram_total_mb = int(line.split()[1]) // 1024
                    elif line.startswith('MemAvailable'):
                        profile.ram_available_mb = int(line.split()[1]) // 1024
        elif sys.platform == 'win32':
            import ctypes
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong),
                    ('ullTotalPhys', ctypes.c_ulonglong), ('ullAvailPhys', ctypes.c_ulonglong),
                    ('ullTotalPageFile', ctypes.c_ulonglong), ('ullAvailPageFile', ctypes.c_ulonglong),
                    ('ullTotalVirtual', ctypes.c_ulonglong), ('ullAvailVirtual', ctypes.c_ulonglong),
                    ('ullAvailExtendedVirtual', ctypes.c_ulonglong),
                ]
            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(stat)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            profile.ram_total_mb = stat.ullTotalPhys // (1024 * 1024)
            profile.ram_available_mb = stat.ullAvailPhys // (1024 * 1024)
        elif sys.platform == 'darwin':
            result = subprocess.run(['sysctl', '-n', 'hw.memsize'], capture_output=True, text=True)
            profile.ram_total_mb = int(result.stdout.strip()) // (1024 * 1024)
            profile.ram_available_mb = profile.ram_total_mb // 2
    except:
        profile.ram_total_mb = 4096
        profile.ram_available_mb = 2048

    try:
        if sys.platform == 'win32':
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'HARDWARE\DESCRIPTION\System\CentralProcessor\0')
            profile.cpu_freq_mhz = winreg.QueryValueEx(key, '~MHz')[0]
        elif sys.platform == 'linux':
            with open('/proc/cpuinfo', 'r') as f:
                for line in f:
                    if 'cpu MHz' in line or 'BogoMIPS' in line:
                        profile.cpu_freq_mhz = int(float(line.split(':')[1].strip()))
                        break
    except:
        profile.cpu_freq_mhz = 1000

    try:
        import torch
        if torch.cuda.is_available():
            profile.has_gpu = True
            profile.gpu_name = torch.cuda.get_device_name(0)
    except:
        pass

    try:
        if sys.platform == 'linux':
            with open('/proc/device-tree/model', 'r') as f:
                model = f.read().strip(chr(0)).strip()
                if 'Raspberry Pi' in model:
                    profile.is_raspberry_pi = True
                    profile.pi_model = model
    except:
        pass

    if profile.architecture in ('aarch64', 'armv7l'):
        profile.has_neon = True

    if profile.is_raspberry_pi:
        profile.device_name = profile.pi_model
    elif profile.has_gpu:
        profile.device_name = f'GPU: {profile.gpu_name}'
    else:
        profile.device_name = f'{platform.node()} ({profile.architecture})'

    profile.compute_recommendations()
    return profile


# ============================================================
# MODEL ZOO
# ============================================================

@dataclass
class ModelSpec:
    name: str
    task: str
    url: str
    size_mb: float
    input_size: int
    input_name: str
    min_ram_mb: int
    classes: int = 80
    description: str = ''
    int8_size_mb: float = 0
    expected_fps_pi5: float = 0
    expected_fps_pi4: float = 0
    expected_fps_desktop: float = 0

MODEL_ZOO = {
    'yolov5n': ModelSpec(
        name='yolov5n', task='detect',
        url='https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.onnx',
        size_mb=3.6, input_size=640, input_name='images', min_ram_mb=256, classes=80,
        description='YOLOv5 Nano - Fastest detection',
        int8_size_mb=0.9, expected_fps_pi5=2, expected_fps_pi4=0.5, expected_fps_desktop=3.1,
    ),
    'yolov5s': ModelSpec(
        name='yolov5s', task='detect',
        url='https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.onnx',
        size_mb=14.1, input_size=640, input_name='images', min_ram_mb=512, classes=80,
        description='YOLOv5 Small - Better accuracy',
        int8_size_mb=3.5, expected_fps_pi5=0.8, expected_fps_pi4=0.2, expected_fps_desktop=1.5,
    ),
    'mobilenetv2': ModelSpec(
        name='mobilenetv2', task='classify',
        url='https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-12.onnx',
        size_mb=13.3, input_size=224, input_name='input', min_ram_mb=128, classes=1000,
        description='MobileNetV2 - Lightweight classification',
        int8_size_mb=3.4, expected_fps_pi5=5, expected_fps_pi4=2, expected_fps_desktop=13.3,
    ),
}

def recommend_model(hardware, task='detect'):
    candidates = []
    for name, spec in MODEL_ZOO.items():
        if spec.task != task:
            continue
        model_mb = spec.int8_size_mb if hardware.recommended_quantization == 'int8' else spec.size_mb
        if model_mb > hardware.max_model_mb:
            continue
        if spec.min_ram_mb > hardware.ram_available_mb:
            continue
        if not spec.url:
            continue
        candidates.append(spec)
    if not candidates:
        return None
    return max(candidates, key=lambda s: s.size_mb)


# ============================================================
# INFERENCE BACKEND
# ============================================================

class InferenceBackend:
    def __init__(self, model_path, quantize=False):
        self.model_path = model_path
        self.quantize = quantize
        self.backend_name = 'none'
        self._session = None
        self._engine = None
        self._input_name = None
        self._input_dtype = None
        self._load()

    def _load(self):
        try:
            import onnxruntime as ort
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            opts.intra_op_num_threads = os.cpu_count() or 1
            opts.inter_op_num_threads = 1
            opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            opts.enable_cpu_mem_arena = True
            self._session = ort.InferenceSession(self.model_path, opts)
            inp = self._session.get_inputs()[0]
            self._input_name = inp.name
            self._input_dtype = np.float16 if 'float16' in inp.type else np.float32
            self.backend_name = 'onnxruntime'
            return
        except ImportError:
            pass
        except Exception as e:
            print(f'[Backend] ONNX Runtime failed: {e}')

        try:
            from tinytpu.onnx_engine import TinyTPUEngine
            self._engine = TinyTPUEngine(self.model_path, quantize=self.quantize)
            self._input_name = list(self._engine._input_shapes.keys())[0]
            self._input_dtype = np.float32
            self.backend_name = 'tinytpu'
            return
        except Exception as e:
            print(f'[Backend] TinyTPU failed: {e}')

        raise RuntimeError(f'No backend available for {self.model_path}')

    def run(self, input_data):
        if input_data.dtype != self._input_dtype:
            input_data = input_data.astype(self._input_dtype)
        start = time.perf_counter()
        if self.backend_name == 'onnxruntime':
            outputs = self._session.run(None, {self._input_name: input_data})
            result = outputs[0]
            if result.dtype == np.float16:
                result = result.astype(np.float32)
        elif self.backend_name == 'tinytpu':
            output_dict, _ = self._engine.run({self._input_name: input_data})
            result = list(output_dict.values())[0]
        else:
            raise RuntimeError('No backend')
        elapsed = time.perf_counter() - start
        return result, elapsed

    def __repr__(self):
        return f'InferenceBackend({self.backend_name}, {self.model_path})'


# ============================================================
# DETECTION PIPELINE
# ============================================================

COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def cx(self): return (self.x1 + self.x2) / 2
    @property
    def cy(self): return (self.y1 + self.y2) / 2
    @property
    def width(self): return self.x2 - self.x1
    @property
    def height(self): return self.y2 - self.y1
    @property
    def area(self): return self.width * self.height

    def to_dict(self):
        return {
            'class_id': self.class_id, 'class_name': self.class_name,
            'confidence': round(self.confidence, 3),
            'bbox': [round(self.x1, 1), round(self.y1, 1), round(self.x2, 1), round(self.y2, 1)],
        }


def _nms(boxes, scores, iou_thresh=0.45):
    if len(boxes) == 0:
        return np.array([], dtype=int)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_j = (boxes[order[1:], 2] - boxes[order[1:], 0]) * (boxes[order[1:], 3] - boxes[order[1:], 1])
        iou = inter / (area_i + area_j - inter + 1e-6)
        order = order[np.where(iou <= iou_thresh)[0] + 1]
    return np.array(keep)


class ObjectDetector:
    def __init__(self, model_path, conf_thresh=0.4, nms_thresh=0.45,
                 img_size=640, quantize=False, classes=None):
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.img_size = img_size
        self.classes = classes or COCO_CLASSES
        self.backend = InferenceBackend(model_path, quantize)
        self._resize_fn = None
        try:
            import cv2
            self._resize_fn = lambda img, size: cv2.resize(img, (size, size))
        except ImportError:
            try:
                from PIL import Image as PILImage
                self._resize_fn = lambda img, size: np.array(PILImage.fromarray(img).resize((size, size)))
            except ImportError:
                self._resize_fn = self._numpy_resize
        self._frame_count = 0
        self._total_time = 0

    @classmethod
    def auto(cls, task='detect', conf_thresh=0.4, model_path=None):
        hw = detect_hardware()
        print(f'[EdgeAI] Hardware detected:\n{hw}\n')
        if model_path and os.path.exists(model_path):
            quantize = hw.recommended_quantization == 'int8'
            img_size = hw.recommended_image_size
        else:
            spec = recommend_model(hw, task)
            if spec is None:
                raise RuntimeError(f'No suitable model for {hw.device_name}')
            model_path = f'{spec.name}.onnx'
            if not os.path.exists(model_path):
                print(f'[EdgeAI] Downloading {spec.name} ({spec.size_mb}MB)...')
                import urllib.request
                urllib.request.urlretrieve(spec.url, model_path)
                print(f'[EdgeAI] Downloaded: {model_path}')
            quantize = hw.recommended_quantization == 'int8'
            img_size = min(hw.recommended_image_size, spec.input_size)
        print(f'[EdgeAI] Config: model={model_path}, size={img_size}, quantize={quantize}')
        return cls(model_path, conf_thresh=conf_thresh, img_size=img_size, quantize=quantize)

    @staticmethod
    def _numpy_resize(img, size):
        h, w = img.shape[:2]
        y_idx = (np.arange(size) * h / size).astype(int)
        x_idx = (np.arange(size) * w / size).astype(int)
        return img[np.ix_(y_idx, x_idx)]

    def preprocess(self, image):
        orig_h, orig_w = image.shape[:2]
        resized = self._resize_fn(image, self.img_size)
        x = resized.astype(np.float32) / 255.0
        x = x.transpose(2, 0, 1)[np.newaxis]
        return x, (orig_h, orig_w)

    def detect(self, image, conf_thresh=None):
        conf = conf_thresh or self.conf_thresh
        x, (orig_h, orig_w) = self.preprocess(image)
        output, elapsed = self.backend.run(x)
        self._frame_count += 1
        self._total_time += elapsed
        return self._parse_yolo(output, orig_h, orig_w, conf)

    def _parse_yolo(self, output, orig_h, orig_w, conf_thresh):
        if output.ndim == 3:
            output = output[0]
        obj_mask = output[:, 4] > conf_thresh
        filtered = output[obj_mask]
        if len(filtered) == 0:
            return []
        cx, cy, w, h = filtered[:, 0], filtered[:, 1], filtered[:, 2], filtered[:, 3]
        boxes = np.stack([cx - w/2, cy - h/2, cx + w/2, cy + h/2], axis=1)
        class_scores = filtered[:, 5:]
        class_ids = np.argmax(class_scores, axis=1)
        class_probs = np.max(class_scores, axis=1)
        confidences = filtered[:, 4] * class_probs
        scale_x = orig_w / self.img_size
        scale_y = orig_h / self.img_size
        detections = []
        for cls_id in np.unique(class_ids):
            cls_mask = class_ids == cls_id
            cls_boxes = boxes[cls_mask]
            cls_scores_arr = confidences[cls_mask]
            keep = _nms(cls_boxes, cls_scores_arr, self.nms_thresh)
            for k in keep:
                b = cls_boxes[k]
                detections.append(Detection(
                    class_id=int(cls_id), class_name=self.classes[int(cls_id)] if int(cls_id) < len(self.classes) else f'class_{cls_id}',
                    confidence=float(cls_scores_arr[k]),
                    x1=float(b[0]*scale_x), y1=float(b[1]*scale_y),
                    x2=float(b[2]*scale_x), y2=float(b[3]*scale_y),
                ))
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections

    @property
    def fps(self):
        return self._frame_count / self._total_time if self._total_time > 0 else 0
    @property
    def avg_ms(self):
        return (self._total_time / self._frame_count) * 1000 if self._frame_count > 0 else 0


# ============================================================
# MODEL ZOO
# ============================================================

@dataclass
class ModelSpec:
    name: str
    task: str
    url: str
    size_mb: float
    input_size: int
    input_name: str
    min_ram_mb: int
    classes: int = 80
    description: str = ''
    int8_size_mb: float = 0
    expected_fps_pi5: float = 0
    expected_fps_pi4: float = 0
    expected_fps_desktop: float = 0

MODEL_ZOO = {
    'yolov5n': ModelSpec(
        name='yolov5n', task='detect',
        url='https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.onnx',
        size_mb=3.6, input_size=640, input_name='images', min_ram_mb=256, classes=80,
        description='YOLOv5 Nano - Fastest detection',
        int8_size_mb=0.9, expected_fps_pi5=2, expected_fps_pi4=0.5, expected_fps_desktop=3.1,
    ),
    'yolov5s': ModelSpec(
        name='yolov5s', task='detect',
        url='https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.onnx',
        size_mb=14.1, input_size=640, input_name='images', min_ram_mb=512, classes=80,
        description='YOLOv5 Small - Better accuracy',
        int8_size_mb=3.5, expected_fps_pi5=0.8, expected_fps_pi4=0.2, expected_fps_desktop=1.5,
    ),
    'mobilenetv2': ModelSpec(
        name='mobilenetv2', task='classify',
        url='https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-12.onnx',
        size_mb=13.3, input_size=224, input_name='input', min_ram_mb=128, classes=1000,
        description='MobileNetV2 - Lightweight classification',
        int8_size_mb=3.4, expected_fps_pi5=5, expected_fps_pi4=2, expected_fps_desktop=13.3,
    ),
}

def recommend_model(hardware, task='detect'):
    candidates = []
    for name, spec in MODEL_ZOO.items():
        if spec.task != task:
            continue
        model_mb = spec.int8_size_mb if hardware.recommended_quantization == 'int8' else spec.size_mb
        if model_mb > hardware.max_model_mb:
            continue
        if spec.min_ram_mb > hardware.ram_available_mb:
            continue
        if not spec.url:
            continue
        candidates.append(spec)
    if not candidates:
        return None
    return max(candidates, key=lambda s: s.size_mb)


# ============================================================
# INFERENCE BACKEND
# ============================================================

class InferenceBackend:
    def __init__(self, model_path, quantize=False):
        self.model_path = model_path
        self.quantize = quantize
        self.backend_name = 'none'
        self._session = None
        self._engine = None
        self._input_name = None
        self._input_dtype = None
        self._load()

    def _load(self):
        try:
            import onnxruntime as ort
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            opts.intra_op_num_threads = os.cpu_count() or 1
            opts.inter_op_num_threads = 1
            opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            opts.enable_cpu_mem_arena = True
            self._session = ort.InferenceSession(self.model_path, opts)
            inp = self._session.get_inputs()[0]
            self._input_name = inp.name
            self._input_dtype = np.float16 if 'float16' in inp.type else np.float32
            self.backend_name = 'onnxruntime'
            return
        except ImportError:
            pass
        except Exception as e:
            print(f'[Backend] ONNX Runtime failed: {e}')

        try:
            from tinytpu.onnx_engine import TinyTPUEngine
            self._engine = TinyTPUEngine(self.model_path, quantize=self.quantize)
            self._input_name = list(self._engine._input_shapes.keys())[0]
            self._input_dtype = np.float32
            self.backend_name = 'tinytpu'
            return
        except Exception as e:
            print(f'[Backend] TinyTPU failed: {e}')

        raise RuntimeError(f'No backend available for {self.model_path}')

    def run(self, input_data):
        if input_data.dtype != self._input_dtype:
            input_data = input_data.astype(self._input_dtype)
        start = time.perf_counter()
        if self.backend_name == 'onnxruntime':
            outputs = self._session.run(None, {self._input_name: input_data})
            result = outputs[0]
            if result.dtype == np.float16:
                result = result.astype(np.float32)
        elif self.backend_name == 'tinytpu':
            output_dict, _ = self._engine.run({self._input_name: input_data})
            result = list(output_dict.values())[0]
        else:
            raise RuntimeError('No backend')
        elapsed = time.perf_counter() - start
        return result, elapsed

    def __repr__(self):
        return f'InferenceBackend({self.backend_name}, {self.model_path})'

# ============================================================
# MODEL ZOO
# ============================================================

@dataclass
class ModelSpec:
    name: str
    task: str
    url: str
    size_mb: float
    input_size: int
    input_name: str
    min_ram_mb: int
    classes: int = 80
    description: str = ''
    int8_size_mb: float = 0
    expected_fps_pi5: float = 0
    expected_fps_pi4: float = 0
    expected_fps_desktop: float = 0

MODEL_ZOO = {
    'yolov5n': ModelSpec(
        name='yolov5n', task='detect',
        url='https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.onnx',
        size_mb=3.6, input_size=640, input_name='images', min_ram_mb=256, classes=80,
        description='YOLOv5 Nano - Fastest detection',
        int8_size_mb=0.9, expected_fps_pi5=2, expected_fps_pi4=0.5, expected_fps_desktop=3.1,
    ),
    'yolov5s': ModelSpec(
        name='yolov5s', task='detect',
        url='https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.onnx',
        size_mb=14.1, input_size=640, input_name='images', min_ram_mb=512, classes=80,
        description='YOLOv5 Small - Better accuracy',
        int8_size_mb=3.5, expected_fps_pi5=0.8, expected_fps_pi4=0.2, expected_fps_desktop=1.5,
    ),
    'mobilenetv2': ModelSpec(
        name='mobilenetv2', task='classify',
        url='https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-12.onnx',
        size_mb=13.3, input_size=224, input_name='input', min_ram_mb=128, classes=1000,
        description='MobileNetV2 - Lightweight classification',
        int8_size_mb=3.4, expected_fps_pi5=5, expected_fps_pi4=2, expected_fps_desktop=13.3,
    ),
}

def recommend_model(hardware, task='detect'):
    candidates = []
    for name, spec in MODEL_ZOO.items():
        if spec.task != task: continue
        model_mb = spec.int8_size_mb if hardware.recommended_quantization == 'int8' else spec.size_mb
        if model_mb > hardware.max_model_mb: continue
        if spec.min_ram_mb > hardware.ram_available_mb: continue
        if not spec.url: continue
        candidates.append(spec)
    if not candidates: return None
    return max(candidates, key=lambda s: s.size_mb)


# ============================================================
# INFERENCE BACKEND
# ============================================================

class InferenceBackend:
    def __init__(self, model_path, quantize=False):
        self.model_path = model_path
        self.quantize = quantize
        self.backend_name = 'none'
        self._session = None
        self._engine = None
        self._input_name = None
        self._input_dtype = None
        self._load()

    def _load(self):
        try:
            import onnxruntime as ort
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            opts.intra_op_num_threads = os.cpu_count() or 1
            opts.inter_op_num_threads = 1
            opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            opts.enable_cpu_mem_arena = True
            self._session = ort.InferenceSession(self.model_path, opts)
            inp = self._session.get_inputs()[0]
            self._input_name = inp.name
            self._input_dtype = np.float16 if 'float16' in inp.type else np.float32
            self.backend_name = 'onnxruntime'
            return
        except ImportError:
            pass
        except Exception as e:
            print(f'[Backend] ONNX Runtime failed: {e}')

        try:
            from tinytpu.onnx_engine import TinyTPUEngine
            self._engine = TinyTPUEngine(self.model_path, quantize=self.quantize)
            self._input_name = list(self._engine._input_shapes.keys())[0]
            self._input_dtype = np.float32
            self.backend_name = 'tinytpu'
            return
        except Exception as e:
            print(f'[Backend] TinyTPU failed: {e}')

        raise RuntimeError(f'No backend available for {self.model_path}')

    def run(self, input_data):
        if input_data.dtype != self._input_dtype:
            input_data = input_data.astype(self._input_dtype)
        start = time.perf_counter()
        if self.backend_name == 'onnxruntime':
            outputs = self._session.run(None, {self._input_name: input_data})
            result = outputs[0]
            if result.dtype == np.float16:
                result = result.astype(np.float32)
        elif self.backend_name == 'tinytpu':
            output_dict, _ = self._engine.run({self._input_name: input_data})
            result = list(output_dict.values())[0]
        else:
            raise RuntimeError('No backend')
        elapsed = time.perf_counter() - start
        return result, elapsed


# ============================================================
# DETECTION PIPELINE
# ============================================================

COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def cx(self): return (self.x1 + self.x2) / 2
    @property
    def cy(self): return (self.y1 + self.y2) / 2
    @property
    def width(self): return self.x2 - self.x1
    @property
    def height(self): return self.y2 - self.y1
    @property
    def area(self): return self.width * self.height

    def to_dict(self):
        return {'class_id': self.class_id, 'class_name': self.class_name,
                'confidence': round(self.confidence, 3),
                'bbox': [round(self.x1,1), round(self.y1,1), round(self.x2,1), round(self.y2,1)]}


def _nms(boxes, scores, iou_thresh=0.45):
    if len(boxes) == 0: return np.array([], dtype=int)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1: break
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        area_i = (boxes[i,2]-boxes[i,0]) * (boxes[i,3]-boxes[i,1])
        area_j = (boxes[order[1:],2]-boxes[order[1:],0]) * (boxes[order[1:],3]-boxes[order[1:],1])
        iou = inter / (area_i + area_j - inter + 1e-6)
        order = order[np.where(iou <= iou_thresh)[0] + 1]
    return np.array(keep)


class ObjectDetector:
    def __init__(self, model_path, conf_thresh=0.4, nms_thresh=0.45,
                 img_size=640, quantize=False, classes=None):
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.img_size = img_size
        self.classes = classes or COCO_CLASSES
        self.backend = InferenceBackend(model_path, quantize)
        self._resize_fn = None
        try:
            import cv2
            self._resize_fn = lambda img, size: cv2.resize(img, (size, size))
        except ImportError:
            try:
                from PIL import Image as PILImage
                self._resize_fn = lambda img, size: __import__('numpy').array(PILImage.fromarray(img).resize((size, size)))
            except ImportError:
                self._resize_fn = self._numpy_resize
        self._frame_count = 0
        self._total_time = 0

    @classmethod
    def auto(cls, task='detect', conf_thresh=0.4, model_path=None):
        hw = detect_hardware()
        print(f'[EdgeAI] Hardware detected:\n{hw}\n')
        if model_path and os.path.exists(model_path):
            quantize = hw.recommended_quantization == 'int8'
            img_size = hw.recommended_image_size
        else:
            spec = recommend_model(hw, task)
            if spec is None:
                raise RuntimeError(f'No suitable model for {hw.device_name}')
            model_path = f'{spec.name}.onnx'
            if not os.path.exists(model_path):
                print(f'[EdgeAI] Downloading {spec.name} ({spec.size_mb}MB)...')
                import urllib.request
                urllib.request.urlretrieve(spec.url, model_path)
                print(f'[EdgeAI] Downloaded: {model_path}')
            quantize = hw.recommended_quantization == 'int8'
            img_size = min(hw.recommended_image_size, spec.input_size)
        print(f'[EdgeAI] Config: model={model_path}, size={img_size}, quantize={quantize}')
        return cls(model_path, conf_thresh=conf_thresh, img_size=img_size, quantize=quantize)

    @staticmethod
    def _numpy_resize(img, size):
        h, w = img.shape[:2]
        y_idx = (np.arange(size) * h / size).astype(int)
        x_idx = (np.arange(size) * w / size).astype(int)
        return img[np.ix_(y_idx, x_idx)]

    def preprocess(self, image):
        orig_h, orig_w = image.shape[:2]
        resized = self._resize_fn(image, self.img_size)
        x = resized.astype(np.float32) / 255.0
        x = x.transpose(2, 0, 1)[np.newaxis]
        return x, (orig_h, orig_w)

    def detect(self, image, conf_thresh=None):
        conf = conf_thresh or self.conf_thresh
        x, (orig_h, orig_w) = self.preprocess(image)
        output, elapsed = self.backend.run(x)
        self._frame_count += 1
        self._total_time += elapsed
        return self._parse_yolo(output, orig_h, orig_w, conf)

    def _parse_yolo(self, output, orig_h, orig_w, conf_thresh):
        if output.ndim == 3: output = output[0]
        obj_mask = output[:, 4] > conf_thresh
        filtered = output[obj_mask]
        if len(filtered) == 0: return []
        cx, cy, w, h = filtered[:,0], filtered[:,1], filtered[:,2], filtered[:,3]
        boxes = np.stack([cx-w/2, cy-h/2, cx+w/2, cy+h/2], axis=1)
        class_scores = filtered[:, 5:]
        class_ids = np.argmax(class_scores, axis=1)
        class_probs = np.max(class_scores, axis=1)
        confidences = filtered[:, 4] * class_probs
        scale_x = orig_w / self.img_size
        scale_y = orig_h / self.img_size
        detections = []
        for cls_id in np.unique(class_ids):
            cls_mask = class_ids == cls_id
            cls_boxes = boxes[cls_mask]
            cls_scores_arr = confidences[cls_mask]
            keep = _nms(cls_boxes, cls_scores_arr, self.nms_thresh)
            for k in keep:
                b = cls_boxes[k]
                cname = self.classes[int(cls_id)] if int(cls_id) < len(self.classes) else f'class_{cls_id}'
                detections.append(Detection(
                    class_id=int(cls_id), class_name=cname,
                    confidence=float(cls_scores_arr[k]),
                    x1=float(b[0]*scale_x), y1=float(b[1]*scale_y),
                    x2=float(b[2]*scale_x), y2=float(b[3]*scale_y)))
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections

    @property
    def fps(self):
        return self._frame_count / self._total_time if self._total_time > 0 else 0
    @property
    def avg_ms(self):
        return (self._total_time / self._frame_count) * 1000 if self._frame_count > 0 else 0


@dataclass
class RobotCommand:
    linear_x: float = 0.0
    linear_y: float = 0.0
    angular_z: float = 0.0
    action: str = 'stop'
    description: str = ''

    def to_twist_dict(self):
        return {'linear': {'x': self.linear_x, 'y': self.linear_y, 'z': 0.0},
                'angular': {'x': 0.0, 'y': 0.0, 'z': self.angular_z}}


class RobotController:
    def __init__(self, mode='follow', max_linear=0.3, max_angular=0.5, target_classes=None):
        self.mode = mode
        self.max_linear = max_linear
        self.max_angular = max_angular
        self.target_classes = target_classes or ['person']
        self._last_target_time = 0
        self._search_direction = 1.0

    def update(self, detections, image_width=640, image_height=480):
        if self.mode == 'follow': return self._follow(detections, image_width, image_height)
        elif self.mode == 'avoid': return self._avoid(detections, image_width, image_height)
        elif self.mode == 'patrol': return self._patrol(detections, image_width, image_height)
        return RobotCommand(action='idle', description='Unknown mode')

    def _follow(self, detections, img_w, img_h):
        targets = [d for d in detections if d.class_name in self.target_classes]
        if not targets:
            if time.time() - self._last_target_time > 5:
                self._search_direction *= -1
            return RobotCommand(angular_z=self._search_direction * self.max_angular * 0.4,
                                action='searching', description=f'Looking for {self.target_classes[0]}')
        self._last_target_time = time.time()
        target = max(targets, key=lambda d: d.area)
        error_x = (target.cx - img_w / 2) / (img_w / 2)
        angular_z = -error_x * self.max_angular
        relative_size = target.area / (img_w * img_h)
        if relative_size > 0.15:
            return RobotCommand(angular_z=angular_z, action='reached', description=f'{target.class_name} reached')
        elif relative_size > 0.04:
            return RobotCommand(linear_x=self.max_linear*0.5, angular_z=angular_z, action='approaching', description=f'Approaching {target.class_name}')
        else:
            return RobotCommand(linear_x=self.max_linear, angular_z=angular_z, action='following', description=f'Following {target.class_name}')

    def _avoid(self, detections, img_w, img_h):
        obstacles = [d for d in detections if d.class_name in ('person','car','truck','chair','couch','dog')]
        if not obstacles:
            return RobotCommand(linear_x=self.max_linear, action='driving', description='Path clear')
        closest = max(obstacles, key=lambda d: d.area)
        relative_size = closest.area / (img_w * img_h)
        if relative_size > 0.08:
            turn = 1.0 if closest.cx < img_w / 2 else -1.0
            return RobotCommand(angular_z=turn*self.max_angular, action='avoiding', description=f'Avoiding {closest.class_name}')
        error_x = (closest.cx - img_w / 2) / (img_w / 2)
        return RobotCommand(linear_x=self.max_linear*0.5, angular_z=-error_x*self.max_angular*0.3, action='cautious', description=f'{closest.class_name} ahead')

    def _patrol(self, detections, img_w, img_h):
        import math
        t = time.time()
        return RobotCommand(linear_x=self.max_linear*0.3, angular_z=math.sin(t*0.3)*self.max_angular*0.3,
                            action='patrolling', description=f'See {len(detections)} objects')


class EdgeAI:
    def __init__(self, detector, controller=None):
        self.detector = detector
        self.controller = controller or RobotController(mode='follow')
        self.hardware = None

    @classmethod
    def auto(cls, mode='follow', target='person', model_path=None, conf_thresh=0.4):
        hw = detect_hardware()
        detector = ObjectDetector.auto(task='detect', conf_thresh=conf_thresh, model_path=model_path)
        controller = RobotController(mode=mode, target_classes=[target])
        instance = cls(detector, controller)
        instance.hardware = hw
        return instance

    def detect(self, frame): return self.detector.detect(frame)

    def decide(self, detections, image_width=640, image_height=480):
        return self.controller.update(detections, image_width, image_height)

    def process(self, frame):
        h, w = frame.shape[:2]
        detections = self.detect(frame)
        command = self.decide(detections, w, h)
        return {'detections': detections, 'command': command,
                'fps': self.detector.fps, 'avg_ms': self.detector.avg_ms}

    def run_camera(self, source=0, show=True):
        try:
            import cv2
        except ImportError:
            print('OpenCV required: pip install opencv-python')
            return
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f'Cannot open camera {source}')
            return
        print(f'[EdgeAI] Live detection (camera={source}, backend={self.detector.backend.backend_name})')
        print('Press q=quit, m=change mode')
        while True:
            ret, frame = cap.read()
            if not ret: break
            result = self.process(frame)
            if show:
                for det in result['detections']:
                    color = (0,255,0) if det.class_name in self.controller.target_classes else (0,165,255)
                    cv2.rectangle(frame, (int(det.x1),int(det.y1)), (int(det.x2),int(det.y2)), color, 2)
                    cv2.putText(frame, f'{det.class_name} {det.confidence:.0%}',
                               (int(det.x1),int(det.y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cmd = result['command']
                cv2.putText(frame, f'TinyTPU | {cmd.action}', (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                cv2.imshow('TinyTPU EdgeAI', frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'): break
                elif key == ord('m'):
                    modes = ['follow','avoid','patrol']
                    idx = (modes.index(self.controller.mode) + 1) % len(modes)
                    self.controller.mode = modes[idx]
                    print(f'Mode: {self.controller.mode}')
        cap.release()
        cv2.destroyAllWindows()
        print(f'Average: {self.detector.fps:.1f} FPS')


def main():
    import argparse
    parser = argparse.ArgumentParser(description='TinyTPU Edge AI')
    parser.add_argument('command', nargs='?', default='hardware', choices=['detect','camera','benchmark','hardware'])
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--mode', type=str, default='follow')
    parser.add_argument('--target', type=str, default='person')
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--conf', type=float, default=0.4)
    parser.add_argument('--image', type=str, default=None)
    args = parser.parse_args()

    if args.command == 'hardware':
        hw = detect_hardware()
        print('=' * 50)
        print('HARDWARE PROFILE')
        print('=' * 50)
        print(hw)
        for task in ['detect', 'classify']:
            model = recommend_model(hw, task)
            if model:
                print(f'Recommended {task}: {model.name} ({model.size_mb}MB)')

    elif args.command == 'benchmark':
        model_path = args.model or 'yolov5n.onnx'
        if not os.path.exists(model_path):
            spec = MODEL_ZOO.get('yolov5n')
            if spec:
                print(f'Downloading {model_path}...')
                import urllib.request
                urllib.request.urlretrieve(spec.url, model_path)
        detector = ObjectDetector(model_path, conf_thresh=args.conf)
        print(f'Backend: {detector.backend.backend_name}')
        dummy = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        detector.detect(dummy)
        times = []
        for i in range(20):
            start = time.perf_counter()
            detector.detect(dummy)
            times.append(time.perf_counter() - start)
        times = np.array(times[2:]) * 1000
        print(f'Mean: {times.mean():.1f}ms, Median: {np.median(times):.1f}ms, FPS: {1000/np.median(times):.1f}')

    elif args.command == 'camera':
        ai = EdgeAI.auto(mode=args.mode, target=args.target, model_path=args.model)
        ai.run_camera(source=args.camera)

    elif args.command == 'detect':
        if args.image:
            try:
                import cv2
                image = cv2.imread(args.image)
            except ImportError:
                from PIL import Image
                image = np.array(Image.open(args.image))
            detector = ObjectDetector.auto(model_path=args.model, conf_thresh=args.conf)
            for det in detector.detect(image):
                print(f'{det.class_name}: {det.confidence:.0%} at [{det.x1:.0f},{det.y1:.0f},{det.x2:.0f},{det.y2:.0f}]')
        else:
            print('Provide --image or use camera command')


if __name__ == '__main__':
    main()
