# Changelog

All notable changes to TinyTPU will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-02-08

### Added

**Core Framework**
- Pip-installable Python package (`pip install tinytpu`)
- CLI with 7 commands: `version`, `hardware`, `detect`, `benchmark`, `models`, `download`, `backends`
- Lazy-loading top-level API: `tinytpu.Model()`, `tinytpu.Pipeline()`

**Hardware Abstraction Layer**
- `InferenceBackend` abstract interface for all accelerators
- ONNX Runtime backend (CPU/CUDA/TensorRT) — priority 50
- Hailo NPU backend (13–26 TOPS) — priority 90
- Google Coral Edge TPU backend (4 TOPS) — priority 80
- NumPy fallback backend — priority 10
- `auto_backend()` with priority-based selection and format compatibility
- `detect_hardware()` scans for Hailo, Coral, IMX500, NVIDIA GPU, CPU

**Inference**
- Pure-Python ONNX engine (50+ operators) as universal fallback
- Model zoo with 6 pre-configured models (YOLOv8n/s/m, pose, seg, MobileNetV2)
- Auto-download with local caching
- YOLO pre/post-processing with NMS
- `Model.predict()` returns `PredictionResult` with typed `Detection` objects

**Perception**
- `ObjectTracker`: SORT-style multi-object tracking with IoU matching
- `KalmanFilter2D`: 8-state filter [cx, cy, w, h, vx, vy, vw, vh]
- Track lifecycle: tentative → confirmed (after `min_hits` consecutive detections)
- Persistent track IDs across frames

**Control**
- `Pipeline`: Full async perception-to-action loop (3 threads: capture, inference, control)
- Follow mode: proportional control (horizontal error → angular, vertical size → linear)
- Patrol mode: slow rotation scan, stops on detection
- `SafetyController`: E-stop, watchdog timeout, velocity ramping, proximity stop
- Hard velocity limits with configurable max linear/angular speeds

**Monitoring**
- `ThermalMonitor`: Reads SoC temperature, warns at thresholds
- `MemoryWatchdog`: Tracks RSS, triggers GC at limits
- `BlackBoxRecorder`: Event logging for post-incident analysis

**Numerical Methods**
- Richardson extrapolation for quantization error reduction
- HITTER iterative eigenvalue decomposition for on-device PCA
- Horner-form polynomial activations (sigmoid, GELU, SiLU)

**Testing**
- 61 comprehensive tests across 2 test files
- Tests for: imports, hardware detection, model zoo, CLI, preprocessing,
  postprocessing, backends, pipeline, safety, tracker integration

**Demo**
- `demo_e2e.py`: End-to-end demo with image detection, live webcam, tracking,
  safety controller, and inference benchmarking
- Live webcam mode with bounding boxes, track IDs, and motor command visualization

### Performance
- YOLOv8n inference: 148ms on Intel i7-10510U (CPU), 7.3 FPS
- Model loading: 145ms (cached), 28s (first download + ONNX export)
- 61 tests pass in ~6.6s

[0.1.0]: https://github.com/SKBiswas1998/tinytpu/releases/tag/v0.1.0
