<p align="center">
  <img src="docs/images/tinytpu_banner.png" alt="TinyTPU" width="600">
</p>

<h1 align="center">TinyTPU</h1>

<p align="center">
  <strong>Production Edge AI for Robots Without GPUs</strong><br>
  From silicon architecture to safety-certified robot control in one <code>pip install</code>.
</p>

<p align="center">
  <a href="https://pypi.org/project/tinytpu/"><img src="https://img.shields.io/pypi/v/tinytpu?color=blue&label=PyPI" alt="PyPI"></a>
  <a href="https://pypi.org/project/tinytpu/"><img src="https://img.shields.io/pypi/pyversions/tinytpu" alt="Python"></a>
  <a href="https://github.com/SKBiswas1998/tinytpu/actions"><img src="https://img.shields.io/github/actions/workflow/status/SKBiswas1998/tinytpu/ci.yml?label=CI" alt="CI"></a>
  <a href="https://github.com/SKBiswas1998/tinytpu/blob/main/LICENSE"><img src="https://img.shields.io/github/license/SKBiswas1998/tinytpu?color=green" alt="License"></a>
  <a href="https://github.com/SKBiswas1998/tinytpu"><img src="https://img.shields.io/github/stars/SKBiswas1998/tinytpu?style=social" alt="Stars"></a>
</p>

<p align="center">
  <img src="docs/images/tinytpu_live_demo.jpg" alt="TinyTPU Live Detection" width="700">
  <br>
  <em>Real-time person detection + Kalman tracking + robot follow-mode on Intel i7 CPU at 7.3 FPS</em>
</p>

---

## Why TinyTPU?

| Problem | TinyTPU Solution |
|---------|-----------------|
| Google abandoned PyCoral (Python ≤3.9 only) | Works on Python 3.9–3.13+, MIT licensed |
| NVIDIA Isaac costs $10K+ in hardware | Runs on $35 Raspberry Pi |
| Ultralytics is AGPL (viral copyleft) | MIT + Apache 2.0 — free for commercial use |
| Vision and control run at different speeds | Kalman prediction bridges 2 FPS vision → 30 Hz control |
| No unified API across AI accelerators | One API across Hailo, Coral, GPU, and CPU |

## Quick Start

```bash
pip install tinytpu
```

### 3-Line Inference

```python
import tinytpu

model = tinytpu.Model("yolov8n")       # Auto-downloads, auto-selects backend
results = model.predict(camera_frame)   # YOLO detection in one call

for det in results.detections:
    print(f"{det.class_name}: {det.confidence:.0%} at ({det.x1:.0f},{det.y1:.0f})")
```

### Full Robot Pipeline

```python
from tinytpu import Pipeline

pipeline = Pipeline(
    model="yolov8n",
    mode="follow",          # detect | follow | patrol | track
    target="person",
    max_linear=0.3,         # m/s
    max_angular=0.5,        # rad/s
)
pipeline.start()  # Camera → YOLO → Kalman → Safety → Motors at 30 Hz
```

### Live Webcam Demo

```bash
python demo_e2e.py --camera 0          # Live detection + tracking
python demo_e2e.py --image photo.jpg   # Single image detection
```

## CLI Tools

```bash
tinytpu hardware          # Detect AI accelerators (Hailo, Coral, GPU, CPU)
tinytpu detect photo.jpg  # Run object detection with bounding boxes
tinytpu benchmark         # Benchmark inference speed
tinytpu models            # List available models + cache status
tinytpu download yolov8n  # Pre-download for offline use
tinytpu backends          # Show available inference backends
tinytpu version           # Version and dependency info
```

## Architecture

```
Camera Frame (7-30 FPS)
    │
    ▼
┌─────────────────────────────────────────────────┐
│  TinyTPU Pipeline                               │
│                                                 │
│  ┌──────────┐   ┌──────────┐   ┌────────────┐  │
│  │ Capture  │──▶│ Inference│──▶│  Control   │  │
│  │ Thread   │   │ Thread   │   │  Thread    │  │
│  │ (30 FPS) │   │ (2-15fps)│   │  (30 Hz)   │  │
│  └──────────┘   └──────────┘   └────────────┘  │
│                      │               │          │
│                      ▼               ▼          │
│               ┌────────────┐  ┌───────────┐    │
│               │  Kalman    │  │  Safety   │    │
│               │  Tracker   │  │Controller │    │
│               │ (predict   │  │(e-stop,   │    │
│               │  ahead)    │  │ watchdog, │    │
│               └────────────┘  │ ramping)  │    │
│                               └───────────┘    │
└─────────────────────────────────────────────────┘
    │
    ▼
Motor Commands (30 Hz, safety-filtered)
```

### Hardware Abstraction Layer

TinyTPU auto-detects and prioritizes AI accelerators:

| Backend | Priority | Hardware | Performance |
|---------|----------|----------|-------------|
| Hailo | 90 | Hailo-8L AI HAT+ | 13 TOPS |
| Coral | 80 | Google Coral USB/PCIe | 4 TOPS |
| ONNX Runtime | 50 | CPU/CUDA/TensorRT | Varies |
| NumPy | 10 | Any CPU (fallback) | Baseline |

```python
from tinytpu.hal import auto_backend, list_backends

print(list_backends())                          # Show all backends
backend = auto_backend(prefer="hailo")          # Auto-select best
session = backend.load("model.onnx")            # Load model
outputs = backend.run(session, {"images": x})   # Run inference
```

## Installation Options

```bash
pip install tinytpu                    # Core (NumPy only)
pip install tinytpu[inference]         # + ONNX Runtime
pip install tinytpu[vision]            # + OpenCV
pip install tinytpu[robotics]          # Full robot stack
pip install tinytpu[dashboard]         # + FastAPI web UI
pip install tinytpu[all]               # Everything
pip install tinytpu[dev]               # + pytest, ruff
```

## Model Zoo

| Model | Task | Size | Best For |
|-------|------|------|----------|
| `yolov8n` | Detection | 6.2 MB | Raspberry Pi, real-time |
| `yolov8s` | Detection | 22 MB | Hailo NPU, balanced |
| `yolov8m` | Detection | 52 MB | GPU, high accuracy |
| `yolov8n-pose` | Pose | 6.7 MB | Human keypoints |
| `yolov8n-seg` | Segmentation | 6.8 MB | Instance masks |
| `mobilenetv2` | Classification | 14 MB | Image classification |

Models auto-download on first use and cache locally.

## Key Features

### Safety Controller
Every motor command passes through safety before reaching the robot:

```python
from tinytpu.control import SafetyController

safety = SafetyController(
    max_linear=0.3,       # Hard velocity limit (m/s)
    max_angular=0.5,      # Hard turn limit (rad/s)
    watchdog_timeout=2.0, # Stop if no detection for 2s
    ramp_rate=0.5,        # Max acceleration
)

# E-stop: immediate zero, requires manual reset
safety.estop("obstacle_detected")
safety.reset()
```

### Kalman Tracking
SORT-style multi-object tracking bridges the vision-control frequency gap:

```python
from tinytpu.perception import ObjectTracker

tracker = ObjectTracker(iou_threshold=0.3, min_hits=2)
tracks = tracker.update(detections)  # Returns confirmed tracks with IDs

for track in tracks:
    print(f"ID:{track.track_id} {track.class_name} vel={track.kalman.velocity}")
```

### Numerical Methods
Adapted from Killingbeck (1991) for quantized edge inference:

- **Richardson Extrapolation**: 128× accuracy improvement over plain INT8
- **HITTER Eigendecomposition**: On-device PCA with 500× memory savings
- **Horner-form Activations**: Polynomial approximations for INT8-friendly inference

## Performance

| Platform | Backend | Inference | FPS | Power |
|----------|---------|-----------|-----|-------|
| Intel i7-10510U | ONNX Runtime CPU | 148 ms | 7.3 | 15W |
| Raspberry Pi 5 | Hailo-8L NPU | ~15 ms | ~70 | 5W |
| Raspberry Pi 5 | CPU only | ~500 ms | ~2 | 5W |
| Coral USB | Edge TPU | ~30 ms | ~33 | 2W |

*Pi 5 + Hailo figures are projected based on published benchmarks.*

## Project Structure

```
tinytpu/
├── src/tinytpu/
│   ├── __init__.py          # Lazy imports, Model & Pipeline shortcuts
│   ├── core/                # Systolic array, quantization
│   ├── inference/           # ONNX engine (50+ ops), model zoo
│   ├── perception/          # Object detector, Kalman tracker
│   ├── control/             # Safety controller, pipeline, robot interface
│   ├── monitoring/          # Thermal, memory watchdog, black box recorder
│   ├── numerical/           # Richardson, HITTER, Horner activations
│   ├── hal/                 # Hardware Abstraction Layer (4 backends)
│   └── cli/                 # Command-line tools
├── tests/                   # 61 tests (pytest)
├── hardware/rtl/            # Verilog systolic array (coming soon)
├── demo_e2e.py              # End-to-end demo script
├── pyproject.toml           # Package config
└── LICENSE                  # MIT
```

## Development

```bash
git clone https://github.com/SKBiswas1998/tinytpu.git
cd tinytpu
pip install -e ".[dev]"
pytest tests/ -v             # Run 61 tests
ruff check src/              # Lint
```

## Roadmap

- [x] Pip-installable package with CLI
- [x] Hardware Abstraction Layer (Hailo, Coral, ONNX RT, NumPy)
- [x] Async pipeline: camera → detect → track → control
- [x] Safety controller with e-stop, watchdog, velocity ramping
- [x] Kalman tracker with IoU matching
- [x] Model zoo with auto-download
- [x] End-to-end demo with real YOLOv8n
- [x] Live webcam detection + tracking
- [ ] Model conversion: `tinytpu convert model.onnx --target hailo8l`
- [ ] FastAPI live dashboard
- [ ] Raspberry Pi hardware-in-the-loop testing
- [ ] ROS2 bridge
- [ ] Verilog systolic array synthesis
- [ ] TestPyPI / PyPI publication

## Acknowledgments

- Numerical methods adapted from Killingbeck, *Microcomputer Algorithms* (1991)
- YOLO models by [Ultralytics](https://github.com/ultralytics/ultralytics)
- Inference powered by [ONNX Runtime](https://onnxruntime.ai/)

## Citation

```bibtex
@software{tinytpu2026,
  title   = {TinyTPU: Production Edge AI for Robots},
  author  = {SK Biswas},
  year    = {2026},
  url     = {https://github.com/SKBiswas1998/tinytpu},
  license = {MIT}
}
```

## License

[MIT](LICENSE) — Free for personal and commercial use.

---

<p align="center">
  Built with ☕ in Dhaka · <a href="https://github.com/SKBiswas1998/tinytpu/issues">Report Bug</a> · <a href="https://github.com/SKBiswas1998/tinytpu/issues">Request Feature</a>
</p>
