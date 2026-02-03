<p align="center">
  <img src="docs/images/hero_banner.png" alt="TinyTPU" width="100%">
</p>

<p align="center">
  <b>Edge AI inference engine for robots without GPUs</b><br>
  Auto-configures to your hardware. Downloads the right model. Runs in 3 lines.
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> &bull;
  <a href="#why-tinytpu">Why TinyTPU</a> &bull;
  <a href="#benchmarks">Benchmarks</a> &bull;
  <a href="#edge-ai-toolkit">Edge AI Toolkit</a> &bull;
  <a href="#ros2-robotics">ROS2 Robotics</a> &bull;
  <a href="#deployment">Deployment</a>
</p>

---

## Quick Start

```bash
pip install numpy onnxruntime
git clone https://github.com/SKBiswas1998/tinytpu.git
cd tinytpu
```

**Detect objects in 3 lines:**

```python
from software.tinytpu.edge_ai import EdgeAI

ai = EdgeAI.auto()  # Detects your hardware, downloads best model
result = ai.process(camera_frame)
# result["detections"] = [{class_name: "person", confidence: 0.92, ...}]
# result["command"] = {action: "following", linear_x: 0.3, angular_z: -0.1}
```

**Or use the CLI:**

```bash
# What can my hardware handle?
python -m software.tinytpu.edge_ai hardware

# Benchmark inference speed
python -m software.tinytpu.edge_ai benchmark

# Live camera detection with robot control
python -m software.tinytpu.edge_ai camera --mode follow --target person

# Detect objects in an image
python -m software.tinytpu.edge_ai detect --image photo.jpg
```

---

## Why TinyTPU

Most AI frameworks assume you have a GPU, a fast CPU, and plenty of RAM. Robots often have none of these.

TinyTPU solves the **last-mile problem** for edge AI:

| Problem | TinyTPU Solution |
|---------|-----------------|
| "Which model fits my 2GB Pi?" | `detect_hardware()` profiles your device, `recommend_model()` picks the best fit |
| "ONNX Runtime or custom engine?" | `InferenceBackend` tries ONNX Runtime first (3-10x faster), falls back to pure Python |
| "How do I go from detection to motor command?" | `RobotController` converts bounding boxes to velocity commands |
| "I just want it to work" | `EdgeAI.auto()` handles everything: hardware detection, model download, quantization, inference |

**What you get:**
- Auto hardware profiling (Pi 4/5, Jetson, x86)
- Auto model selection from the built-in model zoo
- Auto INT8 quantization on memory-constrained devices (75% RAM reduction)
- YOLO object detection with NMS post-processing
- Proportional robot control (follow, avoid, patrol)
- ROS2 integration or standalone operation
- Live camera demo with one command

<p align="center">
  <img src="docs/images/architecture.png" alt="Architecture" width="90%">
</p>

---

## Benchmarks

<p align="center">
  <img src="docs/images/benchmarks.png" alt="Benchmarks" width="90%">
</p>

### Inference Speed

| Model | Task | Desktop (x86) | Raspberry Pi 5 | Pi 4 |
|-------|------|--------------|----------------|------|
| YOLOv5-nano | Detection | 3.1 FPS | ~2 FPS | ~0.5 FPS |
| YOLOv5-small | Detection | 1.5 FPS | ~0.8 FPS | ~0.2 FPS |
| MobileNetV2 | Classification | 13.3 FPS | ~5 FPS | ~2 FPS |
| GPT-2 124M | Text Generation | 15 tok/s | ~3-5 tok/s | ~1-3 tok/s |

All models produce **identical outputs** to ONNX Runtime (correlation = 1.000000).

### Core Engine

| Metric | Value |
|--------|-------|
| Peak GFLOPS (1024x1024 matmul) | **199.2** |
| ReLU vs PyTorch | **1.7x faster** |
| GELU vs PyTorch | **1.3x faster** |
| LayerNorm vs PyTorch | **1.1x faster** |
| INT8 memory reduction | **75%** |
| ONNX operators supported | **50+** |

---

## Edge AI Toolkit

The toolkit auto-configures everything based on your hardware.

### Hardware Detection

```python
from software.tinytpu.edge_ai import detect_hardware, recommend_model

hw = detect_hardware()
print(hw)
# Device: Raspberry Pi 5 Model B Rev 1.0
# CPU: 4 cores @ 2400MHz (aarch64)
# RAM: 4096MB total, 3200MB available
# Max model size: 1120MB
# Recommended: int8, 640px

model = recommend_model(hw, task="detect")
print(model.name)  # yolov5s
```

### Object Detection

```python
from software.tinytpu.edge_ai import ObjectDetector

# Auto: detects hardware, picks model, downloads it
detector = ObjectDetector.auto()

# Or manual: bring your own model
detector = ObjectDetector("yolov5n.onnx", conf_thresh=0.4, img_size=640)

detections = detector.detect(image)
for det in detections:
    print(f"{det.class_name}: {det.confidence:.0%} at ({det.cx:.0f}, {det.cy:.0f})")
```

### Robot Control

```python
from software.tinytpu.edge_ai import RobotController, EdgeAI

# Standalone controller
controller = RobotController(mode="follow", target_classes=["person"])
cmd = controller.update(detections, image_width=640, image_height=480)
print(f"v={cmd.linear_x:.2f} m/s, w={cmd.angular_z:.2f} rad/s")

# Or full pipeline
ai = EdgeAI.auto(mode="follow", target="person")
for frame in camera:
    result = ai.process(frame)
    robot.set_velocity(result["command"].linear_x, result["command"].angular_z)
```

### Control Modes

| Mode | Behavior |
|------|----------|
| `follow` | Track target, proportional steering, speed based on distance |
| `avoid` | Drive forward, steer away from obstacles, stop if too close |
| `patrol` | Wander with sinusoidal path, report detected objects |

### Live Camera

```python
ai = EdgeAI.auto(mode="follow")
ai.run_camera(source=0)  # Opens webcam, shows detections, press m to switch modes
```

<p align="center">
  <img src="docs/images/pipeline.png" alt="Pipeline" width="90%">
</p>

---

## ONNX Engine

The built-in ONNX engine runs any ONNX model with 50+ operators. It serves as a fallback when ONNX Runtime is not available.

<p align="center">
  <img src="docs/images/onnx_operators.png" alt="ONNX Operators" width="90%">
</p>

```python
from software.tinytpu.onnx_engine import TinyTPUEngine

engine = TinyTPUEngine("model.onnx")
engine.summary()
output, elapsed = engine.run({"input": data})
```

### Systolic Array

Under the hood, TinyTPU implements a systolic array architecture for matrix multiplication.

<p align="center">
  <img src="docs/images/systolic_array.png" alt="Systolic Array" width="70%">
</p>

### INT8 Quantization

<p align="center">
  <img src="docs/images/quantization.png" alt="Quantization" width="90%">
</p>

| Model | FP32 | INT8 | Savings | Accuracy |
|-------|------|------|---------|----------|
| GPT-2 124M | 471 MB | 118 MB | **75%** | r=0.9999 |
| MobileNetV2 | 13.3 MB | 3.4 MB | **74%** | r=0.9999 |
| YOLOv5-nano | 3.6 MB | 0.9 MB | **75%** | r=0.9999 |

---

## ROS2 Robotics

TinyTPU includes a ROS2 package with three nodes that work together:

<p align="center">
  <img src="docs/images/ros2_graph.png" alt="ROS2 Graph" width="90%">
</p>

| Node | Subscribes | Publishes | Function |
|------|-----------|-----------|----------|
| **vision_node** | `/camera/image_raw` | `/tinytpu/detections` | YOLOv5 object detection |
| **llm_node** | `/tinytpu/llm_input` | `/tinytpu/llm_output` | GPT-2 scene description + command parsing |
| **brain_node** | detections + llm_output | `/cmd_vel` | Decision making + motor commands |

### Launch

```bash
cd ros2/tinytpu_ros
colcon build --packages-select tinytpu_ros
source install/setup.bash

# Launch all nodes
ros2 launch tinytpu_ros tinytpu_launch.py

# Or with a specific mode
ros2 launch tinytpu_ros tinytpu_launch.py mode:=avoid_obstacles
```

### Natural Language Commands

```bash
ros2 topic pub /tinytpu/llm_input std_msgs/String '{"type":"command","prompt":"follow the person"}'
```

| Command | Action |
|---------|--------|
| "follow the person" | Track and approach detected person |
| "what do you see" | Describe visible objects |
| "find the cup" | Rotate and search |
| "turn left" / "turn right" | Angular velocity command |
| "stop" | Halt all movement |
| "go back" | Reverse |

### Standalone (No ROS Required)

```python
from ros2.tinytpu_ros.tinytpu_ros.vision_node import VisionProcessor
from ros2.tinytpu_ros.tinytpu_ros.brain_node import BrainProcessor

vision = VisionProcessor("yolov5n.onnx")
brain = BrainProcessor(mode="follow_person")

detections, elapsed = vision.detect(camera_frame)
command = brain.process(detections)
robot.set_velocity(command["linear_x"], command["angular_z"])
```

---

## Deployment

<p align="center">
  <img src="docs/images/devices.png" alt="Devices" width="90%">
</p>

### Raspberry Pi

```bash
pip install numpy onnxruntime opencv-python
git clone https://github.com/SKBiswas1998/tinytpu.git
cd tinytpu
python -m software.tinytpu.edge_ai camera --mode follow
```

The toolkit auto-detects your Pi model and adjusts:
- **Pi 5 (4-8GB):** YOLOv5-small, INT8, 640px
- **Pi 4 (2-4GB):** YOLOv5-nano, INT8, 480px
- **Pi Zero (512MB):** MobileNetV2 0.5x, INT4, 160px

### Dependencies

| Package | Required | Purpose |
|---------|----------|---------|
| numpy | Yes | Core operations |
| onnxruntime | Recommended | Fast inference (3-10x over pure Python) |
| opencv-python | For camera | Live camera demo |
| torch | Optional | Faster backend for some operations |
| Pillow | Optional | Image loading fallback |

---

## Roadmap

<p align="center">
  <img src="docs/images/roadmap.png" alt="Roadmap" width="90%">
</p>

- [x] **Phase 1: Core Engine** -- Systolic array, Python API, GPT-2 at 15 tok/s, INT8 quantization, ONNX engine with 50+ ops
- [x] **Phase 2: Perception** -- MobileNetV2 at 13 FPS, YOLOv5 at 3 FPS, NMS, output correlation 1.0 with ONNX Runtime
- [x] **Phase 3: Robotics** -- ROS2 package, vision + LLM + brain nodes, follow/avoid/patrol, natural language commands
- [ ] **Phase 4: Deployment** -- Live camera demo, Raspberry Pi testing, MicroROS for Pico, PyPI package
- [ ] **Phase 5: Advanced** -- SLAM integration, voice commands, TinyLlama on Pi 5, FPGA deployment

---

## Project Structure

```
tinytpu/
  software/tinytpu/
    edge_ai.py              # EdgeAI toolkit (start here)
    tpu_v2.py               # Core TinyTPU engine
    int8_quantization.py    # INT8 quantization
    gpt2_optimized.py       # LLM inference
    onnx_engine/
      engine.py             # ONNX runtime (50+ operators)
  ros2/tinytpu_ros/
    tinytpu_ros/
      vision_node.py        # Camera -> YOLO -> detections
      llm_node.py           # NL understanding + commands
      brain_node.py         # Decision making -> /cmd_vel
    launch/
      tinytpu_launch.py     # ROS2 launch file
    config/
      tinytpu_config.yaml   # Node parameters
  docs/images/              # Architecture diagrams
  test_edge_ai.py           # Toolkit test
  test_yolo.py              # YOLOv5 test
  test_robotics.py          # Full pipeline test
  speed_test.py             # Core benchmarks
```

---

## Contributing

Contributions welcome, especially:
- Raspberry Pi performance testing and optimization
- Additional ONNX operator implementations
- New robot control behaviors
- FPGA/ASIC synthesis of the systolic array RTL
- Model zoo additions (pose estimation, depth, segmentation)

---

## License

MIT

---

<p align="center">
  <i>Built for robots that think at the edge.</i><br><br>
  <code>pip install numpy onnxruntime</code><br>
  <code>ai = EdgeAI.auto()</code><br>
  <code>result = ai.process(frame)</code>
</p>
