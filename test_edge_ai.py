"""
Test the EdgeAI toolkit.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'software'))

from tinytpu.edge_ai import (
    detect_hardware, recommend_model, MODEL_ZOO,
    ObjectDetector, RobotController, EdgeAI, Detection
)
import numpy as np

print("=" * 70)
print("TINYTPU EDGE AI TOOLKIT TEST")
print("=" * 70)

# 1. Hardware Detection
print("\n[1. HARDWARE DETECTION]")
hw = detect_hardware()
print(hw)

# 2. Model Recommendation
print("\n[2. MODEL RECOMMENDATIONS]")
for task in ['detect', 'classify']:
    model = recommend_model(hw, task)
    if model:
        print(f"  {task}: {model.name} ({model.size_mb}MB, "
              f"expect ~{model.expected_fps_desktop} FPS desktop)")

# 3. Auto-configured Detector
print("\n[3. AUTO OBJECT DETECTOR]")
detector = ObjectDetector.auto(task="detect")
print(f"  Backend: {detector.backend.backend_name}")
print(f"  Image size: {detector.img_size}")

# Test detection
img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
dets = detector.detect(img)
print(f"  Test detection: {len(dets)} objects, {detector.avg_ms:.0f}ms")

# 4. Robot Controller
print("\n[4. ROBOT CONTROLLER]")
controller = RobotController(mode="follow", target_classes=["person"])

scenarios = [
    ("No target", []),
    ("Person left", [Detection(0, 'person', 0.9, 50, 100, 150, 380)]),
    ("Person center", [Detection(0, 'person', 0.9, 250, 100, 390, 380)]),
    ("Person close", [Detection(0, 'person', 0.95, 100, 20, 540, 460)]),
    ("Person + car", [
        Detection(0, 'person', 0.9, 300, 100, 400, 380),
        Detection(2, 'car', 0.8, 50, 200, 200, 350),
    ]),
]

for name, dets in scenarios:
    cmd = controller.update(dets, 640, 480)
    print(f"  {name:20s} -> {cmd.action:12s} v={cmd.linear_x:+.2f} w={cmd.angular_z:+.2f}  {cmd.description}")

# 5. Full Pipeline
print("\n[5. FULL PIPELINE]")
ai = EdgeAI(detector, controller)

from PIL import Image, ImageDraw
pil_img = Image.new('RGB', (640, 480), (180, 200, 180))
draw = ImageDraw.Draw(pil_img)
draw.ellipse([250, 300, 350, 400], fill=(255, 200, 0))
frame = np.array(pil_img)

result = ai.process(frame)
print(f"  Detections: {len(result['detections'])}")
print(f"  Command: {result['command'].action} - {result['command'].description}")
print(f"  Speed: {result['avg_ms']:.0f}ms ({result['fps']:.1f} FPS)")

# 6. Benchmark
print("\n[6. BENCHMARK (10 frames)]")
times = []
for i in range(10):
    result = ai.process(frame)
    times.append(result['avg_ms'])

print(f"  Backend: {detector.backend.backend_name}")
print(f"  Average: {np.mean(times):.0f}ms")
print(f"  FPS: {detector.fps:.1f}")

# 7. CLI usage
print("\n[7. CLI USAGE]")
print("""
  # Detect hardware capabilities
  python -m tinytpu.edge_ai hardware

  # Benchmark on your device  
  python -m tinytpu.edge_ai benchmark

  # Run live camera detection
  python -m tinytpu.edge_ai camera --mode follow --target person

  # Detect objects in image
  python -m tinytpu.edge_ai detect --image photo.jpg

  # Python API (3 lines)
  from tinytpu.edge_ai import EdgeAI
  ai = EdgeAI.auto()
  result = ai.process(camera_frame)
""")

print("=" * 70)
print("EDGE AI TOOLKIT: READY FOR DEPLOYMENT")
print("=" * 70)
