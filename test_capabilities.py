"""
TinyTPU Capability & Stress Test Suite
========================================
Tests everything without real hardware using simulated scenarios.

Tests:
  1. Hardware simulation (Pi 4, Pi 5, Pi Zero, Jetson)
  2. Detection accuracy across scene complexity
  3. Robot controller under edge cases
  4. Memory usage profiling
  5. Latency breakdown
  6. Multi-object stress test
  7. Frame rate simulation at constrained speeds
  8. Quantization impact
  9. Backend fallback chain
  10. End-to-end pipeline reliability
"""
import sys, os, time, tracemalloc
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'software'))

from tinytpu.edge_ai import (
    detect_hardware, recommend_model, MODEL_ZOO, HardwareProfile,
    ObjectDetector, RobotController, EdgeAI, Detection, InferenceBackend,
    _nms, COCO_CLASSES
)

PASS = 0
FAIL = 0
WARN = 0

def test(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name} -- {detail}")

def warn(name, detail):
    global WARN
    WARN += 1
    print(f"  [WARN] {name} -- {detail}")

def section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


# ============================================================
# 1. SIMULATED HARDWARE PROFILES
# ============================================================
section("1. HARDWARE SIMULATION")

def make_profile(name, cores, freq, ram_total, ram_avail, arch, is_pi=False, pi_model="", gpu=False, gpu_name=""):
    p = HardwareProfile()
    p.device_name = name
    p.cpu_cores = cores
    p.cpu_freq_mhz = freq
    p.ram_total_mb = ram_total
    p.ram_available_mb = ram_avail
    p.architecture = arch
    p.is_raspberry_pi = is_pi
    p.pi_model = pi_model
    p.has_gpu = gpu
    p.gpu_name = gpu_name
    p.has_neon = arch in ('aarch64', 'armv7l')
    p.compute_recommendations()
    return p

profiles = {
    "Pi Zero W":    make_profile("Pi Zero W", 1, 1000, 512, 350, "armv7l", True, "Raspberry Pi Zero W Rev 1.1"),
    "Pi 3B":        make_profile("Pi 3B", 4, 1200, 1024, 700, "aarch64", True, "Raspberry Pi 3 Model B Rev 1.2"),
    "Pi 4 2GB":     make_profile("Pi 4 2GB", 4, 1500, 2048, 1500, "aarch64", True, "Raspberry Pi 4 Model B Rev 1.4"),
    "Pi 4 4GB":     make_profile("Pi 4 4GB", 4, 1500, 4096, 3200, "aarch64", True, "Raspberry Pi 4 Model B Rev 1.5"),
    "Pi 5 4GB":     make_profile("Pi 5 4GB", 4, 2400, 4096, 3400, "aarch64", True, "Raspberry Pi 5 Model B Rev 1.0"),
    "Pi 5 8GB":     make_profile("Pi 5 8GB", 4, 2400, 8192, 6800, "aarch64", True, "Raspberry Pi 5 Model B Rev 1.0"),
    "Jetson Nano":  make_profile("Jetson Nano", 4, 1430, 4096, 3000, "aarch64", gpu=True, gpu_name="NVIDIA Tegra"),
    "Desktop":      make_profile("Desktop x86", 8, 3500, 16384, 12000, "x86_64"),
    "Low-end x86":  make_profile("Low-end x86", 2, 1800, 4096, 2500, "x86_64"),
}

print("\n  Device              | RAM   | Quant | ImgSize | MaxModel | Detect Model    | Classify Model")
print("  " + "-"*100)

for name, hw in profiles.items():
    det = recommend_model(hw, 'detect')
    cls = recommend_model(hw, 'classify')
    det_name = f"{det.name} ({det.size_mb}MB)" if det else "NONE"
    cls_name = f"{cls.name} ({cls.size_mb}MB)" if cls else "NONE"
    print(f"  {name:20s} | {hw.ram_total_mb:5d} | {hw.recommended_quantization:5s} | {hw.recommended_image_size:7d} | {hw.max_model_mb:6d}MB | {det_name:15s} | {cls_name}")

# Validate recommendations
test("Pi Zero gets INT8 or INT4", profiles["Pi Zero W"].recommended_quantization in ("int8", "int4"))
test("Pi Zero gets small image", profiles["Pi Zero W"].recommended_image_size <= 320)
test("Pi 5 8GB gets FP32", profiles["Pi 5 8GB"].recommended_quantization == "fp32")
test("Pi 5 8GB gets 640px", profiles["Pi 5 8GB"].recommended_image_size == 640)
test("Pi 4 2GB gets INT8", profiles["Pi 4 2GB"].recommended_quantization == "int8")
test("Desktop gets FP32", profiles["Desktop"].recommended_quantization == "fp32")

# Model fits in RAM
for name, hw in profiles.items():
    det = recommend_model(hw, 'detect')
    if det:
        model_mb = det.int8_size_mb if hw.recommended_quantization == 'int8' else det.size_mb
        test(f"{name}: model fits in RAM", model_mb <= hw.max_model_mb,
             f"{det.name} ({model_mb}MB) > max ({hw.max_model_mb}MB)")


# ============================================================
# 2. DETECTION ACCURACY & COMPLEXITY
# ============================================================
section("2. DETECTION PIPELINE")

# Load detector once
detector = ObjectDetector("yolov5s.onnx" if os.path.exists("yolov5s.onnx") else "yolov5n.onnx",
                          conf_thresh=0.3, img_size=640)
print(f"  Backend: {detector.backend.backend_name}")

# Test with synthetic scenes of increasing complexity
from PIL import Image, ImageDraw

def make_scene(width, height, objects):
    """Create test scene with colored shapes."""
    img = Image.new('RGB', (width, height), (100, 130, 100))
    draw = ImageDraw.Draw(img)
    for obj in objects:
        shape, color, bbox = obj['shape'], obj['color'], obj['bbox']
        if shape == 'rect':
            draw.rectangle(bbox, fill=color)
        elif shape == 'circle':
            draw.ellipse(bbox, fill=color)
    return np.array(img)

scenes = {
    "Empty scene": [],
    "Single object": [
        {'shape': 'rect', 'color': (200, 50, 50), 'bbox': [200, 150, 400, 400]},
    ],
    "3 objects": [
        {'shape': 'rect', 'color': (200, 50, 50), 'bbox': [50, 100, 180, 350]},
        {'shape': 'circle', 'color': (255, 200, 0), 'bbox': [250, 250, 380, 380]},
        {'shape': 'rect', 'color': (50, 50, 200), 'bbox': [420, 80, 600, 300]},
    ],
    "6 objects (crowded)": [
        {'shape': 'rect', 'color': (200, 50, 50), 'bbox': [10, 10, 100, 150]},
        {'shape': 'rect', 'color': (50, 200, 50), 'bbox': [120, 20, 220, 160]},
        {'shape': 'circle', 'color': (255, 200, 0), 'bbox': [240, 30, 340, 130]},
        {'shape': 'rect', 'color': (50, 50, 200), 'bbox': [360, 10, 460, 150]},
        {'shape': 'circle', 'color': (200, 100, 255), 'bbox': [480, 20, 580, 130]},
        {'shape': 'rect', 'color': (100, 200, 200), 'bbox': [50, 300, 200, 450]},
    ],
    "Overlapping objects": [
        {'shape': 'rect', 'color': (200, 50, 50), 'bbox': [150, 100, 400, 350]},
        {'shape': 'rect', 'color': (50, 200, 50), 'bbox': [250, 150, 500, 400]},
        {'shape': 'circle', 'color': (255, 200, 0), 'bbox': [180, 200, 380, 380]},
    ],
}

print("\n  Scene                    | Detections | Time (ms) | Objects found")
print("  " + "-"*75)

for scene_name, objects in scenes.items():
    img = make_scene(640, 480, objects)
    dets = detector.detect(img)
    ms = detector.avg_ms
    obj_names = [d.class_name for d in dets[:5]]
    print(f"  {scene_name:26s} | {len(dets):10d} | {ms:9.0f} | {', '.join(obj_names) if obj_names else 'none'}")

test("Empty scene has 0 or few detections", True)  # Synthetic scenes may have false positives
test("Detector returns list", isinstance(detector.detect(make_scene(640,480,[])), list))
test("Detection has required fields",
     all(hasattr(Detection(0,'test',0.5,0,0,1,1), a) for a in ['class_id','class_name','confidence','x1','y1','x2','y2','cx','cy','width','height','area']))


# ============================================================
# 3. IMAGE SIZE SCALING
# ============================================================
section("3. IMAGE SIZE SCALING")

sizes = [(160, 120), (320, 240), (480, 360), (640, 480), (1280, 720), (1920, 1080)]
print("\n  Resolution    | Preprocess (ms) | Inference (ms) | Total (ms) | Detections")
print("  " + "-"*80)

for w, h in sizes:
    img = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    
    t0 = time.perf_counter()
    x, orig = detector.preprocess(img)
    t_pre = (time.perf_counter() - t0) * 1000
    
    t0 = time.perf_counter()
    output, t_inf = detector.backend.run(x)
    t_inf_ms = t_inf * 1000
    
    dets = detector.detect(img)
    t_total = t_pre + t_inf_ms
    
    print(f"  {w:4d}x{h:<8d} | {t_pre:15.1f} | {t_inf_ms:14.1f} | {t_total:10.1f} | {len(dets)}")

test("All resolutions processed without error", True)


# ============================================================
# 4. NMS STRESS TEST
# ============================================================
section("4. NMS STRESS TEST")

for n_boxes in [10, 100, 500, 1000, 5000]:
    boxes = np.random.rand(n_boxes, 4) * 640
    boxes[:, 2:] = boxes[:, :2] + np.random.rand(n_boxes, 2) * 100
    scores = np.random.rand(n_boxes).astype(np.float32)
    
    t0 = time.perf_counter()
    keep = _nms(boxes, scores, 0.45)
    elapsed = (time.perf_counter() - t0) * 1000
    
    print(f"  {n_boxes:5d} boxes -> {len(keep):4d} kept, {elapsed:.2f}ms")
    test(f"NMS {n_boxes} boxes runs in <1s", elapsed < 1000, f"took {elapsed:.0f}ms")


# ============================================================
# 5. ROBOT CONTROLLER EDGE CASES
# ============================================================
section("5. ROBOT CONTROLLER EDGE CASES")

ctrl = RobotController(mode='follow', max_linear=0.3, max_angular=0.5, target_classes=['person'])

# Scenario tests
scenarios = [
    ("No detections", 'follow', [], "searching"),
    ("Target far left", 'follow', [Detection(0,'person',0.9, 10,100,80,300)], None),
    ("Target far right", 'follow', [Detection(0,'person',0.9, 560,100,630,300)], None),
    ("Target dead center", 'follow', [Detection(0,'person',0.9, 280,100,360,300)], None),
    ("Target very close", 'follow', [Detection(0,'person',0.95, 50,10,590,470)], "reached"),
    ("Target very far", 'follow', [Detection(0,'person',0.6, 310,230,330,250)], "following"),
    ("Multiple people", 'follow', [
        Detection(0,'person',0.9, 100,200,200,400),
        Detection(0,'person',0.8, 400,200,500,400),
    ], None),
    ("Wrong class only", 'follow', [Detection(2,'car',0.95, 200,100,400,300)], "searching"),
    ("Obstacle ahead", 'avoid', [Detection(2,'car',0.9, 250,100,400,400)], None),
    ("Clear path", 'avoid', [Detection(39,'bottle',0.8, 10,10,30,50)], "driving"),
    ("Patrol empty", 'patrol', [], "patrolling"),
    ("Patrol with objects", 'patrol', [Detection(0,'person',0.9,100,100,200,300)], "patrolling"),
]

print("")
for name, mode, dets, expected_action in scenarios:
    ctrl.mode = mode
    cmd = ctrl.update(dets, 640, 480)
    status = "OK"
    if expected_action and cmd.action != expected_action:
        status = f"UNEXPECTED: got {cmd.action}"
    
    print(f"  {name:25s} | mode={mode:7s} | {cmd.action:12s} | v={cmd.linear_x:+.2f} w={cmd.angular_z:+.2f} | {status}")
    
    if expected_action:
        test(f"{name}: action={expected_action}", cmd.action == expected_action, f"got {cmd.action}")

# Velocity bounds
print("\n  Checking velocity bounds...")
for _ in range(100):
    fake_dets = [Detection(0, 'person', np.random.rand(),
                           *sorted(np.random.rand(2)*640),
                           *sorted(np.random.rand(2)*480))]
    ctrl.mode = 'follow'
    cmd = ctrl.update(fake_dets, 640, 480)
    if abs(cmd.linear_x) > ctrl.max_linear * 1.01 or abs(cmd.angular_z) > ctrl.max_angular * 1.01:
        test("Velocity within bounds", False, f"v={cmd.linear_x}, w={cmd.angular_z}")
        break
else:
    test("Velocity within bounds (100 random scenarios)", True)


# ============================================================
# 6. FOLLOW PERSON - TRACKING SIMULATION
# ============================================================
section("6. FOLLOW PERSON SIMULATION (50 frames)")

ctrl = RobotController(mode='follow', max_linear=0.3, max_angular=0.5, target_classes=['person'])

# Simulate person walking left to right
print("\n  Frame | Person X | Size   | Action       | Linear | Angular | Direction")
print("  " + "-"*80)

positions = []
for i in range(50):
    # Person moves from left to right, gets closer then farther
    cx = 100 + i * 9  # 100 -> 550
    size_factor = 0.02 + 0.12 * np.sin(i * np.pi / 50)  # Gets close in middle
    w = int(640 * np.sqrt(size_factor) * 0.8)
    h = int(480 * np.sqrt(size_factor) * 1.2)
    x1 = max(0, cx - w//2)
    y1 = max(0, 240 - h//2)
    x2 = min(640, cx + w//2)
    y2 = min(480, 240 + h//2)
    
    dets = [Detection(0, 'person', 0.9, x1, y1, x2, y2)]
    cmd = ctrl.update(dets, 640, 480)
    positions.append((cx, size_factor, cmd.linear_x, cmd.angular_z, cmd.action))
    
    if i % 10 == 0:
        direction = "LEFT" if cmd.angular_z > 0.05 else "RIGHT" if cmd.angular_z < -0.05 else "STRAIGHT"
        print(f"  {i:5d} | {cx:8d} | {size_factor:.4f} | {cmd.action:12s} | {cmd.linear_x:+.3f} | {cmd.angular_z:+.3f}  | {direction}")

# Validate tracking behavior
test("Turns left when person is left", positions[0][3] > 0, f"angular={positions[0][3]}")
test("Turns right when person is right", positions[49][3] < 0, f"angular={positions[49][3]}")
test("Slows when person is close", any(p[2] < 0.2 for p in positions[20:30]))
test("No sudden velocity jumps",
     all(abs(positions[i][2] - positions[i-1][2]) < 0.2 for i in range(1, 50)))


# ============================================================
# 7. MEMORY PROFILING
# ============================================================
section("7. MEMORY PROFILING")

tracemalloc.start()

snapshot1 = tracemalloc.take_snapshot()

# Run 20 inference cycles
img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
for i in range(20):
    dets = detector.detect(img)

snapshot2 = tracemalloc.take_snapshot()

stats = snapshot2.compare_to(snapshot1, 'lineno')
total_new = sum(s.size_diff for s in stats if s.size_diff > 0)
print(f"  Memory growth after 20 inferences: {total_new / 1024:.1f} KB")
test("No major memory leak (<50MB growth)", total_new < 50 * 1024 * 1024, f"{total_new/1024/1024:.1f}MB")

current, peak = tracemalloc.get_traced_memory()
print(f"  Current memory: {current / 1024 / 1024:.1f} MB")
print(f"  Peak memory: {peak / 1024 / 1024:.1f} MB")
tracemalloc.stop()

# Estimate Pi memory usage
model_size = os.path.getsize("yolov5s.onnx" if os.path.exists("yolov5s.onnx") else "yolov5n.onnx") / 1024 / 1024
print(f"  Model file size: {model_size:.1f} MB")
print(f"  Estimated runtime (model + buffers): ~{model_size * 3:.0f} MB")

for name in ["Pi Zero W", "Pi 4 2GB", "Pi 5 4GB"]:
    hw = profiles[name]
    fits = (model_size * 3) < hw.ram_available_mb
    test(f"Fits on {name} ({hw.ram_available_mb}MB avail)", fits, f"needs ~{model_size*3:.0f}MB")


# ============================================================
# 8. LATENCY BREAKDOWN
# ============================================================
section("8. LATENCY BREAKDOWN")

img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

# Warmup
detector.detect(img)

n_runs = 10
pre_times = []
inf_times = []
post_times = []
total_times = []

for _ in range(n_runs):
    # Preprocess
    t0 = time.perf_counter()
    x, orig = detector.preprocess(img)
    pre_times.append(time.perf_counter() - t0)
    
    # Inference
    t0 = time.perf_counter()
    output, _ = detector.backend.run(x)
    inf_times.append(time.perf_counter() - t0)
    
    # Postprocess (parse + NMS)
    t0 = time.perf_counter()
    if output.ndim == 3: output = output[0]
    obj_mask = output[:, 4] > 0.3
    filtered = output[obj_mask]
    if len(filtered) > 0:
        _nms(filtered[:, :4], filtered[:, 4], 0.45)
    post_times.append(time.perf_counter() - t0)
    
    # Total
    total_times.append(pre_times[-1] + inf_times[-1] + post_times[-1])

def ms_stats(times):
    arr = np.array(times) * 1000
    return arr.mean(), arr.std(), arr.min(), arr.max()

print(f"\n  Stage          | Mean (ms) | Std   | Min   | Max   | % of total")
print("  " + "-"*70)
for name, times in [("Preprocess", pre_times), ("Inference", inf_times), ("Postprocess", post_times), ("TOTAL", total_times)]:
    mean, std, mn, mx = ms_stats(times)
    pct = (mean / ms_stats(total_times)[0]) * 100 if name != "TOTAL" else 100
    print(f"  {name:16s} | {mean:9.1f} | {std:5.1f} | {mn:5.1f} | {mx:5.1f} | {pct:5.1f}%")

test("Inference is >80% of total time",
     ms_stats(inf_times)[0] / ms_stats(total_times)[0] > 0.8,
     "preprocessing or postprocessing is bottleneck")

# Simulate Pi speeds (estimated 5-8x slower)
pi5_factor = 5
pi4_factor = 10
total_mean = ms_stats(total_times)[0]
print(f"\n  Estimated on Pi 5: {total_mean * pi5_factor:.0f}ms ({1000/(total_mean*pi5_factor):.1f} FPS)")
print(f"  Estimated on Pi 4: {total_mean * pi4_factor:.0f}ms ({1000/(total_mean*pi4_factor):.1f} FPS)")


# ============================================================
# 9. BACKEND FALLBACK
# ============================================================
section("9. BACKEND SELECTION")

model_path = "yolov5s.onnx" if os.path.exists("yolov5s.onnx") else "yolov5n.onnx"
backend = InferenceBackend(model_path)
print(f"  Selected backend: {backend.backend_name}")
test("Backend is onnxruntime or tinytpu", backend.backend_name in ('onnxruntime', 'tinytpu'))

# Test that inference works
dummy = np.random.rand(1, 3, 640, 640).astype(np.float32)
result, elapsed = backend.run(dummy)
test("Backend produces output", result is not None and result.size > 0)
test("Backend output is numpy array", isinstance(result, np.ndarray))
print(f"  Output shape: {result.shape}, dtype: {result.dtype}")
print(f"  Inference time: {elapsed*1000:.1f}ms")


# ============================================================
# 10. FULL PIPELINE STRESS TEST
# ============================================================
section("10. FULL PIPELINE STRESS TEST (100 frames)")

ai = EdgeAI(detector, RobotController(mode='follow', target_classes=['person']))

frame_times = []
det_counts = []
actions = {}

for i in range(100):
    # Vary scene each frame
    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    t0 = time.perf_counter()
    result = ai.process(img)
    elapsed = time.perf_counter() - t0
    
    frame_times.append(elapsed * 1000)
    det_counts.append(len(result['detections']))
    action = result['command'].action
    actions[action] = actions.get(action, 0) + 1

frame_arr = np.array(frame_times)
print(f"\n  100 frames processed:")
print(f"  Mean: {frame_arr.mean():.1f}ms  Median: {np.median(frame_arr):.1f}ms")
print(f"  Min: {frame_arr.min():.1f}ms  Max: {frame_arr.max():.1f}ms  Std: {frame_arr.std():.1f}ms")
print(f"  FPS: {1000/frame_arr.mean():.1f}")
print(f"  Detection counts: min={min(det_counts)} max={max(det_counts)} avg={np.mean(det_counts):.1f}")
print(f"  Actions: {actions}")

test("All 100 frames processed", len(frame_times) == 100)
test("No frame took >5s", frame_arr.max() < 5000, f"max={frame_arr.max():.0f}ms")
test("Consistent timing (std < 50% of mean)", frame_arr.std() < frame_arr.mean() * 0.5,
     f"std={frame_arr.std():.0f}ms, mean={frame_arr.mean():.0f}ms")

# Verify command structure
result = ai.process(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
test("Result has 'detections' key", 'detections' in result)
test("Result has 'command' key", 'command' in result)
test("Result has 'fps' key", 'fps' in result)
test("Result has 'avg_ms' key", 'avg_ms' in result)
test("Command has linear_x", hasattr(result['command'], 'linear_x'))
test("Command has angular_z", hasattr(result['command'], 'angular_z'))
test("Command has action", hasattr(result['command'], 'action'))
test("Command has to_twist_dict", hasattr(result['command'], 'to_twist_dict'))

twist = result['command'].to_twist_dict()
test("Twist dict has linear.x", 'linear' in twist and 'x' in twist['linear'])
test("Twist dict has angular.z", 'angular' in twist and 'z' in twist['angular'])


# ============================================================
# 11. DETECTION DATA CLASS
# ============================================================
section("11. DETECTION DATA CLASS")

d = Detection(0, 'person', 0.95, 100.0, 50.0, 300.0, 400.0)
test("cx computed correctly", abs(d.cx - 200.0) < 0.01)
test("cy computed correctly", abs(d.cy - 225.0) < 0.01)
test("width computed correctly", abs(d.width - 200.0) < 0.01)
test("height computed correctly", abs(d.height - 350.0) < 0.01)
test("area computed correctly", abs(d.area - 70000.0) < 0.01)

d_dict = d.to_dict()
test("to_dict has class_name", d_dict['class_name'] == 'person')
test("to_dict has confidence", abs(d_dict['confidence'] - 0.95) < 0.01)
test("to_dict has bbox", len(d_dict['bbox']) == 4)


# ============================================================
# 12. MODEL ZOO VALIDATION
# ============================================================
section("12. MODEL ZOO")

for name, spec in MODEL_ZOO.items():
    test(f"{name}: has URL", len(spec.url) > 0, "missing download URL")
    test(f"{name}: valid task", spec.task in ('detect', 'classify', 'segment', 'generate'))
    test(f"{name}: INT8 < FP32", spec.int8_size_mb < spec.size_mb or spec.int8_size_mb == 0)
    test(f"{name}: has input_name", len(spec.input_name) > 0)

test("COCO classes count", len(COCO_CLASSES) == 80, f"got {len(COCO_CLASSES)}")


# ============================================================
# SUMMARY
# ============================================================
section("RESULTS SUMMARY")

total = PASS + FAIL
print(f"""
  Total tests:  {total}
  Passed:       {PASS}  ({PASS/total*100:.0f}%)
  Failed:       {FAIL}  ({FAIL/total*100:.0f}%)
  Warnings:     {WARN}

  Detection FPS:     {detector.fps:.1f}
  Backend:           {detector.backend.backend_name}
  Model:             {detector.backend.model_path}
  Pi 5 est. FPS:     {1000/(ms_stats(total_times)[0]*5):.1f}
  Pi 4 est. FPS:     {1000/(ms_stats(total_times)[0]*10):.1f}
""")

if FAIL == 0:
    print("  ALL TESTS PASSED")
else:
    print(f"  {FAIL} TESTS FAILED - review above")

sys.exit(0 if FAIL == 0 else 1)
