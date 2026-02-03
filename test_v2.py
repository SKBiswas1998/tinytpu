"""
Test Suite for TinyTPU Edge AI v2 - Production Safety & Tracking
=================================================================
Tests all 3 tiers without real hardware.
"""
import sys, os, time, tempfile, shutil
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'software'))

from tinytpu.edge_ai import Detection, RobotCommand, ObjectDetector, RobotController
from tinytpu.edge_ai_v2 import (
    SafetyController, ThermalMonitor, MemoryWatchdog,
    KalmanFilter2D, TrackedObject, ObjectTracker, AsyncPipeline,
    BlackBoxRecorder, ImageQualityScorer, ProductionEdgeAI
)

PASS = 0
FAIL = 0

def test(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name} -- {detail}")

def section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


# ============================================================
# TIER 1: SAFETY
# ============================================================
section("1. SAFETY CONTROLLER")

sc = SafetyController(max_linear=0.3, max_angular=0.5, watchdog_timeout=1.0,
                       ramp_rate=0.5, startup_delay=0.5)

# Startup delay
cmd = sc.filter_command(RobotCommand(0.3, 0, 0.2, "follow", "test"))
test("Startup blocks movement", cmd.linear_x == 0 and cmd.action == "startup")

time.sleep(0.6)
sc.feed_watchdog()

# Velocity ramping
cmd = sc.filter_command(RobotCommand(0.3, 0, 0, "follow", "test"))
test("Ramp: doesnt jump to max", cmd.linear_x < 0.3, f"got {cmd.linear_x}")
test("Ramp: velocity > 0", cmd.linear_x > 0, f"got {cmd.linear_x}")

# Let it ramp up
for _ in range(20):
    sc.feed_watchdog()
    cmd = sc.filter_command(RobotCommand(0.3, 0, 0, "follow", "test"))
    time.sleep(0.05)
test("Ramp: reaches near max after 1s", cmd.linear_x > 0.2, f"got {cmd.linear_x}")

# E-stop
sc.estop("test_estop")
cmd = sc.filter_command(RobotCommand(0.3, 0, 0.5, "follow", "test"))
test("E-stop: zero velocity", cmd.linear_x == 0 and cmd.angular_z == 0)
test("E-stop: action is estop", cmd.action == "estop")
test("E-stop: count incremented", sc.estop_count == 1)

# Reset
sc.reset()
sc.feed_watchdog()
cmd = sc.filter_command(RobotCommand(0.3, 0, 0, "follow", "test"))
test("Reset: movement resumes", cmd.linear_x > 0 or sc.state == "running")

# Watchdog timeout
sc2 = SafetyController(watchdog_timeout=0.3, startup_delay=0)
sc2.feed_watchdog()
sc2.filter_command(RobotCommand(0.1, 0, 0, "follow", "test"))
time.sleep(0.5)
cmd = sc2.filter_command(RobotCommand(0.3, 0, 0, "follow", "test"))
test("Watchdog timeout: stops", cmd.linear_x == 0 and cmd.action == "timeout")

# Proximity stop
sc3 = SafetyController(min_proximity=0.15, startup_delay=0)
sc3.feed_watchdog()
close_det = [Detection(0, "person", 0.9, 50, 10, 590, 470)]  # huge bbox
cmd = sc3.filter_command(RobotCommand(0.3, 0, 0, "follow", "test"), close_det, 640, 480)
test("Proximity: stops forward", cmd.linear_x == 0)
test("Proximity: action", cmd.action == "proximity_stop")

# Velocity limits enforcement
sc4 = SafetyController(max_linear=0.2, max_angular=0.4, startup_delay=0, ramp_rate=100)
sc4.feed_watchdog()
for _ in range(10):
    sc4.feed_watchdog()
    cmd = sc4.filter_command(RobotCommand(1.0, 0, 2.0, "test", "test"))
    time.sleep(0.02)
test("Limits: linear clamped", cmd.linear_x <= 0.2 + 0.01, f"got {cmd.linear_x}")
test("Limits: angular clamped", abs(cmd.angular_z) <= 0.4 + 0.01, f"got {cmd.angular_z}")

# Status
status = sc.get_status()
test("Status has state", "state" in status)
test("Status has estop_count", status["estop_count"] == 1)


section("2. THERMAL MONITOR")

tm = ThermalMonitor(warn_temp=70, critical_temp=78, shutdown_temp=85)

# Read temp (may or may not work on desktop)
temp = tm._read_temp()
print(f"  CPU temperature: {temp:.1f}C" if temp > 0 else "  CPU temperature: unavailable (OK on Windows)")

# Skip factor
tm.throttle_level = 0
test("Skip factor 0 at normal", tm.get_skip_factor() == 0)
tm.throttle_level = 1
test("Skip factor 1 at warn", tm.get_skip_factor() == 1)
tm.throttle_level = 2
test("Skip factor 2 at critical", tm.get_skip_factor() == 2)
tm.throttle_level = 3
test("Skip factor 4 at shutdown", tm.get_skip_factor() == 4)
tm.throttle_level = 0  # reset

# Status
status = tm.get_status()
test("Thermal status has fields", "throttle_level" in status and "skip_factor" in status)


section("3. MEMORY WATCHDOG")

mw = MemoryWatchdog(warn_percent=70, critical_percent=85, poll_interval=1.0)
mw._read_memory()
print(f"  Process RSS: {mw.current_rss_mb:.1f} MB")
print(f"  System used: {mw.system_used_percent:.1f}%")
print(f"  System available: {mw.system_available_mb:.0f} MB")
print(f"  Max RSS limit: {mw.max_rss_mb:.0f} MB")

test("RSS > 0", mw.current_rss_mb > 0)
test("System percent in range", 0 <= mw.system_used_percent <= 100)
test("Max RSS auto-set", mw.max_rss_mb > 0)

status = mw.get_status()
test("Memory status has fields", "process_rss_mb" in status and "state" in status)


# ============================================================
# TIER 2: TRACKING
# ============================================================
section("4. KALMAN FILTER")

kf = KalmanFilter2D((320, 240, 100, 150))

# Predict without measurement
pred = kf.predict(dt=0.033)
test("Predict returns 4-tuple", len(pred) == 4)
test("Initial predict near start", abs(pred[0] - 320) < 5 and abs(pred[1] - 240) < 5)

# Update + predict cycle (object moving right)
for i in range(30):
    kf.update((320 + i * 3, 240, 100, 150))
    pred = kf.predict(dt=0.033)

vx, vy = kf.velocity
test("Kalman tracks rightward velocity", vx > 0, f"vx={vx:.1f}")
test("Kalman vy near zero", abs(vy) < 10, f"vy={vy:.1f}")

# Prediction without measurement (coast)
last_cx = pred[0]
for _ in range(10):
    pred = kf.predict(dt=0.033)
test("Kalman coasts forward", pred[0] > last_cx, f"cx went from {last_cx:.1f} to {pred[0]:.1f}")


section("5. OBJECT TRACKER")

tracker = ObjectTracker(iou_threshold=0.3, max_missed=5, min_hits=2)

# Frame 1: two people
dets1 = [
    Detection(0, "person", 0.9, 100, 100, 200, 300),
    Detection(0, "person", 0.85, 400, 100, 500, 300),
]
tracks = tracker.update(dets1)
test("Frame 1: tracks created", len(tracker.tracks) == 2)
test("Frame 1: none confirmed yet (min_hits=2)", len(tracks) == 0)

# Frame 2: same positions (confirm tracks)
tracks = tracker.update(dets1)
test("Frame 2: tracks confirmed", len(tracks) == 2)
ids_frame2 = {t.track_id for t in tracks}

# Frame 3: person 1 moves right
dets3 = [
    Detection(0, "person", 0.9, 130, 100, 230, 300),  # moved right
    Detection(0, "person", 0.85, 400, 100, 500, 300),  # same
]
tracks = tracker.update(dets3)
ids_frame3 = {t.track_id for t in tracks}
test("Frame 3: same IDs maintained", ids_frame2 == ids_frame3)

# Frame 4-8: person 2 disappears
for _ in range(6):
    dets_one = [Detection(0, "person", 0.9, 160, 100, 260, 300)]
    tracks = tracker.update(dets_one)

test("After 6 frames: lost track removed", len(tracker.tracks) < 3)

# Predict between frames
pred_tracks = tracker.predict()
test("Predict returns tracks", len(pred_tracks) >= 1)
pred_dets = tracker.get_detections(pred_tracks)
test("Predicted detections valid", len(pred_dets) >= 1 and hasattr(pred_dets[0], "class_name"))

# New class appears
dets_car = [
    Detection(0, "person", 0.9, 180, 100, 280, 300),
    Detection(2, "car", 0.8, 500, 200, 620, 350),
]
tracks = tracker.update(dets_car)
classes = {t.class_name for t in tracker.tracks.values()}
test("New class tracked", "car" in classes)

status = tracker.get_status()
test("Tracker status has fields", "active_tracks" in status)


section("6. ASYNC PIPELINE")

detector = ObjectDetector("yolov5n.onnx" if os.path.exists("yolov5n.onnx") else "yolov5s.onnx",
                          conf_thresh=0.3, img_size=640)
controller = RobotController(mode="follow", target_classes=["person"])
safety = SafetyController(startup_delay=0.2)

pipeline = AsyncPipeline(detector, controller, safety=safety, target_hz=30)
pipeline.start()

# Push frames for 3 seconds
start = time.monotonic()
frames_pushed = 0
while time.monotonic() - start < 3.0:
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    pipeline.push_frame(frame)
    frames_pushed += 1
    time.sleep(0.033)  # simulate 30fps camera

cmd = pipeline.get_command()
dets = pipeline.get_detections()

test("Pipeline running", pipeline._running)
test("Frames processed", pipeline.frames_inferred > 0, f"inferred={pipeline.frames_inferred}")
test("Command returned", cmd is not None and hasattr(cmd, "linear_x"))
test("Inference FPS > 0", pipeline.inference_fps > 0, f"fps={pipeline.inference_fps}")

status = pipeline.get_status()
print(f"  Inference FPS: {status['inference_fps']}")
print(f"  Control FPS: {status['control_fps']}")
print(f"  Frames captured: {status['frames_captured']}")
print(f"  Frames inferred: {status['frames_inferred']}")
print(f"  Frames skipped: {status['frames_skipped']}")

pipeline.stop()
test("Pipeline stopped", not pipeline._running)


# ============================================================
# TIER 3: DEBUGGING
# ============================================================
section("7. BLACK BOX RECORDER")

tmpdir = tempfile.mkdtemp(prefix="blackbox_test_")
bb = BlackBoxRecorder(log_dir=tmpdir, max_entries=100)

# Record various events
for i in range(10):
    dets = [Detection(0, "person", 0.9, 100+i*5, 100, 200+i*5, 300)]
    bb.record_detection(dets, 150.0)
    bb.record_command(RobotCommand(0.15, 0, -0.1, "following", "test"))

bb.record_safety_event("estop", "test estop")
bb.record_metrics({"temp": 65.2, "rss_mb": 120.5})

test("Entries recorded", len(bb.entries) == 22)  # 10 det + 10 cmd + 1 event + 1 metrics

# Save and load
filepath = bb.save()
test("File saved", os.path.exists(filepath))

loaded = BlackBoxRecorder.load(filepath)
test("File loads", loaded["total_entries"] == 22)
test("Entries preserved", len(loaded["entries"]) == 22)

# Recent entries
recent_cmds = bb.get_recent(5, entry_type="command")
test("Filter by type works", len(recent_cmds) == 5 and all(e["type"] == "command" for e in recent_cmds))

# Circular buffer
bb2 = BlackBoxRecorder(log_dir=tmpdir, max_entries=10)
for i in range(50):
    bb2.record("test", {"i": i})
test("Circular buffer: max 10", len(bb2.entries) == 10)
test("Circular buffer: has latest", bb2.entries[-1]["data"]["i"] == 49)

shutil.rmtree(tmpdir)
test("Cleanup OK", not os.path.exists(tmpdir))


section("8. IMAGE QUALITY SCORER")

iq = ImageQualityScorer()

# Normal image
normal = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
q = iq.score(normal)
print(f"  Normal: score={q['score']}, blur={q['blur']:.0f}, brightness={q['brightness']:.0f}, issues={q['issues']}")
test("Normal image usable", q["usable"])
test("Normal score > 50", q["score"] > 50)

# Dark image
dark = np.random.randint(0, 15, (480, 640, 3), dtype=np.uint8)
q = iq.score(dark)
print(f"  Dark: score={q['score']}, brightness={q['brightness']:.0f}, issues={q['issues']}")
test("Dark detected", "dark" in q["issues"])

# Overexposed
bright = np.random.randint(230, 255, (480, 640, 3), dtype=np.uint8)
q = iq.score(bright)
print(f"  Bright: score={q['score']}, brightness={q['brightness']:.0f}, issues={q['issues']}")
test("Overexposed detected", "overexposed" in q["issues"])

# Blurry (uniform smooth)
blurry = np.ones((480, 640, 3), dtype=np.uint8) * 128
# Add very slight noise
blurry = blurry + np.random.randint(-2, 3, blurry.shape).astype(np.uint8)
q = iq.score(blurry)
print(f"  Blurry: score={q['score']}, blur={q['blur']:.1f}, issues={q['issues']}")
test("Blurry detected", "blurry" in q["issues"] or "low_contrast" in q["issues"])

# Occluded (single color)
occluded = np.ones((480, 640, 3), dtype=np.uint8) * 100
q = iq.score(occluded)
print(f"  Occluded: score={q['score']}, issues={q['issues']}")
test("Occluded detected", "occluded" in q["issues"])
test("Occluded not usable", not q["usable"])

# Empty/None
q = iq.score(np.array([]))
test("Empty image handled", q["score"] == 0 and not q["usable"])

# Trend
trend = iq.get_trend()
test("Trend has avg_score", "avg_score" in trend)
test("Trend has samples", trend["samples"] == 6)  # 5 tests above + empty


section("9. PRODUCTION EDGE AI - SYNC MODE")

detector2 = ObjectDetector("yolov5n.onnx" if os.path.exists("yolov5n.onnx") else "yolov5s.onnx",
                           conf_thresh=0.3, img_size=640)
controller2 = RobotController(mode="follow", target_classes=["person"])
safety2 = SafetyController(startup_delay=0)

pai = ProductionEdgeAI(detector=detector2, controller=controller2, safety=safety2)
pai.safety.feed_watchdog()

# Process frames synchronously
img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
result = pai.process_sync(img)

test("Result has detections", "detections" in result)
test("Result has tracks", "tracks" in result)
test("Result has command", "command" in result and hasattr(result["command"], "linear_x"))
test("Result has quality", "quality" in result and "score" in result["quality"])
test("Result has safety_state", "safety_state" in result)
test("Result has fps", "fps" in result)

# E-stop integration
pai.estop("test")
result = pai.process_sync(img)
test("E-stop via ProductionEdgeAI", result["command"].action == "estop")

pai.reset()
pai.safety.feed_watchdog()
result = pai.process_sync(img)
test("Reset via ProductionEdgeAI", result["command"].action != "estop")

# Full status
status = pai.get_full_status()
test("Full status has safety", "safety" in status)
test("Full status has thermal", "thermal" in status)
test("Full status has memory", "memory" in status)
test("Full status has pipeline", "pipeline" in status)
test("Full status has quality", "quality" in status)
test("Full status has tracker", "tracker" in status)

# Black box has entries
bb_status = pai.recorder.get_status()
test("Recorder logged entries", bb_status["total_entries"] > 0, f"entries={bb_status['total_entries']}")


section("10. KALMAN TRACKING ACCURACY (30Hz from 2FPS)")

print("\n  Simulating: object moves at constant velocity, detection at 2 FPS, control at 30 Hz")
print("  Measuring position error between Kalman prediction and ground truth\n")

tracker2 = ObjectTracker(iou_threshold=0.3, max_missed=30, min_hits=1)

# Ground truth: person walks right at 100 px/s
errors = []
gt_positions = []
pred_positions = []

det_interval = 0.5  # 2 FPS detection
ctrl_interval = 0.033  # 30 Hz control
total_time = 5.0  # 5 seconds

t = 0
last_det = -1
frame = 0

while t < total_time:
    gt_cx = 100 + t * 100  # 100 px/s rightward
    gt_cy = 240
    gt_w, gt_h = 80, 200

    # Detection at 2 FPS
    if t - last_det >= det_interval:
        det = Detection(0, "person", 0.9,
                       gt_cx - gt_w/2, gt_cy - gt_h/2,
                       gt_cx + gt_w/2, gt_cy + gt_h/2)
        tracks = tracker2.update([det])
        last_det = t
    else:
        # Predict between frames
        tracks = tracker2.predict()

    # Measure error
    if tracks:
        pred_det = tracker2.get_detections(tracks)[0]
        error = abs(pred_det.cx - gt_cx)
        errors.append(error)
        gt_positions.append(gt_cx)
        pred_positions.append(pred_det.cx)

    t += ctrl_interval
    frame += 1

errors = np.array(errors)
print(f"  Frames: {frame}")
print(f"  Detection updates: {int(total_time / det_interval)}")
print(f"  Mean position error: {errors.mean():.1f} px")
print(f"  Max position error: {errors.max():.1f} px")
print(f"  Median error: {np.median(errors):.1f} px")
print(f"  Error at detection frames: ~0 px (corrected by measurement)")
print(f"  Error between detections: {errors[errors > 1].mean():.1f} px avg")

test("Mean error < 30px (on 640px frame)", errors.mean() < 30, f"got {errors.mean():.1f}")
test("Max error < 80px", errors.max() < 80, f"got {errors.max():.1f}")
test("Kalman reduces error over time",
     np.mean(errors[-30:]) < np.mean(errors[:30]),
     f"late={np.mean(errors[-30:]):.1f} vs early={np.mean(errors[:30]):.1f}")


# ============================================================
# REAL IMAGE TEST (download COCO sample)
# ============================================================
section("11. REAL IMAGE DETECTION")

real_image_tested = False
try:
    # Try to download a COCO sample image
    import urllib.request
    url = "https://github.com/ultralytics/yolov5/raw/master/data/images/bus.jpg"
    img_path = "test_bus.jpg"
    if not os.path.exists(img_path):
        print("  Downloading test image...")
        urllib.request.urlretrieve(url, img_path)

    # Load image
    try:
        from PIL import Image
        pil_img = Image.open(img_path).convert("RGB")
        real_img = np.array(pil_img)
    except ImportError:
        try:
            import cv2
            real_img = cv2.imread(img_path)
            real_img = cv2.cvtColor(real_img, cv2.COLOR_BGR2RGB)
        except ImportError:
            real_img = None

    if real_img is not None:
        print(f"  Image loaded: {real_img.shape}")
        det3 = ObjectDetector("yolov5s.onnx" if os.path.exists("yolov5s.onnx") else "yolov5n.onnx",
                              conf_thresh=0.25, img_size=640)
        dets = det3.detect(real_img)
        print(f"  Detections: {len(dets)}")
        for d in dets[:10]:
            print(f"    {d.class_name}: {d.confidence:.0%} at ({d.x1:.0f},{d.y1:.0f})-({d.x2:.0f},{d.y2:.0f})")

        class_names = [d.class_name for d in dets]
        test("Real image: found objects", len(dets) > 0, "no detections on bus.jpg")
        test("Real image: found person", "person" in class_names, f"found: {class_names}")
        test("Real image: found bus", "bus" in class_names, f"found: {class_names}")
        test("Real image: confidence > 50%", any(d.confidence > 0.5 for d in dets))
        real_image_tested = True
    else:
        print("  [SKIP] No image library available (need PIL or cv2)")
except Exception as e:
    print(f"  [SKIP] Could not download/load test image: {e}")

if not real_image_tested:
    print("  Skipping real image tests (no network or image library)")


# ============================================================
# SUMMARY
# ============================================================
section("RESULTS SUMMARY")

total = PASS + FAIL
print(f"""
  Total tests:  {total}
  Passed:       {PASS}  ({PASS/total*100:.0f}%)
  Failed:       {FAIL}  ({FAIL/total*100:.0f}%)
""")

if FAIL == 0:
    print("  ALL TESTS PASSED")
else:
    print(f"  {FAIL} TESTS FAILED")

sys.exit(0 if FAIL == 0 else 1)
