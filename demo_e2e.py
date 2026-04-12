"""
TinyTPU End-to-End Demo — Real YOLOv8n on real images.

Run from your tinytpu repo:
    python demo_e2e.py
    python demo_e2e.py --image path/to/photo.jpg
    python demo_e2e.py --camera 0     # Live webcam
"""

import argparse
import sys
import time
import os

# ─────────────────────────────────────────────────────
# 0. Check environment
# ─────────────────────────────────────────────────────
def check_deps():
    """Check all required dependencies are installed."""
    missing = []
    for pkg, imp in [("numpy", "numpy"), ("onnxruntime", "onnxruntime"), ("opencv", "cv2"), ("pillow", "PIL")]:
        try:
            __import__(imp)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"Missing: {', '.join(missing)}")
        print(f"Install: pip install tinytpu[robotics]")
        sys.exit(1)


def banner(text):
    print(f"\n{'─'*60}")
    print(f"  {text}")
    print(f"{'─'*60}")


def main():
    parser = argparse.ArgumentParser(description="TinyTPU End-to-End Demo")
    parser.add_argument("--image", type=str, default=None, help="Path to image file")
    parser.add_argument("--camera", type=int, default=None, help="Camera index for live demo")
    parser.add_argument("--model", type=str, default="yolov8n", help="Model name (default: yolov8n)")
    parser.add_argument("--confidence", type=float, default=0.4, help="Confidence threshold")
    parser.add_argument("--no-display", action="store_true", help="Skip window display")
    args = parser.parse_args()

    check_deps()
    import numpy as np
    import cv2

    banner("TinyTPU End-to-End Demo")

    import tinytpu
    print(f"  TinyTPU v{tinytpu.__version__}")

    # ─────────────────────────────────────────────────
    # 1. Hardware detection
    # ─────────────────────────────────────────────────
    banner("1. Hardware Detection")
    hw = tinytpu.detect_hardware()
    for device in hw.devices:
        status = "✓" if device.available else "✗"
        print(f"  {status} {device.name}: {device.backend}")
    print(f"\n  Recommended: {hw.recommended} → {hw.recommended_model}")

    # ─────────────────────────────────────────────────
    # 2. Download & load model
    # ─────────────────────────────────────────────────
    banner(f"2. Loading {args.model}")
    t0 = time.perf_counter()

    # First check if ONNX exists in cache
    from tinytpu.inference.model_zoo import ModelZoo, Model
    zoo = ModelZoo()
    cached = zoo.get_model_path(args.model)

    if cached:
        print(f"  Found cached: {cached}")
    else:
        print(f"  Downloading {args.model}... (first time only)")
        # Try direct ONNX export via ultralytics
        try:
            from ultralytics import YOLO
            print(f"  Exporting {args.model} to ONNX via ultralytics...")
            yolo = YOLO(f"{args.model}.pt")
            export_path = yolo.export(format="onnx", imgsz=640)
            # Move to cache
            import shutil
            cache_dir = zoo.cache_dir / args.model
            cache_dir.mkdir(parents=True, exist_ok=True)
            dest = cache_dir / f"{args.model}.onnx"
            shutil.move(export_path, str(dest))
            print(f"  Exported to: {dest}")
        except ImportError:
            print("  ERROR: ultralytics is required to export YOLOv8 models to ONNX.")
            print("  Install it with: pip install ultralytics")
            print("  Or use a model that ships as ONNX directly: yolov5n, yolov5s, mobilenetv2")
            sys.exit(1)

    model = Model(args.model, conf_threshold=args.confidence)
    load_ms = (time.perf_counter() - t0) * 1000
    print(f"  Model ready in {load_ms:.0f}ms")

    # ─────────────────────────────────────────────────
    # 3. Run inference
    # ─────────────────────────────────────────────────
    if args.camera is not None:
        run_camera_demo(model, args)
    elif args.image:
        run_image_demo(model, args)
    else:
        run_sample_demo(model, args)

    banner("Demo Complete")


def draw_detections(img, results, show_stats=True):
    """Draw bounding boxes and labels on image."""
    import cv2

    COLORS = {
        "person": (0, 255, 0), "bicycle": (255, 165, 0), "car": (0, 0, 255),
        "motorcycle": (255, 0, 255), "bus": (0, 128, 255), "truck": (128, 0, 255),
        "dog": (255, 200, 0), "cat": (0, 255, 200), "bird": (200, 255, 0),
    }
    DEFAULT_COLOR = (255, 255, 255)

    output = img.copy()
    for det in results.detections:
        color = COLORS.get(det.class_name, DEFAULT_COLOR)
        x1, y1 = int(max(0, det.x1)), int(max(0, det.y1))
        x2, y2 = int(min(img.shape[1], det.x2)), int(min(img.shape[0], det.y2))

        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

        label = f"{det.class_name} {det.confidence:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(output, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
        cv2.putText(output, label, (x1 + 3, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    if show_stats:
        stats = f"TinyTPU | {results.elapsed_ms:.1f}ms | {len(results.detections)} objects"
        cv2.putText(output, stats, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)

    return output


def run_image_demo(model, args):
    """Run detection on a single image."""
    import cv2

    banner(f"3. Detecting: {args.image}")
    img = cv2.imread(args.image)
    if img is None:
        print(f"  ERROR: Cannot read {args.image}")
        return

    print(f"  Image: {img.shape[1]}x{img.shape[0]}")

    # Warmup
    model.predict(img)

    # Real run (average of 5)
    times = []
    for _ in range(5):
        results = model.predict(img)
        times.append(results.elapsed_ms)

    print(f"  Inference: {min(times):.1f}ms (best of 5)")
    print(f"  Detections: {len(results.detections)}")
    for det in results.detections:
        print(f"    {det.class_name}: {det.confidence:.0%} at ({det.x1:.0f},{det.y1:.0f})-({det.x2:.0f},{det.y2:.0f})")

    # Save annotated image
    output = draw_detections(img, results)
    out_path = args.image.rsplit(".", 1)[0] + "_tinytpu.jpg"
    cv2.imwrite(out_path, output)
    print(f"\n  Saved: {out_path}")

    # Display
    if not args.no_display:
        cv2.imshow("TinyTPU Detection", output)
        print("  Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Save JSON results
    json_path = args.image.rsplit(".", 1)[0] + "_results.json"
    results.save(json_path)
    print(f"  Results: {json_path}")


def run_camera_demo(model, args):
    """Live webcam detection with tracking."""
    import cv2
    from tinytpu.perception.tracker import ObjectTracker
    from tinytpu.control.safety import SafetyController

    banner(f"3. Live Camera Demo (cam {args.camera})")

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print(f"  ERROR: Cannot open camera {args.camera}")
        return

    tracker = ObjectTracker(iou_threshold=0.3, min_hits=2)
    safety = SafetyController(max_linear=0.3, max_angular=0.5, startup_delay=0.0)
    safety.feed_watchdog()

    print("  Camera opened. Press 'q' to quit, 's' to save frame.")
    print("  Showing: detections (green), tracks (cyan), follow command (yellow)")

    frame_count = 0
    fps_times = []

    while True:
        t0 = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            break

        # Detect
        results = model.predict(frame)

        # Track
        tracks = tracker.update(results.detections)

        # Generate follow command for "person"
        person_dets = [d for d in results.detections if d.class_name == "person"]
        if person_dets:
            safety.feed_watchdog()
            target = max(person_dets, key=lambda d: d.area)
            cx = (target.x1 + target.x2) / 2
            error = (cx - 320) / 320
            cmd = {"linear_x": 0.2, "angular_z": -error * 0.5}
        else:
            cmd = {"linear_x": 0.0, "angular_z": 0.0}

        safe_cmd = safety.filter_command(cmd, results.detections)

        # Draw
        output = draw_detections(frame, results)

        # Draw track IDs
        for t in tracks:
            x1, y1, x2, y2 = [int(v) for v in t.bbox_xyxy]
            cv2.putText(output, f"ID:{t.track_id}", (x1, y2 + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        # Draw motor command
        bar_x = 580
        lin_h = int(safe_cmd.linear_x / 0.3 * 50)
        ang_w = int(safe_cmd.angular_z / 0.5 * 30)
        cv2.rectangle(output, (bar_x, 240 - lin_h), (bar_x + 20, 240), (0, 255, 255), -1)
        cv2.rectangle(output, (bar_x + 10 + ang_w, 260), (bar_x + 10, 270), (0, 255, 255), -1)
        cv2.putText(output, f"L:{safe_cmd.linear_x:.2f} A:{safe_cmd.angular_z:.2f}",
                    (bar_x - 50, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)

        # FPS
        elapsed = time.perf_counter() - t0
        fps_times.append(elapsed)
        if len(fps_times) > 30:
            fps_times = fps_times[-30:]
        fps = 1.0 / max(0.001, sum(fps_times) / len(fps_times))
        cv2.putText(output, f"FPS: {fps:.1f} | Tracks: {len(tracks)}",
                    (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        cv2.imshow("TinyTPU Live", output)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            save_path = f"capture_{frame_count:04d}.jpg"
            cv2.imwrite(save_path, output)
            print(f"  Saved {save_path}")

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n  Processed {frame_count} frames")


def run_sample_demo(model, args):
    """No image provided — create a sample and detect."""
    import cv2
    import numpy as np

    banner("3. Sample Image Detection")
    print("  No image specified. Generating sample scene...")
    print("  Tip: python demo_e2e.py --image photo.jpg")
    print("       python demo_e2e.py --camera 0\n")

    # Generate a test scene
    img = np.zeros((480, 640, 3), dtype=np.uint8)

    # Background
    for y in range(240):
        v = int(180 - y * 0.3)
        img[y, :] = [v + 40, v + 20, v]
    img[240:, :] = [50, 90, 50]

    # Road
    cv2.rectangle(img, (180, 240), (460, 480), (70, 70, 70), -1)

    # Person
    cv2.rectangle(img, (270, 100), (370, 380), (60, 60, 200), -1)
    cv2.circle(img, (320, 80), 30, (60, 60, 200), -1)

    # Car
    cv2.rectangle(img, (440, 270), (620, 400), (200, 60, 60), -1)

    cv2.imwrite("sample_scene.jpg", img)

    # Run detection
    results = model.predict(img)
    print(f"  Inference: {results.elapsed_ms:.1f}ms")
    print(f"  Detections: {len(results.detections)}")
    for det in results.detections:
        print(f"    {det.class_name}: {det.confidence:.0%}")

    # Save output
    output = draw_detections(img, results)
    cv2.imwrite("sample_output.jpg", output)
    print(f"\n  Saved: sample_output.jpg")

    if not args.no_display:
        try:
            cv2.imshow("TinyTPU Sample", output)
            print("  Press any key to close...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except cv2.error:
            pass

    # ─────────────────────────────────────────────────
    # 4. Full pipeline test
    # ─────────────────────────────────────────────────
    banner("4. Full Pipeline Test (10 simulated frames)")
    from tinytpu.perception.tracker import ObjectTracker
    from tinytpu.control.safety import SafetyController

    tracker = ObjectTracker(min_hits=2)
    safety = SafetyController(startup_delay=0.0)
    safety.feed_watchdog()

    for i in range(10):
        # Shift person rightward each frame
        shifted = img.copy()
        offset = i * 8
        shifted[:, offset:] = img[:, :640-offset]

        results = model.predict(shifted)
        tracks = tracker.update(results.detections)

        # Follow logic
        persons = [d for d in results.detections if d.class_name == "person"]
        if persons:
            safety.feed_watchdog()
            target = persons[0]
            cx = (target.x1 + target.x2) / 2
            error = (cx - 320) / 320
            cmd = {"linear_x": 0.2, "angular_z": -error * 0.5}
        else:
            cmd = {"linear_x": 0.0, "angular_z": 0.0}

        import time as t; t.sleep(0.01)
        safe = safety.filter_command(cmd, results.detections)
        print(f"  Frame {i}: {len(results.detections)} dets, {len(tracks)} tracks, "
              f"cmd=({safe.linear_x:.2f}, {safe.angular_z:.2f})")

    # ─────────────────────────────────────────────────
    # 5. Benchmark
    # ─────────────────────────────────────────────────
    banner("5. Inference Benchmark")
    times = []
    for _ in range(50):
        results = model.predict(img)
        times.append(results.elapsed_ms)

    import numpy as np
    times = np.array(times)
    print(f"  Model: {model.name}")
    print(f"  Backend: {model._backend_name}")
    print(f"  Runs: {len(times)}")
    print(f"  Mean: {times.mean():.1f}ms ({1000/times.mean():.0f} FPS)")
    print(f"  Min:  {times.min():.1f}ms ({1000/times.min():.0f} FPS)")
    print(f"  Max:  {times.max():.1f}ms")
    print(f"  Std:  {times.std():.1f}ms")


if __name__ == "__main__":
    main()
