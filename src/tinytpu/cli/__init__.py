"""TinyTPU CLI - Command-line tools for edge AI."""

import argparse
import sys
import time
import platform


def cmd_version(args):
    from tinytpu import __version__
    print(f"TinyTPU v{__version__}")
    print(f"Python {platform.python_version()}")
    print(f"Platform: {platform.system()} {platform.machine()}")
    deps = {"numpy": "numpy", "onnx": "onnx", "onnxruntime": "onnxruntime",
            "opencv": "cv2", "pillow": "PIL", "ultralytics": "ultralytics"}
    print("\nDependencies:")
    for name, module in deps.items():
        try:
            mod = __import__(module)
            ver = getattr(mod, "__version__", "installed")
            print(f"  + {name} {ver}")
        except ImportError:
            print(f"  - {name} (not installed)")


def cmd_hardware(args):
    print("Scanning for AI accelerators...\n")
    from tinytpu.hal.detect import detect_hardware
    hw = detect_hardware()
    print(f"{'Device':<30} {'Status':<12} {'TOPS':<15} {'Notes'}")
    print("-" * 75)
    for device in hw.devices:
        status = "+ Ready" if device.available else "- Missing"
        notes = device.notes or ""
        print(f"  {device.name:<28} {status:<12} {device.tops or '-':<15} {notes[:30]}")
    print(f"\nRecommended backend: {hw.recommended}")
    print(f"Recommended model:   {hw.recommended_model}")
    if hw.warnings:
        print(f"\nWarnings:")
        for w in hw.warnings:
            print(f"  ! {w}")


def cmd_detect(args):
    """Run object detection on an image."""
    import os
    if not os.path.exists(args.image):
        print(f"Error: File not found: {args.image}")
        sys.exit(1)

    try:
        import cv2
    except ImportError:
        print("OpenCV required. Install: pip install tinytpu[vision]")
        sys.exit(1)

    print(f"Loading {args.model}...")
    from tinytpu.inference.model_zoo import Model
    model = Model(args.model, conf_threshold=args.confidence)

    print(f"Reading {args.image}...")
    img = cv2.imread(args.image)
    if img is None:
        print(f"Error: Cannot read image: {args.image}")
        sys.exit(1)

    print(f"Image: {img.shape[1]}x{img.shape[0]}")

    # Warmup + inference
    model.predict(img)
    results = model.predict(img)

    print(f"\nInference: {results.elapsed_ms:.1f}ms")
    print(f"Detections: {len(results.detections)}\n")

    if not results.detections:
        print("No objects detected.")
        return

    print(f"{'Class':<15} {'Conf':<8} {'Box'}")
    print("-" * 50)
    for det in results.detections:
        print(f"  {det.class_name:<13} {det.confidence:.0%}    "
              f"({det.x1:.0f},{det.y1:.0f})->({det.x2:.0f},{det.y2:.0f})")

    # Save annotated output
    if args.output:
        out_path = args.output
    else:
        base, ext = os.path.splitext(args.image)
        out_path = f"{base}_detected{ext}"

    output_img = img.copy()
    COLORS = {
        "person": (0, 255, 0), "car": (0, 0, 255), "bicycle": (255, 165, 0),
        "dog": (255, 200, 0), "cat": (0, 255, 200), "truck": (128, 0, 255),
        "bus": (0, 128, 255), "motorcycle": (255, 0, 255),
    }
    for det in results.detections:
        color = COLORS.get(det.class_name, (255, 255, 255))
        x1, y1 = int(max(0, det.x1)), int(max(0, det.y1))
        x2, y2 = int(min(img.shape[1], det.x2)), int(min(img.shape[0], det.y2))
        cv2.rectangle(output_img, (x1, y1), (x2, y2), color, 2)
        label = f"{det.class_name} {det.confidence:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(output_img, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
        cv2.putText(output_img, label, (x1 + 3, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(output_img, f"TinyTPU | {results.elapsed_ms:.1f}ms | {len(results.detections)} objects",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(out_path, output_img)
    print(f"\nSaved: {out_path}")

    # Save JSON
    json_path = os.path.splitext(out_path)[0] + ".json"
    results.save(json_path)
    print(f"JSON:  {json_path}")


def cmd_benchmark(args):
    import numpy as np
    print(f"Benchmarking ({args.runs} runs)...\n")
    results = run_benchmark(runs=args.runs)

    print(f"{'Test':<25} {'Mean':<12} {'Performance'}")
    print("-" * 55)
    for key, val in results["tests"].items():
        perf = f"({val['gflops']:.1f} GFLOPS)" if 'gflops' in val else ""
        print(f"  {key:<23} {val['mean_ms']:.2f}ms    {perf}")

    print(f"\nPlatform: {results['platform']}")

    # If onnxruntime available, also benchmark inference
    try:
        from tinytpu.inference.model_zoo import ModelZoo
        zoo = ModelZoo()
        cached = zoo.get_model_path("yolov8n")
        if cached:
            print(f"\nInference benchmark (yolov8n):")
            from tinytpu.inference.model_zoo import Model
            model = Model("yolov8n")
            img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            model.predict(img)  # warmup
            times = []
            for _ in range(args.runs):
                r = model.predict(img)
                times.append(r.elapsed_ms)
            times = np.array(times)
            print(f"  Mean: {times.mean():.1f}ms ({1000/times.mean():.0f} FPS)")
            print(f"  Min:  {times.min():.1f}ms ({1000/times.min():.0f} FPS)")
    except Exception:
        pass


def cmd_models(args):
    from tinytpu.inference.model_zoo import MODEL_REGISTRY, ModelZoo
    zoo = ModelZoo()
    downloaded = set(zoo.list_downloaded())

    print(f"{'Model':<22} {'Task':<15} {'Size':<10} {'Cached'}")
    print("-" * 60)
    for name, info in MODEL_REGISTRY.items():
        cached = "✓ yes" if name in downloaded else "  no"
        print(f"  {name:<20} {info['task']:<15} {info['size']:<10} {cached}")
    print(f"\nTotal: {len(MODEL_REGISTRY)} models, {len(downloaded)} cached")
    print(f"Cache: {zoo.cache_dir}")


def cmd_download(args):
    from tinytpu.inference.model_zoo import ModelZoo
    zoo = ModelZoo()
    print(f"Downloading {args.model}...")
    path = zoo.download(args.model, force=args.force)
    print(f"Ready: {path}")


def cmd_backends(args):
    from tinytpu.hal.backends import list_backends
    print(list_backends())


def run_benchmark(model_name="yolov8n", runs=20):
    import numpy as np
    results = {"model": model_name, "runs": runs, "platform": platform.machine(), "tests": {}}

    sizes = [64, 128, 256, 512]
    for size in sizes:
        a = np.random.randn(size, size).astype(np.float32)
        b = np.random.randn(size, size).astype(np.float32)
        np.matmul(a, b)  # warmup
        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            np.matmul(a, b)
            times.append(time.perf_counter() - t0)
        mean_s = np.mean(times)
        gflops = (2 * size**3) / (mean_s * 1e9)
        results["tests"][f"matmul_{size}x{size}"] = {
            "mean_ms": round(mean_s * 1000, 2), "gflops": round(gflops, 1)
        }

    x = np.random.randn(1000, 768).astype(np.float32)
    activations = [
        ("relu", lambda x: np.maximum(x, 0)),
        ("sigmoid", lambda x: 1 / (1 + np.exp(-np.clip(x, -500, 500)))),
        ("softmax", lambda x: np.exp(x - x.max(axis=-1, keepdims=True)) /
                               np.exp(x - x.max(axis=-1, keepdims=True)).sum(axis=-1, keepdims=True)),
    ]
    for name, fn in activations:
        fn(x)
        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            fn(x)
            times.append(time.perf_counter() - t0)
        results["tests"][name] = {"mean_ms": round(np.mean(times) * 1000, 3)}

    # INT8 quantization speed
    a = np.random.randn(512, 512).astype(np.float32)
    t0 = time.perf_counter()
    for _ in range(runs):
        scale = np.max(np.abs(a)) / 127
        q = np.clip(np.round(a / scale), -128, 127).astype(np.int8)
    quant_ms = (time.perf_counter() - t0) / runs * 1000
    results["tests"]["int8_quantize_512"] = {"mean_ms": round(quant_ms, 2)}

    return results


def main():
    parser = argparse.ArgumentParser(
        prog="tinytpu",
        description="TinyTPU — Production Edge AI for Robots",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  tinytpu version                    Show version and dependencies
  tinytpu hardware                   Detect AI accelerators
  tinytpu detect photo.jpg           Run object detection
  tinytpu detect photo.jpg -m yolov8s -c 0.5
  tinytpu benchmark                  Benchmark your hardware
  tinytpu models                     List available models
  tinytpu download yolov8n           Download model for offline use
  tinytpu backends                   List inference backends
""",
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("version", help="Show version and dependency info")
    sub.add_parser("hardware", help="Detect available AI accelerators")

    p_detect = sub.add_parser("detect", help="Run object detection on an image")
    p_detect.add_argument("image", help="Path to image file")
    p_detect.add_argument("-m", "--model", default="yolov8n", help="Model name (default: yolov8n)")
    p_detect.add_argument("-c", "--confidence", type=float, default=0.4, help="Confidence threshold")
    p_detect.add_argument("-o", "--output", type=str, default=None, help="Output image path")

    p_bench = sub.add_parser("benchmark", help="Benchmark inference speed")
    p_bench.add_argument("--runs", type=int, default=20, help="Number of runs")

    sub.add_parser("models", help="List available models and cache status")

    p_dl = sub.add_parser("download", help="Download a model for offline use")
    p_dl.add_argument("model", help="Model name to download")
    p_dl.add_argument("--force", action="store_true", help="Re-download even if cached")

    sub.add_parser("backends", help="List inference backends")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    cmds = {
        "version": cmd_version,
        "hardware": cmd_hardware,
        "detect": cmd_detect,
        "benchmark": cmd_benchmark,
        "models": cmd_models,
        "download": cmd_download,
        "backends": cmd_backends,
    }
    func = cmds.get(args.command)
    if func:
        func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
