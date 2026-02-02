"""
Test TinyTPU ONNX Engine with MobileNetV2
==========================================
Download and run a real image classification model.
"""

import numpy as np
import time
import sys
import os
import urllib.request

sys.path.insert(0, 'software')
from tinytpu.onnx_engine import TinyTPUEngine

print("=" * 70)
print("TINYTPU ONNX ENGINE - MobileNetV2 Test")
print("=" * 70)

# Download MobileNetV2 ONNX model
MODEL_URL = "https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-12.onnx"
MODEL_PATH = "mobilenetv2.onnx"
LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
LABELS_PATH = "imagenet_classes.txt"

if not os.path.exists(MODEL_PATH):
    print(f"\nDownloading MobileNetV2...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("  Done!")

if not os.path.exists(LABELS_PATH):
    print("Downloading ImageNet labels...")
    urllib.request.urlretrieve(LABELS_URL, LABELS_PATH)
    print("  Done!")

# Load labels
with open(LABELS_PATH) as f:
    labels = [line.strip() for line in f.readlines()]

# Load model
print("\n--- FP32 Model ---")
engine_fp32 = TinyTPUEngine(MODEL_PATH, quantize=False)

print("\n--- INT8 Model ---")
engine_int8 = TinyTPUEngine(MODEL_PATH, quantize=True)

# Create test input (224x224 RGB image)
# Simulate a "cat" image with noise
np.random.seed(42)
dummy_image = np.random.randn(1, 3, 224, 224).astype(np.float32) * 0.5

# Run FP32
print("\n" + "=" * 70)
print("INFERENCE TEST")
print("=" * 70)

print("\n[FP32 Inference]")
output_fp32, elapsed_fp32 = engine_fp32.run({"input": dummy_image})
print(f"  Time: {elapsed_fp32*1000:.1f}ms")

if output_fp32:
    out_key = list(output_fp32.keys())[0]
    probs = output_fp32[out_key]
    if probs.ndim > 1:
        probs = probs.squeeze()
    
    # Softmax if needed
    if probs.max() > 1.0:
        exp_p = np.exp(probs - probs.max())
        probs = exp_p / exp_p.sum()
    
    top5 = np.argsort(probs)[-5:][::-1]
    print("  Top-5 predictions:")
    for i, idx in enumerate(top5):
        label = labels[idx] if idx < len(labels) else f"class_{idx}"
        print(f"    {i+1}. {label}: {probs[idx]*100:.1f}%")

# Run INT8
print("\n[INT8 Inference]")
output_int8, elapsed_int8 = engine_int8.run({"input": dummy_image})
print(f"  Time: {elapsed_int8*1000:.1f}ms")

if output_int8:
    out_key = list(output_int8.keys())[0]
    probs = output_int8[out_key]
    if probs.ndim > 1:
        probs = probs.squeeze()
    
    if probs.max() > 1.0:
        exp_p = np.exp(probs - probs.max())
        probs = exp_p / exp_p.sum()
    
    top5 = np.argsort(probs)[-5:][::-1]
    print("  Top-5 predictions:")
    for i, idx in enumerate(top5):
        label = labels[idx] if idx < len(labels) else f"class_{idx}"
        print(f"    {i+1}. {label}: {probs[idx]*100:.1f}%")

# Benchmark
print("\n" + "=" * 70)
print("BENCHMARK")
print("=" * 70)

print("\n[FP32]")
stats_fp32 = engine_fp32.benchmark({"input": dummy_image}, runs=10)
print(f"  Mean: {stats_fp32['mean_ms']:.1f}ms")
print(f"  FPS: {stats_fp32['fps']:.1f}")

print("\n[INT8]")
stats_int8 = engine_int8.benchmark({"input": dummy_image}, runs=10)
print(f"  Mean: {stats_int8['mean_ms']:.1f}ms")
print(f"  FPS: {stats_int8['fps']:.1f}")

# Compare with ONNX Runtime
print("\n[ONNX Runtime (reference)]")
try:
    import onnxruntime as ort
    session = ort.InferenceSession(MODEL_PATH)
    
    # Warmup
    for _ in range(3):
        session.run(None, {"input": dummy_image})
    
    times = []
    for _ in range(10):
        start = time.perf_counter()
        session.run(None, {"input": dummy_image})
        times.append(time.perf_counter() - start)
    
    ort_ms = np.median(times) * 1000
    ort_fps = 1.0 / np.mean(times)
    print(f"  Mean: {ort_ms:.1f}ms")
    print(f"  FPS: {ort_fps:.1f}")
    
    print(f"\n  TinyTPU vs ONNX Runtime: {stats_fp32['mean_ms']/ort_ms:.2f}x")
except ImportError:
    print("  Not installed (pip install onnxruntime)")

print("\n" + "=" * 70)
print("ONNX ENGINE TEST COMPLETE!")
print("=" * 70)
