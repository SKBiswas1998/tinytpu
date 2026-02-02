"""
TinyTPU ONNX - Real Image Classification
"""

import numpy as np
import sys
import os
from PIL import Image

sys.path.insert(0, 'software')
from tinytpu.onnx_engine import TinyTPUEngine

print("=" * 70)
print("REAL IMAGE CLASSIFICATION")
print("=" * 70)

# Create a synthetic test image (colored patterns that trigger classifiers)
print("Creating test images...")

# Test 1: Red/orange pattern (often triggers fire truck, sports car)
img1 = np.zeros((224, 224, 3), dtype=np.uint8)
img1[:, :, 0] = 200  # Red
img1[:, :, 1] = 50
img1[50:180, 30:200, :] = [255, 140, 0]  # Orange rectangle

# Test 2: Green pattern (often triggers nature/plant classes)
img2 = np.zeros((224, 224, 3), dtype=np.uint8)
img2[:, :, 1] = 150  # Green
img2[20:200, 20:200, 1] = 200

# Test 3: Striped pattern (often triggers fabric/pattern classes)
img3 = np.zeros((224, 224, 3), dtype=np.uint8)
for i in range(0, 224, 20):
    img3[i:i+10, :, :] = [255, 255, 255]

def preprocess(img_array):
    x = img_array.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x = (x - mean) / std
    x = x.transpose(2, 0, 1)
    return x[np.newaxis, :, :, :].astype(np.float32)

# Load model
engine = TinyTPUEngine("mobilenetv2.onnx")

with open("imagenet_classes.txt") as f:
    labels = [line.strip() for line in f.readlines()]

# Test each image
tests = [
    ("Red/Orange pattern", img1),
    ("Green pattern", img2),
    ("Striped pattern", img3),
]

for name, img in tests:
    x = preprocess(img)
    output, elapsed = engine.run({"input": x})
    
    out_key = list(output.keys())[0]
    logits = output[out_key].squeeze()
    exp_l = np.exp(logits - logits.max())
    probs = exp_l / exp_l.sum()
    
    top5 = np.argsort(probs)[-5:][::-1]
    
    print(f"\n[{name}] ({elapsed*1000:.1f}ms)")
    for i, idx in enumerate(top5):
        label = labels[idx] if idx < len(labels) else f"class_{idx}"
        bar = "#" * int(probs[idx] * 40)
        print(f"  {i+1}. {label}: {probs[idx]*100:.1f}% {bar}")

# Now test with a real photo if user has one
print("\n" + "=" * 70)
print("To classify your own image:")
print('  from tinytpu.onnx_engine import TinyTPUEngine')
print('  from PIL import Image')
print('  import numpy as np')
print('')
print('  engine = TinyTPUEngine("mobilenetv2.onnx")')
print('  img = Image.open("your_photo.jpg").resize((224, 224))')
print('  x = np.array(img).astype(np.float32) / 255.0')
print('  x = (x - [0.485,0.456,0.406]) / [0.229,0.224,0.225]')
print('  x = x.transpose(2,0,1)[np.newaxis].astype(np.float32)')
print('  output, _ = engine.run({"input": x})')
print("=" * 70)

# ONNX Runtime comparison
print("\n[Accuracy check vs ONNX Runtime]")
try:
    import onnxruntime as ort
    session = ort.InferenceSession("mobilenetv2.onnx")
    
    x = preprocess(img1)
    tiny_out = engine.run({"input": x})[0]
    ort_out = session.run(None, {"input": x})[0]
    
    tiny_logits = list(tiny_out.values())[0].squeeze()
    ort_logits = ort_out.squeeze()
    
    corr = np.corrcoef(tiny_logits, ort_logits)[0,1]
    max_diff = np.max(np.abs(tiny_logits - ort_logits))
    
    print(f"  Correlation: {corr:.6f}")
    print(f"  Max difference: {max_diff:.4f}")
    print(f"  Match: {'YES' if corr > 0.99 else 'CLOSE' if corr > 0.95 else 'NO'}")
except ImportError:
    print("  ONNX Runtime not installed")

print("\nDone!")
