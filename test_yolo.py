"""
TinyTPU ONNX - YOLOv5 Object Detection (Fixed)
"""

import numpy as np
import sys
import os
import time
from PIL import Image, ImageDraw

sys.path.insert(0, 'software')
from tinytpu.onnx_engine import TinyTPUEngine

print("=" * 70)
print("TINYTPU - YOLO OBJECT DETECTION")
print("=" * 70)

COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def nms(boxes, scores, iou_threshold=0.45):
    """Non-maximum suppression."""
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h
        
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_j = (boxes[order[1:], 2] - boxes[order[1:], 0]) * (boxes[order[1:], 3] - boxes[order[1:], 1])
        iou = inter / (area_i + area_j - inter + 1e-6)
        
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    
    return np.array(keep)

# Load model
engine = TinyTPUEngine("yolov5n.onnx")

# Create test scene
print("\n[Creating test scene]")
img = Image.new('RGB', (640, 640), (200, 200, 200))
draw = ImageDraw.Draw(img)
draw.rectangle([50, 100, 200, 400], fill=(50, 50, 150))
draw.rectangle([300, 200, 500, 350], fill=(200, 50, 50))
draw.rectangle([400, 50, 550, 150], fill=(50, 150, 50))
draw.ellipse([100, 450, 200, 550], fill=(255, 200, 0))
img.save("test_scene.png")

x = np.array(img).astype(np.float32) / 255.0
x = x.transpose(2, 0, 1)[np.newaxis]

# TinyTPU inference
print("\n[TinyTPU Inference]")
output, elapsed = engine.run({"images": x})
out = list(output.values())[0]
print(f"  Time: {elapsed*1000:.1f}ms")
print(f"  Output shape: {out.shape}")

# Parse YOLO output
detections = out[0]  # [25200, 85]
obj_scores = detections[:, 4]
mask = obj_scores > 0.25
filtered = detections[mask]

if len(filtered) > 0:
    # Get boxes (cx, cy, w, h -> x1, y1, x2, y2)
    cx, cy, w, h = filtered[:, 0], filtered[:, 1], filtered[:, 2], filtered[:, 3]
    boxes = np.stack([cx - w/2, cy - h/2, cx + w/2, cy + h/2], axis=1)
    
    class_scores = filtered[:, 5:]
    class_ids = np.argmax(class_scores, axis=1)
    class_probs = np.max(class_scores, axis=1)
    confidences = filtered[:, 4] * class_probs
    
    # NMS per class
    final_boxes = []
    final_scores = []
    final_classes = []
    
    for cls_id in np.unique(class_ids):
        cls_mask = class_ids == cls_id
        cls_boxes = boxes[cls_mask]
        cls_scores_arr = confidences[cls_mask]
        
        keep = nms(cls_boxes, cls_scores_arr)
        final_boxes.extend(cls_boxes[keep])
        final_scores.extend(cls_scores_arr[keep])
        final_classes.extend([cls_id] * len(keep))
    
    # Sort by confidence
    order = np.argsort(final_scores)[::-1]
    
    print(f"\n  Detections after NMS: {len(order)}")
    for i in order[:10]:
        box = final_boxes[i]
        conf = final_scores[i]
        cls_name = COCO_CLASSES[final_classes[i]]
        print(f"    {cls_name}: {conf*100:.1f}% at [{box[0]:.0f},{box[1]:.0f},{box[2]:.0f},{box[3]:.0f}]")
    
    # Draw detections on image
    img_out = img.copy()
    draw = ImageDraw.Draw(img_out)
    colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255)]
    
    for i in order[:10]:
        box = final_boxes[i]
        conf = final_scores[i]
        cls_name = COCO_CLASSES[final_classes[i]]
        color = colors[final_classes[i] % len(colors)]
        
        draw.rectangle([box[0], box[1], box[2], box[3]], outline=color, width=3)
        draw.text((box[0], box[1]-12), f"{cls_name} {conf*100:.0f}%", fill=color)
    
    img_out.save("test_scene_detected.png")
    print("\n  Saved: test_scene_detected.png")
else:
    print("  No detections")

# Benchmark
print("\n[Benchmark]")
stats = engine.benchmark({"images": x}, runs=5)
print(f"  Mean: {stats['mean_ms']:.1f}ms")
print(f"  FPS: {stats['fps']:.1f}")

# ONNX Runtime comparison (with float16 input)
print("\n[ONNX Runtime comparison]")
try:
    import onnxruntime as ort
    session = ort.InferenceSession("yolov5n.onnx")
    inp = session.get_inputs()[0]
    print(f"  Expected input: {inp.name}, type={inp.type}, shape={inp.shape}")
    
    x_ort = x.astype(np.float16) if 'float16' in inp.type else x
    
    times = []
    for _ in range(5):
        start = time.perf_counter()
        session.run(None, {inp.name: x_ort})
        times.append(time.perf_counter() - start)
    
    ort_ms = np.median(times) * 1000
    print(f"  ONNX Runtime: {ort_ms:.1f}ms ({1000/ort_ms:.1f} FPS)")
    print(f"  TinyTPU: {stats['mean_ms']:.1f}ms ({stats['fps']:.1f} FPS)")
    print(f"  Ratio: {stats['mean_ms']/ort_ms:.1f}x")
    
    # Compare outputs
    ort_out = session.run(None, {inp.name: x_ort})[0]
    if ort_out.dtype == np.float16:
        ort_out = ort_out.astype(np.float32)
    tiny_out = list(output.values())[0]
    
    corr = np.corrcoef(tiny_out.flatten()[:1000], ort_out.flatten()[:1000])[0,1]
    print(f"  Output correlation: {corr:.6f}")
except Exception as e:
    print(f"  Error: {e}")

print("\n" + "=" * 70)
print("YOLO OBJECT DETECTION COMPLETE!")
print("=" * 70)
