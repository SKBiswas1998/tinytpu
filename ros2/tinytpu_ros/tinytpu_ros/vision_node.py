"""
TinyTPU Vision Node for ROS2
=============================
Camera -> YOLO Object Detection -> /detections

Runs on Raspberry Pi without GPU.
Works with any USB camera or Pi Camera.
"""

import numpy as np
import time
import json
import os
import sys

# ROS2 imports (graceful fallback for testing without ROS)
try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
    from sensor_msgs.msg import Image
    HAS_ROS = True
except ImportError:
    HAS_ROS = False
    print("[VisionNode] ROS2 not found - running in standalone mode")

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("[VisionNode] OpenCV not found - install with: pip install opencv-python")

# Add TinyTPU to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'software'))
try:
    from tinytpu.onnx_engine import TinyTPUEngine
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'software'))
    from tinytpu.onnx_engine import TinyTPUEngine


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
    if len(boxes) == 0:
        return np.array([], dtype=int)
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


class VisionProcessor:
    """
    Standalone vision processor (works with or without ROS).
    
    Usage:
        vision = VisionProcessor("yolov5n.onnx")
        detections = vision.detect(image_numpy)
    """
    
    def __init__(self, model_path="yolov5n.onnx", conf_thresh=0.4, nms_thresh=0.45, 
                 img_size=640, quantize=False):
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.img_size = img_size
        
        print(f"[Vision] Loading model: {model_path}")
        self.engine = TinyTPUEngine(model_path, quantize=quantize)
        print(f"[Vision] Ready! (conf={conf_thresh}, nms={nms_thresh})")
    
    def preprocess(self, image):
        """Preprocess image for YOLO."""
        if HAS_CV2:
            img = cv2.resize(image, (self.img_size, self.img_size))
        else:
            from PIL import Image
            img = np.array(Image.fromarray(image).resize((self.img_size, self.img_size)))
        
        x = img.astype(np.float32) / 255.0
        x = x.transpose(2, 0, 1)  # HWC -> CHW
        x = x[np.newaxis]          # Add batch
        return x, image.shape[:2]  # Return original size for box scaling
    
    def postprocess(self, output, original_size):
        """Parse YOLO output into detections."""
        detections = output[0]  # [25200, 85]
        
        # Filter by objectness
        obj_mask = detections[:, 4] > self.conf_thresh
        filtered = detections[obj_mask]
        
        if len(filtered) == 0:
            return []
        
        # Get boxes, scores, classes
        cx, cy, w, h = filtered[:, 0], filtered[:, 1], filtered[:, 2], filtered[:, 3]
        boxes = np.stack([cx - w/2, cy - h/2, cx + w/2, cy + h/2], axis=1)
        
        class_scores = filtered[:, 5:]
        class_ids = np.argmax(class_scores, axis=1)
        class_probs = np.max(class_scores, axis=1)
        confidences = filtered[:, 4] * class_probs
        
        # NMS per class
        final = []
        for cls_id in np.unique(class_ids):
            cls_mask = class_ids == cls_id
            cls_boxes = boxes[cls_mask]
            cls_scores = confidences[cls_mask]
            keep = nms(cls_boxes, cls_scores, self.nms_thresh)
            
            for k in keep:
                # Scale boxes to original image size
                scale_y = original_size[0] / self.img_size
                scale_x = original_size[1] / self.img_size
                box = cls_boxes[k]
                
                final.append({
                    'class_id': int(cls_id),
                    'class_name': COCO_CLASSES[int(cls_id)] if int(cls_id) < len(COCO_CLASSES) else f'class_{cls_id}',
                    'confidence': float(cls_scores[k]),
                    'box': {
                        'x1': float(box[0] * scale_x),
                        'y1': float(box[1] * scale_y),
                        'x2': float(box[2] * scale_x),
                        'y2': float(box[3] * scale_y),
                        'cx': float((box[0] + box[2]) / 2 * scale_x),
                        'cy': float((box[1] + box[3]) / 2 * scale_y),
                        'width': float((box[2] - box[0]) * scale_x),
                        'height': float((box[3] - box[1]) * scale_y),
                    }
                })
        
        # Sort by confidence
        final.sort(key=lambda x: x['confidence'], reverse=True)
        return final
    
    def detect(self, image):
        """
        Run detection on an image.
        
        Args:
            image: numpy array (H, W, 3) BGR or RGB
            
        Returns:
            list of detections with class, confidence, box
        """
        x, orig_size = self.preprocess(image)
        output, elapsed = self.engine.run({"images": x})
        out = list(output.values())[0]
        detections = self.postprocess(out, orig_size)
        
        return detections, elapsed
    
    def detect_and_draw(self, image):
        """Detect and draw bounding boxes on image."""
        detections, elapsed = self.detect(image)
        
        annotated = image.copy()
        colors = [(0,255,0), (255,0,0), (0,0,255), (255,255,0), (0,255,255)]
        
        if HAS_CV2:
            for det in detections:
                box = det['box']
                color = colors[det['class_id'] % len(colors)]
                x1, y1 = int(box['x1']), int(box['y1'])
                x2, y2 = int(box['x2']), int(box['y2'])
                
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                label = f"{det['class_name']} {det['confidence']*100:.0f}%"
                cv2.putText(annotated, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return annotated, detections, elapsed


# ============================================================
# ROS2 NODE
# ============================================================

if HAS_ROS:
    class VisionNode(Node):
        """
        ROS2 node for TinyTPU object detection.
        
        Subscribes: /camera/image_raw (sensor_msgs/Image)
        Publishes:  /tinytpu/detections (std_msgs/String - JSON)
                    /tinytpu/annotated_image (sensor_msgs/Image)
        """
        
        def __init__(self):
            super().__init__('tinytpu_vision')
            
            # Parameters
            self.declare_parameter('model', 'yolov5n.onnx')
            self.declare_parameter('confidence_threshold', 0.4)
            self.declare_parameter('nms_threshold', 0.45)
            self.declare_parameter('image_size', 640)
            self.declare_parameter('quantize', False)
            self.declare_parameter('input_topic', '/camera/image_raw')
            self.declare_parameter('rate', 10.0)
            
            model = self.get_parameter('model').value
            conf = self.get_parameter('confidence_threshold').value
            nms_t = self.get_parameter('nms_threshold').value
            img_size = self.get_parameter('image_size').value
            quantize = self.get_parameter('quantize').value
            
            # Initialize vision processor
            self.vision = VisionProcessor(model, conf, nms_t, img_size, quantize)
            
            # Publishers
            self.det_pub = self.create_publisher(String, '/tinytpu/detections', 10)
            self.img_pub = self.create_publisher(Image, '/tinytpu/annotated_image', 10)
            
            # Subscriber
            input_topic = self.get_parameter('input_topic').value
            self.sub = self.create_subscription(Image, input_topic, self.image_callback, 10)
            
            # Rate limiter
            self.last_process_time = 0
            self.min_interval = 1.0 / self.get_parameter('rate').value
            
            self.get_logger().info(f'TinyTPU Vision Node started! Listening on {input_topic}')
        
        def image_callback(self, msg):
            """Process incoming camera image."""
            now = time.time()
            if now - self.last_process_time < self.min_interval:
                return
            self.last_process_time = now
            
            try:
                from cv_bridge import CvBridge
                bridge = CvBridge()
                image = bridge.imgmsg_to_cv2(msg, 'bgr8')
            except Exception as e:
                # Manual conversion
                image = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
            
            # Detect
            annotated, detections, elapsed = self.vision.detect_and_draw(image)
            
            # Publish detections as JSON
            det_msg = String()
            det_msg.data = json.dumps({
                'timestamp': now,
                'inference_ms': elapsed * 1000,
                'num_detections': len(detections),
                'detections': detections
            })
            self.det_pub.publish(det_msg)
            
            # Publish annotated image
            try:
                from cv_bridge import CvBridge
                bridge = CvBridge()
                self.img_pub.publish(bridge.cv2_to_imgmsg(annotated, 'bgr8'))
            except:
                pass
            
            if detections:
                names = [f"{d['class_name']}({d['confidence']*100:.0f}%)" for d in detections[:5]]
                self.get_logger().info(f'Detected: {", ".join(names)} [{elapsed*1000:.0f}ms]')


# ============================================================
# STANDALONE MODE (no ROS needed)
# ============================================================

def run_standalone(model_path="yolov5n.onnx", source=0):
    """
    Run vision detection standalone (no ROS).
    
    Args:
        model_path: Path to ONNX model
        source: Camera index (0) or video file path
    """
    vision = VisionProcessor(model_path)
    
    if not HAS_CV2:
        print("OpenCV required for standalone mode")
        print("Install: pip install opencv-python")
        
        # Demo with synthetic image
        print("\n[Demo with synthetic image]")
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        img[100:300, 200:400] = [50, 50, 200]  # Blue rectangle
        
        detections, elapsed = vision.detect(img)
        print(f"  Time: {elapsed*1000:.1f}ms")
        print(f"  Detections: {len(detections)}")
        for d in detections:
            print(f"    {d['class_name']}: {d['confidence']*100:.1f}%")
        return
    
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Cannot open camera {source}")
        return
    
    print(f"\n[Live Detection] Camera: {source}")
    print("Press 'q' to quit")
    
    fps_history = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        annotated, detections, elapsed = vision.detect_and_draw(frame)
        fps = 1.0 / elapsed if elapsed > 0 else 0
        fps_history.append(fps)
        avg_fps = np.mean(fps_history[-30:])
        
        # Draw FPS
        cv2.putText(annotated, f"TinyTPU: {avg_fps:.1f} FPS", (10, 30),
                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('TinyTPU Vision', annotated)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"\nAverage FPS: {np.mean(fps_history):.1f}")


def main():
    """ROS2 entry point."""
    if HAS_ROS:
        rclpy.init()
        node = VisionNode()
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            pass
        node.destroy_node()
        rclpy.shutdown()
    else:
        print("ROS2 not available - running standalone")
        run_standalone()


if __name__ == '__main__':
    main()
