"""
TinyTPU Robotics Demo - Full Pipeline Test
==========================================
Tests the complete vision -> brain -> command pipeline
without requiring ROS2 installation.
"""

import sys
import os
import numpy as np
import json

sys.path.insert(0, os.path.join('software'))
sys.path.insert(0, os.path.join('ros2', 'tinytpu_ros'))

from tinytpu_ros.vision_node import VisionProcessor
from tinytpu_ros.llm_node import LLMProcessor
from tinytpu_ros.brain_node import BrainProcessor

print("=" * 70)
print("TINYTPU ROBOTICS DEMO - Full Pipeline")
print("=" * 70)

# 1. Vision
print("\n[1. VISION - Object Detection]")
vision = VisionProcessor("yolov5n.onnx", conf_thresh=0.3)

# Create test image with objects
from PIL import Image, ImageDraw
img = Image.new('RGB', (640, 480), (180, 200, 180))
draw = ImageDraw.Draw(img)
draw.rectangle([200, 80, 350, 420], fill=(60, 60, 160))   # Person-like
draw.ellipse([450, 300, 530, 380], fill=(255, 200, 0))     # Ball-like
draw.rectangle([50, 350, 150, 430], fill=(139, 90, 43))    # Box-like
test_image = np.array(img)

detections, elapsed = vision.detect(test_image)
print(f"  Inference: {elapsed*1000:.0f}ms")
print(f"  Detections: {len(detections)}")
for d in detections:
    print(f"    {d['class_name']}: {d['confidence']*100:.0f}% at ({d['box']['cx']:.0f}, {d['box']['cy']:.0f})")

# 2. LLM
print("\n[2. LLM - Scene Understanding]")
llm = LLMProcessor.__new__(LLMProcessor)
llm.model = None
llm.tokenizer = None
llm.max_tokens = 50
llm.temperature = 0.7

if not detections:
    # Use synthetic detections for demo
    detections = [
        {'class_name': 'person', 'confidence': 0.85,
         'box': {'cx': 275, 'cy': 250, 'x1': 200, 'y1': 80, 'x2': 350, 'y2': 420,
                 'width': 150, 'height': 340}},
        {'class_name': 'sports ball', 'confidence': 0.7,
         'box': {'cx': 490, 'cy': 340, 'x1': 450, 'y1': 300, 'x2': 530, 'y2': 380,
                 'width': 80, 'height': 80}},
    ]
    print("  (Using synthetic detections for demo)")

scene = llm.describe_scene(detections)
print(f"  Scene: {scene}")

# 3. Brain
print("\n[3. BRAIN - Decision Making]")
brain = BrainProcessor(mode="follow_person", max_speed=0.3, max_angular=0.5,
                       image_width=640, image_height=480)

result = brain.process(detections)
print(f"  Action: {result['action']}")
print(f"  Speed: {result['linear_x']:.2f} m/s")
print(f"  Turn: {result['angular_z']:.2f} rad/s")
print(f"  Description: {result['description']}")

# 4. Command Interpretation
print("\n[4. COMMAND INTERPRETATION]")
commands = [
    "follow the person",
    "what do you see",
    "stop",
    "turn left",
    "find the sports ball",
    "go back",
]

for cmd in commands:
    parsed = llm.interpret_command(cmd, detections)
    action_result = brain.execute_command(parsed, detections)
    print(f"  '{cmd}'")
    print(f"    -> {action_result['action']}: v={action_result['linear_x']:.2f}, w={action_result['angular_z']:.2f}")

# 5. Full pipeline simulation
print("\n[5. FULL PIPELINE SIMULATION]")
print("  Simulating 5 frames of follow_person mode...")

for frame_i in range(5):
    # Simulate person moving
    cx = 200 + frame_i * 50
    person_det = [{
        'class_name': 'person', 'confidence': 0.9,
        'box': {'cx': cx, 'cy': 240, 'x1': cx-50, 'y1': 100, 'x2': cx+50, 'y2': 380,
                'width': 100, 'height': 280}
    }]
    
    result = brain.process(person_det)
    direction = "LEFT" if result['angular_z'] > 0.05 else "RIGHT" if result['angular_z'] < -0.05 else "STRAIGHT"
    print(f"  Frame {frame_i+1}: Person at cx={cx}, -> {result['action']} {direction} (v={result['linear_x']:.2f}, w={result['angular_z']:.2f})")

print("\n" + "=" * 70)
print("PIPELINE COMPLETE!")
print("=" * 70)
print("""
ROS2 Usage:
  # Build
  cd ros2/tinytpu_ros
  colcon build --packages-select tinytpu_ros
  source install/setup.bash
  
  # Launch all nodes
  ros2 launch tinytpu_ros tinytpu_launch.py
  
  # Or run individually
  ros2 run tinytpu_ros vision_node
  ros2 run tinytpu_ros llm_node
  ros2 run tinytpu_ros brain_node
  
  # Send command
  ros2 topic pub /tinytpu/llm_input std_msgs/String '{"type":"command","prompt":"follow the person"}'
  
  # Monitor
  ros2 topic echo /tinytpu/detections
  ros2 topic echo /cmd_vel
  ros2 topic echo /tinytpu/status

Standalone (no ROS):
  from tinytpu_ros.vision_node import VisionProcessor
  from tinytpu_ros.brain_node import BrainProcessor
  
  vision = VisionProcessor("yolov5n.onnx")
  brain = BrainProcessor(mode="follow_person")
  
  detections, _ = vision.detect(camera_frame)
  command = brain.process(detections)
  robot.set_velocity(command['linear_x'], command['angular_z'])
""")
