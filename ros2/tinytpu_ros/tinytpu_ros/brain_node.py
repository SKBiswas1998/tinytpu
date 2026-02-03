"""
TinyTPU Brain Node for ROS2
============================
Combines vision detections + LLM reasoning -> robot commands.

The brain connects everything:
  Camera -> Vision Node -> Brain Node -> Motor Commands
                             |
                           LLM Node (when reasoning needed)

Subscribes:
  /tinytpu/detections (String - JSON)
  /tinytpu/llm_output (String - JSON)
  
Publishes:
  /cmd_vel (geometry_msgs/Twist)
  /tinytpu/status (String)
  /tinytpu/llm_input (String - JSON)
"""

import json
import time
import math
import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
    from geometry_msgs.msg import Twist
    HAS_ROS = True
except ImportError:
    HAS_ROS = False


class BrainProcessor:
    """
    Robot brain - converts detections into actions.
    
    Modes:
    - follow_person: Follow detected person
    - avoid_obstacles: Drive while avoiding objects  
    - explore: Wander and describe environment
    - command: Wait for voice/text commands
    """
    
    def __init__(self, mode="follow_person", max_speed=0.3, max_angular=0.5,
                 image_width=640, image_height=480):
        self.mode = mode
        self.max_speed = max_speed
        self.max_angular = max_angular
        self.image_width = image_width
        self.image_height = image_height
        
        self.priority_objects = ['person', 'cat', 'dog', 'car', 'stop sign']
        self.obstacle_objects = ['car', 'truck', 'bus', 'chair', 'couch', 'bed', 'dining table']
        
        self.last_detections = []
        self.last_action_time = 0
        self.tracking_target = None
        
        print(f"[Brain] Mode: {mode}, max_speed={max_speed}")
    
    def process(self, detections):
        """
        Process detections and return velocity command.
        
        Returns: dict with linear_x, linear_y, angular_z, action, description
        """
        self.last_detections = detections
        
        if self.mode == "follow_person":
            return self._follow_person(detections)
        elif self.mode == "avoid_obstacles":
            return self._avoid_obstacles(detections)
        elif self.mode == "explore":
            return self._explore(detections)
        elif self.mode == "command":
            return self._command_mode(detections)
        else:
            return self._stop("Unknown mode")
    
    def _stop(self, reason=""):
        return {
            'linear_x': 0.0, 'linear_y': 0.0, 'angular_z': 0.0,
            'action': 'stop', 'description': reason
        }
    
    def _follow_person(self, detections):
        """Follow the largest/closest person."""
        persons = [d for d in detections if d['class_name'] == 'person']
        
        if not persons:
            # No person visible - rotate to search
            return {
                'linear_x': 0.0, 'linear_y': 0.0, 'angular_z': 0.2,
                'action': 'searching', 'description': 'Looking for person...'
            }
        
        # Find largest person (closest)
        target = max(persons, key=lambda d: d['box']['width'] * d['box']['height'])
        box = target['box']
        
        # Calculate steering
        # Center of person relative to image center
        error_x = (box['cx'] - self.image_width / 2) / (self.image_width / 2)  # -1 to 1
        
        # Size of person relative to image (proxy for distance)
        person_size = (box['width'] * box['height']) / (self.image_width * self.image_height)
        
        # Angular: turn toward person
        angular_z = -error_x * self.max_angular
        
        # Linear: move forward if person is small (far), stop if large (close)
        if person_size > 0.15:
            # Person is close - stop
            linear_x = 0.0
            action = 'reached'
            desc = f'Person reached (size={person_size:.2f})'
        elif person_size > 0.05:
            # Person at medium distance - slow approach
            linear_x = self.max_speed * 0.5
            action = 'approaching'
            desc = f'Approaching person (size={person_size:.2f})'
        else:
            # Person is far - move faster
            linear_x = self.max_speed
            action = 'following'
            desc = f'Following person (size={person_size:.2f})'
        
        return {
            'linear_x': linear_x, 'linear_y': 0.0, 'angular_z': angular_z,
            'action': action, 'description': desc,
            'target': target
        }
    
    def _avoid_obstacles(self, detections):
        """Drive forward while avoiding obstacles."""
        obstacles = [d for d in detections if d['class_name'] in self.obstacle_objects]
        
        if not obstacles:
            # Clear path - drive forward
            return {
                'linear_x': self.max_speed, 'linear_y': 0.0, 'angular_z': 0.0,
                'action': 'driving', 'description': 'Path clear'
            }
        
        # Find closest obstacle
        closest = max(obstacles, key=lambda d: d['box']['width'] * d['box']['height'])
        box = closest['box']
        size = (box['width'] * box['height']) / (self.image_width * self.image_height)
        
        if size > 0.1:
            # Obstacle very close - stop and turn
            error_x = (box['cx'] - self.image_width / 2) / (self.image_width / 2)
            turn_dir = 1.0 if error_x < 0 else -1.0  # Turn away from obstacle
            
            return {
                'linear_x': 0.0, 'linear_y': 0.0, 'angular_z': turn_dir * self.max_angular,
                'action': 'avoiding',
                'description': f'Avoiding {closest["class_name"]} (size={size:.2f})'
            }
        else:
            # Obstacle visible but far - slow down and steer away
            error_x = (box['cx'] - self.image_width / 2) / (self.image_width / 2)
            angular_z = -error_x * self.max_angular * 0.5
            
            return {
                'linear_x': self.max_speed * 0.5, 'linear_y': 0.0, 'angular_z': angular_z,
                'action': 'cautious',
                'description': f'{closest["class_name"]} ahead, proceeding carefully'
            }
    
    def _explore(self, detections):
        """Wander and observe."""
        t = time.time()
        
        # Gentle sinusoidal wandering
        angular_z = math.sin(t * 0.3) * self.max_angular * 0.3
        linear_x = self.max_speed * 0.3
        
        if detections:
            names = list(set(d['class_name'] for d in detections))
            desc = f'Exploring. I see: {", ".join(names)}'
        else:
            desc = 'Exploring...'
        
        return {
            'linear_x': linear_x, 'linear_y': 0.0, 'angular_z': angular_z,
            'action': 'exploring', 'description': desc
        }
    
    def _command_mode(self, detections):
        """Wait for commands. Just observe."""
        return self._stop("Waiting for command...")
    
    def execute_command(self, command_result, detections=None):
        """
        Execute a parsed command from LLM node.
        
        Args:
            command_result: dict from LLMProcessor.interpret_command()
        """
        action = command_result.get('action', 'unknown')
        target = command_result.get('target')
        
        if action == 'stop':
            return self._stop("Stopping as commanded")
        
        elif action == 'follow' and target:
            targets = [d for d in (detections or self.last_detections) 
                      if d['class_name'] == target]
            if targets:
                self.mode = "follow_person"  # Temporary mode switch
                return self._follow_person(targets)
            return self._stop(f"Cannot find {target}")
        
        elif action == 'navigate' and target:
            return {
                'linear_x': self.max_speed, 'linear_y': 0.0, 'angular_z': 0.0,
                'action': 'navigating', 'description': f'Moving toward {target}'
            }
        
        elif action == 'search':
            return {
                'linear_x': 0.0, 'linear_y': 0.0, 'angular_z': self.max_angular,
                'action': 'searching', 'description': f'Searching for {target}'
            }
        
        elif action == 'turn_left':
            return {
                'linear_x': 0.0, 'linear_y': 0.0, 'angular_z': self.max_angular,
                'action': 'turning', 'description': 'Turning left'
            }
        
        elif action == 'turn_right':
            return {
                'linear_x': 0.0, 'linear_y': 0.0, 'angular_z': -self.max_angular,
                'action': 'turning', 'description': 'Turning right'
            }
        
        elif action == 'reverse':
            return {
                'linear_x': -self.max_speed * 0.5, 'linear_y': 0.0, 'angular_z': 0.0,
                'action': 'reversing', 'description': 'Going backward'
            }
        
        elif action == 'describe':
            desc = command_result.get('description', 'Nothing to describe')
            return self._stop(desc)
        
        return self._stop(f"Unknown action: {action}")


# ============================================================
# ROS2 NODE
# ============================================================

if HAS_ROS:
    class BrainNode(Node):
        def __init__(self):
            super().__init__('tinytpu_brain')
            
            self.declare_parameter('mode', 'follow_person')
            self.declare_parameter('max_speed', 0.3)
            self.declare_parameter('max_angular', 0.5)
            
            mode = self.get_parameter('mode').value
            max_speed = self.get_parameter('max_speed').value
            max_angular = self.get_parameter('max_angular').value
            
            self.brain = BrainProcessor(mode, max_speed, max_angular)
            self.latest_detections = []
            
            # Subscribe to detections
            self.det_sub = self.create_subscription(
                String, '/tinytpu/detections', self.detection_callback, 10)
            
            # Subscribe to LLM output
            self.llm_sub = self.create_subscription(
                String, '/tinytpu/llm_output', self.llm_callback, 10)
            
            # Publish velocity commands
            self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
            
            # Publish status
            self.status_pub = self.create_publisher(String, '/tinytpu/status', 10)
            
            # Publish LLM requests
            self.llm_pub = self.create_publisher(String, '/tinytpu/llm_input', 10)
            
            self.get_logger().info(f'TinyTPU Brain Node started! Mode: {mode}')
        
        def detection_callback(self, msg):
            data = json.loads(msg.data)
            self.latest_detections = data.get('detections', [])
            
            # Process with brain
            result = self.brain.process(self.latest_detections)
            
            # Publish cmd_vel
            twist = Twist()
            twist.linear.x = result['linear_x']
            twist.angular.z = result['angular_z']
            self.cmd_pub.publish(twist)
            
            # Publish status
            status = String()
            status.data = json.dumps(result)
            self.status_pub.publish(status)
        
        def llm_callback(self, msg):
            data = json.loads(msg.data)
            result = self.brain.execute_command(data, self.latest_detections)
            
            twist = Twist()
            twist.linear.x = result['linear_x']
            twist.angular.z = result['angular_z']
            self.cmd_pub.publish(twist)


def main():
    if HAS_ROS:
        rclpy.init()
        node = BrainNode()
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            pass
        node.destroy_node()
        rclpy.shutdown()
    else:
        print("=" * 60)
        print("TinyTPU Brain - Standalone Demo")
        print("=" * 60)
        
        brain = BrainProcessor(mode="follow_person")
        
        # Simulate detections
        scenarios = [
            ("No person visible", []),
            ("Person on the left", [
                {'class_name': 'person', 'confidence': 0.9,
                 'box': {'cx': 100, 'cy': 240, 'x1': 50, 'y1': 100, 'x2': 150, 'y2': 380,
                         'width': 100, 'height': 280}}
            ]),
            ("Person centered and close", [
                {'class_name': 'person', 'confidence': 0.95,
                 'box': {'cx': 320, 'cy': 240, 'x1': 200, 'y1': 50, 'x2': 440, 'y2': 430,
                         'width': 240, 'height': 380}}
            ]),
            ("Person far away", [
                {'class_name': 'person', 'confidence': 0.6,
                 'box': {'cx': 350, 'cy': 300, 'x1': 330, 'y1': 280, 'x2': 370, 'y2': 340,
                         'width': 40, 'height': 60}}
            ]),
        ]
        
        print("\n[Follow Person Mode]")
        for name, dets in scenarios:
            result = brain.process(dets)
            print(f"\n  Scenario: {name}")
            print(f"    Action: {result['action']}")
            print(f"    Speed: {result['linear_x']:.2f} m/s")
            print(f"    Turn: {result['angular_z']:.2f} rad/s")
            print(f"    {result['description']}")
        
        # Test command execution
        print("\n[Command Execution]")
        from tinytpu_ros.llm_node import LLMProcessor
        llm = LLMProcessor.__new__(LLMProcessor)
        llm.model = None
        llm.tokenizer = None
        llm.max_tokens = 50
        llm.temperature = 0.7
        
        test_dets = [
            {'class_name': 'person', 'confidence': 0.9,
             'box': {'cx': 200, 'cy': 240, 'width': 100, 'height': 280}},
            {'class_name': 'cup', 'confidence': 0.7,
             'box': {'cx': 400, 'cy': 300, 'width': 50, 'height': 60}},
        ]
        
        commands = [
            "follow the person",
            "stop",
            "turn left",
            "search for dog",
        ]
        
        for cmd in commands:
            parsed = llm.interpret_command(cmd, test_dets)
            result = brain.execute_command(parsed, test_dets)
            print(f"\n  Command: '{cmd}'")
            print(f"    -> action={result['action']}, speed={result['linear_x']:.2f}, turn={result['angular_z']:.2f}")
            print(f"    {result['description']}")


if __name__ == '__main__':
    main()
