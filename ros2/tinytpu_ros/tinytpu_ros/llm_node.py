"""
TinyTPU LLM Node for ROS2
==========================
On-device language model for robot reasoning.

Subscribes: /tinytpu/llm_input (String)
Publishes:  /tinytpu/llm_output (String)

Use cases:
- Voice command interpretation
- Scene description from detections
- Task planning from natural language
"""

import time
import json
import sys
import os

try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
    HAS_ROS = True
except ImportError:
    HAS_ROS = False

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class LLMProcessor:
    """
    Lightweight LLM for robot reasoning.
    
    Usage:
        llm = LLMProcessor()
        response = llm.generate("What should the robot do if it sees a person?")
    """
    
    def __init__(self, model_name="gpt2", max_tokens=50, temperature=0.7):
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.model = None
        self.tokenizer = None
        
        if not HAS_TORCH:
            print("[LLM] PyTorch not found. Install: pip install torch transformers")
            return
        
        print(f"[LLM] Loading model: {model_name}")
        self.device = torch.device('cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
        self.model.eval()
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        params = sum(p.numel() for p in self.model.parameters()) / 1e6
        print(f"[LLM] Ready! ({params:.0f}M params)")
    
    def generate(self, prompt, max_tokens=None, temperature=None):
        """Generate text from prompt."""
        if self.model is None:
            return {"text": "LLM not loaded", "tokens": 0, "time_ms": 0}
        
        max_tokens = max_tokens or self.max_tokens
        temperature = temperature or self.temperature
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        start = time.perf_counter()
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=0.95,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )
        elapsed = time.perf_counter() - start
        
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        new_tokens = outputs.shape[1] - inputs.input_ids.shape[1]
        
        return {
            "text": text,
            "tokens": int(new_tokens),
            "time_ms": elapsed * 1000,
            "tok_per_sec": new_tokens / elapsed if elapsed > 0 else 0
        }
    
    def describe_scene(self, detections):
        """Convert detections into natural language description."""
        if not detections:
            return "I see an empty scene with no recognizable objects."
        
        objects = {}
        for d in detections:
            name = d['class_name']
            objects[name] = objects.get(name, 0) + 1
        
        parts = []
        for name, count in objects.items():
            if count == 1:
                parts.append(f"a {name}")
            else:
                parts.append(f"{count} {name}s")
        
        if len(parts) == 1:
            description = f"I see {parts[0]}."
        elif len(parts) == 2:
            description = f"I see {parts[0]} and {parts[1]}."
        else:
            description = f"I see {', '.join(parts[:-1])}, and {parts[-1]}."
        
        return description
    
    def interpret_command(self, command, detections=None):
        """
        Interpret natural language command for robot actions.
        
        Returns: dict with action, target, parameters
        """
        command_lower = command.lower().strip()
        
        # Rule-based command parsing (fast, no LLM needed)
        actions = {
            'go to': 'navigate',
            'move to': 'navigate',
            'drive to': 'navigate',
            'follow': 'follow',
            'track': 'follow',
            'stop': 'stop',
            'halt': 'stop',
            'turn left': 'turn_left',
            'turn right': 'turn_right',
            'look at': 'look_at',
            'find': 'search',
            'search for': 'search',
            'where is': 'search',
            'pick up': 'pick',
            'grab': 'pick',
            'put down': 'place',
            'drop': 'place',
            'come here': 'come',
            'back up': 'reverse',
            'go back': 'reverse',
            'speed up': 'faster',
            'slow down': 'slower',
            'take photo': 'capture',
            'what do you see': 'describe',
            'describe': 'describe',
        }
        
        # Match command
        result = {'action': 'unknown', 'target': None, 'raw': command}
        
        for phrase, action in actions.items():
            if phrase in command_lower:
                result['action'] = action
                # Extract target (word after the phrase)
                remainder = command_lower.split(phrase)[-1].strip()
                if remainder:
                    # Match target against visible objects
                    if detections:
                        for det in detections:
                            if det['class_name'] in remainder:
                                result['target'] = det['class_name']
                                result['target_box'] = det['box']
                                break
                    if not result['target']:
                        result['target'] = remainder.split()[0] if remainder else None
                break
        
        # If describe action, generate description
        if result['action'] == 'describe' and detections:
            result['description'] = self.describe_scene(detections)
        
        return result


# ============================================================
# ROS2 NODE
# ============================================================

if HAS_ROS:
    class LLMNode(Node):
        def __init__(self):
            super().__init__('tinytpu_llm')
            
            self.declare_parameter('model', 'gpt2')
            self.declare_parameter('max_tokens', 50)
            self.declare_parameter('temperature', 0.7)
            
            model = self.get_parameter('model').value
            max_tokens = self.get_parameter('max_tokens').value
            temp = self.get_parameter('temperature').value
            
            self.llm = LLMProcessor(model, max_tokens, temp)
            
            self.sub = self.create_subscription(String, '/tinytpu/llm_input', self.input_callback, 10)
            self.pub = self.create_publisher(String, '/tinytpu/llm_output', 10)
            
            self.get_logger().info('TinyTPU LLM Node started!')
        
        def input_callback(self, msg):
            data = json.loads(msg.data)
            prompt = data.get('prompt', '')
            detections = data.get('detections', None)
            
            if data.get('type') == 'command':
                result = self.llm.interpret_command(prompt, detections)
            elif data.get('type') == 'describe':
                result = {'description': self.llm.describe_scene(detections or [])}
            else:
                result = self.llm.generate(prompt)
            
            out_msg = String()
            out_msg.data = json.dumps(result)
            self.pub.publish(out_msg)
            
            self.get_logger().info(f'LLM: {json.dumps(result)[:100]}')


def main():
    if HAS_ROS:
        rclpy.init()
        node = LLMNode()
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            pass
        node.destroy_node()
        rclpy.shutdown()
    else:
        print("=" * 60)
        print("TinyTPU LLM - Standalone Demo")
        print("=" * 60)
        
        llm = LLMProcessor()
        
        # Test scene description
        test_detections = [
            {'class_name': 'person', 'confidence': 0.9, 'box': {'cx': 320, 'cy': 240}},
            {'class_name': 'cup', 'confidence': 0.7, 'box': {'cx': 100, 'cy': 300}},
            {'class_name': 'laptop', 'confidence': 0.8, 'box': {'cx': 400, 'cy': 200}},
        ]
        
        print(f"\n[Scene Description]")
        print(f"  {llm.describe_scene(test_detections)}")
        
        # Test commands
        commands = [
            "follow the person",
            "go to the laptop",
            "what do you see",
            "stop",
            "find the cup",
        ]
        
        print(f"\n[Command Interpretation]")
        for cmd in commands:
            result = llm.interpret_command(cmd, test_detections)
            print(f"  '{cmd}' -> action={result['action']}, target={result.get('target')}")
        
        # Test generation
        if llm.model:
            print(f"\n[Text Generation]")
            result = llm.generate("The robot should", max_tokens=30)
            print(f"  {result['text']}")
            print(f"  [{result['tokens']} tokens, {result['tok_per_sec']:.1f} tok/s]")


if __name__ == '__main__':
    main()
