"""
TinyTPU ROS2 Launch File
=========================
Launch all TinyTPU nodes for robot AI.

Usage:
    ros2 launch tinytpu_ros tinytpu_launch.py
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    return LaunchDescription([
        # Arguments
        DeclareLaunchArgument('model', default_value='yolov5n.onnx'),
        DeclareLaunchArgument('mode', default_value='follow_person'),
        DeclareLaunchArgument('camera_topic', default_value='/camera/image_raw'),
        
        # Vision Node - Object Detection
        Node(
            package='tinytpu_ros',
            executable='vision_node',
            name='tinytpu_vision',
            parameters=[{
                'model': LaunchConfiguration('model'),
                'confidence_threshold': 0.4,
                'input_topic': LaunchConfiguration('camera_topic'),
                'rate': 10.0,
            }],
            output='screen',
        ),
        
        # LLM Node - Language Understanding
        Node(
            package='tinytpu_ros',
            executable='llm_node',
            name='tinytpu_llm',
            parameters=[{
                'model': 'gpt2',
                'max_tokens': 50,
            }],
            output='screen',
        ),
        
        # Brain Node - Decision Making
        Node(
            package='tinytpu_ros',
            executable='brain_node',
            name='tinytpu_brain',
            parameters=[{
                'mode': LaunchConfiguration('mode'),
                'max_speed': 0.3,
                'max_angular': 0.5,
            }],
            output='screen',
        ),
    ])
