from setuptools import setup, find_packages

package_name = 'tinytpu_ros'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/tinytpu_launch.py']),
        ('share/' + package_name + '/config', ['config/tinytpu_config.yaml']),
    ],
    install_requires=['setuptools', 'numpy'],
    zip_safe=True,
    maintainer='SKBiswas1998',
    description='TinyTPU Edge AI for ROS2',
    license='MIT',
    entry_points={
        'console_scripts': [
            'vision_node = tinytpu_ros.vision_node:main',
            'llm_node = tinytpu_ros.llm_node:main',
            'brain_node = tinytpu_ros.brain_node:main',
            'demo_node = tinytpu_ros.demo_node:main',
        ],
    },
)
