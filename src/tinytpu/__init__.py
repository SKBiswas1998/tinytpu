"""
TinyTPU - Production Edge AI for Robots Without GPUs
=====================================================

From silicon architecture to safety-certified robot control in one framework.

Quick Start::

    import tinytpu

    # Auto-detect hardware, run inference in 3 lines
    model = tinytpu.Model("yolov8n")
    results = model.predict(frame)

    # Or go full production
    pipeline = tinytpu.Pipeline(mode="follow", target="person")
    pipeline.start()

Modules:
    tinytpu.core        - Systolic array, backend selection, quantization
    tinytpu.inference   - ONNX engine, model zoo, transforms
    tinytpu.perception  - Object detection, Kalman tracking, camera
    tinytpu.control     - Safety controller, robot control, async pipeline
    tinytpu.monitoring  - Thermal, memory, black box recorder
    tinytpu.numerical   - Richardson extrapolation, HITTER, Horner activations
    tinytpu.hal         - Hardware Abstraction Layer (Hailo, Coral, CPU, GPU)
    tinytpu.cli         - Command-line tools
"""

__version__ = "0.1.0"
__author__ = "SK Biswas"
__license__ = "MIT"

import importlib
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tinytpu.inference.engine import TinyTPUEngine
    from tinytpu.inference.model_zoo import Model, ModelZoo
    from tinytpu.perception.detector import ObjectDetector
    from tinytpu.perception.tracker import ObjectTracker
    from tinytpu.control.safety import SafetyController
    from tinytpu.control.pipeline import Pipeline
    from tinytpu.monitoring.thermal import ThermalMonitor
    from tinytpu.hal.detect import detect_hardware, HardwareInfo


def _lazy(module_path: str, attr: str):
    """Import on first access."""
    mod = importlib.import_module(module_path)
    return getattr(mod, attr)


class _LazyModel:
    """
    Convenience wrapper: tinytpu.Model("yolov8n") loads + configures everything.

    >>> model = tinytpu.Model("yolov8n")
    >>> results = model.predict(frame)
    """
    def __new__(cls, model_name: str = "yolov8n", **kwargs):
        ModelClass = _lazy("tinytpu.inference.model_zoo", "Model")
        return ModelClass(model_name, **kwargs)


class _LazyPipeline:
    """
    Convenience wrapper: tinytpu.Pipeline(mode="follow") sets up full stack.

    >>> pipeline = tinytpu.Pipeline(mode="follow", target="person")
    >>> pipeline.start()
    """
    def __new__(cls, **kwargs):
        PipelineClass = _lazy("tinytpu.control.pipeline", "Pipeline")
        return PipelineClass(**kwargs)


Model = _LazyModel
Pipeline = _LazyPipeline


def detect_hardware():
    """Detect available AI accelerators (Hailo, Coral, GPU, CPU)."""
    func = _lazy("tinytpu.hal.detect", "detect_hardware")
    return func()


def benchmark(model_name: str = "yolov8n", runs: int = 20):
    """Quick benchmark on detected hardware."""
    func = _lazy("tinytpu.cli.main", "run_benchmark")
    return func(model_name=model_name, runs=runs)


def info():
    """Print TinyTPU version and detected hardware."""
    import platform
    print(f"TinyTPU v{__version__}")
    print(f"Python {platform.python_version()} on {platform.system()} {platform.machine()}")
    try:
        hw = detect_hardware()
        print(f"Hardware: {hw}")
    except Exception:
        print("Hardware detection: run 'tinytpu hardware' for details")


_SUBMODULES = {
    "core", "inference", "perception", "control",
    "monitoring", "numerical", "hal", "cli",
}


def __getattr__(name: str):
    if name in _SUBMODULES:
        return importlib.import_module(f"tinytpu.{name}")
    raise AttributeError(f"module 'tinytpu' has no attribute {name!r}")


__all__ = [
    "__version__", "Model", "Pipeline", "detect_hardware", "benchmark", "info",
    "core", "inference", "perception", "control", "monitoring", "numerical", "hal", "cli",
]
