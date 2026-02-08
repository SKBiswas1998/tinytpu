"""TinyTPU Hardware Abstraction Layer."""
from tinytpu.hal.detect import detect_hardware, HardwareInfo, DeviceInfo
from tinytpu.hal.backends import (
    InferenceBackend, auto_backend, get_available_backends, list_backends,
    ONNXRuntimeBackend, HailoBackend, CoralBackend, NumPyBackend,
)

__all__ = [
    "detect_hardware", "HardwareInfo", "DeviceInfo",
    "InferenceBackend", "auto_backend", "get_available_backends", "list_backends",
    "ONNXRuntimeBackend", "HailoBackend", "CoralBackend", "NumPyBackend",
]
