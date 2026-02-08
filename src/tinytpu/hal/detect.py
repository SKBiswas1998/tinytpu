"""
TinyTPU Hardware Abstraction Layer - Detect & abstract AI accelerators.

Supports:
    - Hailo-8L / Hailo-8 / Hailo-10H (AI HAT+)
    - Google Coral Edge TPU (USB / PCIe / M.2)
    - Raspberry Pi AI Camera (IMX500)
    - NVIDIA GPU (via CUDA)
    - ARM CPU (NEON optimized via ONNX Runtime)
    - x86 CPU (fallback)
"""

from dataclasses import dataclass, field
from typing import List, Optional
import platform
import os
import logging

logger = logging.getLogger("tinytpu.hal")


@dataclass
class DeviceInfo:
    """Information about a single AI accelerator."""
    name: str
    backend: str
    available: bool = False
    tops: Optional[str] = None
    memory: Optional[str] = None
    driver_version: Optional[str] = None
    notes: Optional[str] = None


@dataclass
class HardwareInfo:
    """Complete hardware detection results."""
    devices: List[DeviceInfo] = field(default_factory=list)
    recommended: str = "cpu"
    recommended_model: str = "yolov8n"
    platform_name: str = ""
    arch: str = ""
    ram_gb: float = 0.0
    cpu_cores: int = 0
    warnings: List[str] = field(default_factory=list)

    def __repr__(self):
        avail = [d.name for d in self.devices if d.available]
        return f"HardwareInfo(recommended={self.recommended!r}, available={avail})"


def _detect_hailo() -> Optional[DeviceInfo]:
    device = DeviceInfo(name="Hailo NPU", backend="hailo")
    try:
        import hailo_platform
        device.available = True
        device.driver_version = getattr(hailo_platform, "__version__", "unknown")
        try:
            from hailo_platform import HailoRTDevice
            hw_devices = HailoRTDevice.scan()
            if hw_devices:
                chip = hw_devices[0]
                device.name = f"Hailo-{chip.get_chip_name()}"
                device.tops = f"{chip.get_neural_network_core_frequency()} MHz"
        except Exception:
            device.name = "Hailo NPU (HailoRT detected)"
    except ImportError:
        device.available = False
        device.notes = "Install: sudo apt install hailort"
    if not device.available:
        try:
            hailo_devs = [f for f in os.listdir("/dev") if f.startswith("hailo")]
            if hailo_devs:
                device.available = True
                device.name = f"Hailo NPU (/dev/{hailo_devs[0]})"
                device.notes = "Device found but HailoRT Python not installed"
        except (OSError, FileNotFoundError):
            pass
    return device


def _detect_coral() -> Optional[DeviceInfo]:
    device = DeviceInfo(name="Google Coral", backend="coral")
    try:
        import pycoral
        device.available = True
        device.tops = "4 TOPS"
        device.driver_version = getattr(pycoral, "__version__", "unknown")
    except ImportError:
        pass
    if not device.available:
        try:
            import tflite_runtime.interpreter as tflite
            delegates = tflite.load_delegate("libedgetpu.so.1")
            device.available = True
            device.tops = "4 TOPS"
            device.notes = "Edge TPU delegate available"
        except (ImportError, OSError, ValueError):
            pass
    if not device.available:
        try:
            import subprocess
            result = subprocess.run(["lsusb"], capture_output=True, text=True, timeout=5)
            if "1a6e:089a" in result.stdout or "18d1:9302" in result.stdout:
                device.available = True
                device.notes = "USB Coral detected but pycoral not installed"
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
    if not device.available:
        device.notes = "Install: pip install pycoral (Python <3.10 only)"
    return device


def _detect_imx500() -> Optional[DeviceInfo]:
    device = DeviceInfo(name="Pi AI Camera (IMX500)", backend="imx500")
    try:
        import subprocess
        result = subprocess.run(
            ["libcamera-hello", "--list-cameras"],
            capture_output=True, text=True, timeout=5,
        )
        if "imx500" in result.stdout.lower():
            device.available = True
            device.tops = "~1 TOPS"
            device.notes = "On-sensor inference"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        device.notes = "Requires Raspberry Pi AI Camera"
    return device


def _detect_cuda() -> Optional[DeviceInfo]:
    device = DeviceInfo(name="NVIDIA GPU", backend="cuda")
    try:
        import torch
        if torch.cuda.is_available():
            device.available = True
            gpu_name = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_mem / 1e9
            device.name = f"NVIDIA {gpu_name}"
            device.memory = f"{mem:.1f} GB"
    except ImportError:
        pass
    if not device.available:
        try:
            import onnxruntime as ort
            if "CUDAExecutionProvider" in ort.get_available_providers():
                device.available = True
                device.name = "NVIDIA GPU (ONNX Runtime CUDA)"
        except ImportError:
            pass
    if not device.available:
        device.notes = "No CUDA GPU detected"
    return device


def _has_onnxruntime() -> bool:
    try:
        import onnxruntime
        return True
    except ImportError:
        return False


def _detect_cpu() -> DeviceInfo:
    arch = platform.machine().lower()
    is_arm = arch in ("aarch64", "armv7l", "armv8l")
    device = DeviceInfo(
        name=f"CPU ({platform.processor() or arch})",
        backend="onnxruntime" if _has_onnxruntime() else "numpy",
        available=True,
    )
    if is_arm:
        device.notes = "ARM NEON acceleration available"
    else:
        device.notes = "x86 SSE/AVX available"
    if _has_onnxruntime():
        import onnxruntime as ort
        device.driver_version = ort.__version__
    else:
        device.notes += " (numpy fallback - install onnxruntime for 3-10x speedup)"
    return device


def _get_system_info() -> dict:
    info = {"ram_gb": 0.0, "cpu_cores": os.cpu_count() or 1, "platform": platform.machine()}
    try:
        import psutil
        info["ram_gb"] = psutil.virtual_memory().total / 1e9
    except ImportError:
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        info["ram_gb"] = kb / 1e6
                        break
        except (OSError, ValueError):
            pass
    return info


def _recommend_backend(devices: List[DeviceInfo], ram_gb: float) -> tuple:
    priority = ["hailo", "coral", "cuda", "onnxruntime", "numpy"]
    available = {d.backend: d for d in devices if d.available}
    for backend in priority:
        if backend in available:
            recommended = backend
            break
    else:
        recommended = "numpy"
    if recommended == "hailo":
        model = "yolov8s"
    elif recommended == "coral":
        model = "yolov8n"
    elif recommended == "cuda":
        model = "yolov8m"
    else:
        model = "yolov8n"
    return recommended, model


def detect_hardware() -> HardwareInfo:
    """
    Detect all available AI accelerators and recommend the best configuration.

    Returns:
        HardwareInfo with detected devices, recommended backend, and model.

    Example:
        >>> hw = detect_hardware()
        >>> print(hw.recommended)
        >>> print(hw.recommended_model)
    """
    sys_info = _get_system_info()
    devices = []

    hailo = _detect_hailo()
    if hailo:
        devices.append(hailo)
    coral = _detect_coral()
    if coral:
        devices.append(coral)
    imx500 = _detect_imx500()
    if imx500:
        devices.append(imx500)
    cuda = _detect_cuda()
    if cuda:
        devices.append(cuda)
    cpu = _detect_cpu()
    devices.append(cpu)

    warnings = []
    if sys_info["ram_gb"] < 2:
        warnings.append(f"Low RAM ({sys_info['ram_gb']:.1f} GB) - use lightweight models")
    if sys_info["ram_gb"] < 1:
        warnings.append("Very low RAM - consider Pico/MCU deployment instead")

    avail_count = sum(1 for d in devices if d.available)
    if avail_count == 1 and not _has_onnxruntime():
        warnings.append("Only NumPy backend available - install onnxruntime for 3-10x speedup")

    recommended, recommended_model = _recommend_backend(devices, sys_info["ram_gb"])

    return HardwareInfo(
        devices=devices, recommended=recommended, recommended_model=recommended_model,
        platform_name=sys_info["platform"], arch=platform.machine(),
        ram_gb=sys_info["ram_gb"], cpu_cores=sys_info["cpu_cores"], warnings=warnings,
    )
