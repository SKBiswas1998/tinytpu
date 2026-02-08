"""
TinyTPU HAL Backends - Unified inference interface across accelerators.

Architecture:
    Model -> HAL Backend -> {ONNX Runtime, Hailo HEF, Coral TFLite, NumPy}

Each backend implements the same interface:
    load(model_path) -> session
    run(session, inputs) -> outputs
    supports(model_path) -> bool
"""

import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("tinytpu.hal.backends")


class InferenceBackend(ABC):
    """Abstract interface for all inference backends."""
    name: str = "abstract"
    priority: int = 0

    @abstractmethod
    def available(self) -> bool:
        ...

    @abstractmethod
    def load(self, model_path: str, **kwargs) -> Any:
        ...

    @abstractmethod
    def run(self, session: Any, inputs: Dict[str, np.ndarray]) -> List[np.ndarray]:
        ...

    def supports(self, model_path: str) -> bool:
        return True

    def benchmark(self, session: Any, inputs: Dict[str, np.ndarray], runs: int = 20) -> dict:
        for _ in range(3):
            self.run(session, inputs)
        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            self.run(session, inputs)
            times.append(time.perf_counter() - t0)
        mean_ms = np.mean(times) * 1000
        return {
            "backend": self.name, "runs": runs,
            "mean_ms": round(mean_ms, 2), "min_ms": round(min(times) * 1000, 2),
            "max_ms": round(max(times) * 1000, 2),
            "fps": round(1000 / mean_ms, 1) if mean_ms > 0 else 0,
        }


class ONNXRuntimeBackend(InferenceBackend):
    """ONNX Runtime backend - fastest on CPU, supports CUDA/TensorRT."""
    name = "onnxruntime"
    priority = 50

    def __init__(self, providers: List[str] = None):
        self._providers = providers

    def available(self) -> bool:
        try:
            import onnxruntime
            return True
        except ImportError:
            return False

    def _get_providers(self) -> List[str]:
        if self._providers:
            return self._providers
        import onnxruntime as ort
        avail = ort.get_available_providers()
        providers = []
        for p in ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]:
            if p in avail:
                providers.append(p)
        return providers or ["CPUExecutionProvider"]

    def load(self, model_path: str, **kwargs) -> Any:
        import onnxruntime as ort
        providers = self._get_providers()
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_opts.intra_op_num_threads = kwargs.get("threads", 0)
        session = ort.InferenceSession(model_path, sess_opts, providers=providers)
        actual_provider = session.get_providers()[0]
        logger.info(f"Loaded {Path(model_path).name} with ONNX Runtime ({actual_provider})")
        return session

    def run(self, session, inputs: Dict[str, np.ndarray]) -> List[np.ndarray]:
        return session.run(None, inputs)

    def supports(self, model_path: str) -> bool:
        return model_path.endswith(".onnx")

    def get_input_info(self, session) -> List[dict]:
        return [{"name": inp.name, "shape": inp.shape, "dtype": inp.type} for inp in session.get_inputs()]

    def get_output_info(self, session) -> List[dict]:
        return [{"name": out.name, "shape": out.shape, "dtype": out.type} for out in session.get_outputs()]


class HailoBackend(InferenceBackend):
    """Hailo NPU backend - 13-26 TOPS on Raspberry Pi."""
    name = "hailo"
    priority = 90

    def available(self) -> bool:
        try:
            from hailo_platform import HailoRTDevice
            devices = HailoRTDevice.scan()
            return len(devices) > 0
        except (ImportError, Exception):
            return False

    def load(self, model_path: str, **kwargs) -> Any:
        if not model_path.endswith(".hef"):
            raise ValueError(f"Hailo requires .hef format, got {model_path}. Convert with: tinytpu convert {model_path} --target hailo8l")
        from hailo_platform import HailoRTDevice, ConfiguredInferModel, InferModel
        devices = HailoRTDevice.scan()
        device = devices[0]
        model = InferModel(device, model_path)
        configured = model.configure()
        logger.info(f"Loaded {Path(model_path).name} on Hailo {device.get_chip_name()}")
        return configured

    def run(self, session, inputs: Dict[str, np.ndarray]) -> List[np.ndarray]:
        bindings = session.create_bindings()
        for name, tensor in inputs.items():
            bindings.input(name).set_buffer(tensor)
        session.run([bindings])
        return [bindings.output(name).get_buffer() for name in bindings.output_names()]

    def supports(self, model_path: str) -> bool:
        return model_path.endswith(".hef")


class CoralBackend(InferenceBackend):
    """Google Coral Edge TPU - 4 TOPS, TFLite INT8."""
    name = "coral"
    priority = 80

    def available(self) -> bool:
        try:
            from pycoral.utils.edgetpu import list_edge_tpus
            return len(list_edge_tpus()) > 0
        except (ImportError, Exception):
            return False

    def load(self, model_path: str, **kwargs) -> Any:
        if not model_path.endswith("_edgetpu.tflite"):
            raise ValueError(f"Coral requires _edgetpu.tflite format. Convert with: tinytpu convert {model_path} --target coral")
        from pycoral.utils.edgetpu import make_interpreter
        interpreter = make_interpreter(model_path)
        interpreter.allocate_tensors()
        logger.info(f"Loaded {Path(model_path).name} on Coral Edge TPU")
        return interpreter

    def run(self, session, inputs: Dict[str, np.ndarray]) -> List[np.ndarray]:
        input_details = session.get_input_details()
        for i, (name, tensor) in enumerate(inputs.items()):
            session.set_tensor(input_details[i]["index"], tensor)
        session.invoke()
        output_details = session.get_output_details()
        return [session.get_tensor(d["index"]) for d in output_details]

    def supports(self, model_path: str) -> bool:
        return model_path.endswith(".tflite")


class NumPyBackend(InferenceBackend):
    """Pure NumPy fallback - works everywhere, slowest."""
    name = "numpy"
    priority = 10

    def available(self) -> bool:
        return True

    def load(self, model_path: str, **kwargs) -> Any:
        from tinytpu.inference.engine import TinyTPUEngine
        engine = TinyTPUEngine(model_path, quantize=kwargs.get("quantize", False))
        logger.info(f"Loaded {Path(model_path).name} with TinyTPU NumPy engine")
        return engine

    def run(self, session, inputs: Dict[str, np.ndarray]) -> List[np.ndarray]:
        outputs, _ = session.run(inputs)
        if not isinstance(outputs, list):
            outputs = [outputs]
        return outputs

    def supports(self, model_path: str) -> bool:
        return model_path.endswith(".onnx")


ALL_BACKENDS = [HailoBackend, CoralBackend, ONNXRuntimeBackend, NumPyBackend]


def get_backends() -> List[InferenceBackend]:
    return [cls() for cls in ALL_BACKENDS]


def get_available_backends() -> List[InferenceBackend]:
    return [b for b in get_backends() if b.available()]


def auto_backend(model_path: str = None, prefer: str = None) -> InferenceBackend:
    """Auto-select the best available backend."""
    available = get_available_backends()
    if not available:
        raise RuntimeError("No inference backends available. Install numpy at minimum.")
    if prefer:
        for b in available:
            if b.name == prefer:
                if model_path is None or b.supports(model_path):
                    return b
        logger.warning(f"Preferred backend '{prefer}' not available, falling back")
    if model_path:
        compatible = [b for b in available if b.supports(model_path)]
        if compatible:
            available = compatible
    available.sort(key=lambda b: b.priority, reverse=True)
    chosen = available[0]
    logger.info(f"Auto-selected backend: {chosen.name} (priority {chosen.priority})")
    return chosen


def list_backends() -> str:
    lines = []
    lines.append(f"{'Backend':<20} {'Available':<12} {'Priority':<10} {'Formats'}")
    lines.append("-" * 65)
    formats_map = {"hailo": ".hef", "coral": ".tflite", "onnxruntime": ".onnx", "numpy": ".onnx (slow)"}
    for b in get_backends():
        avail = "Yes" if b.available() else "No"
        lines.append(f"  {b.name:<18} {avail:<12} {b.priority:<10} {formats_map.get(b.name, 'unknown')}")
    return "\n".join(lines)
