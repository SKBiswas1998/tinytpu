import numpy as np
from abc import ABC, abstractmethod

class Backend(ABC):
    def __init__(self, array_size: int = 16):
        self.array_size = array_size
        self._connected = False
    
    @property
    @abstractmethod
    def name(self) -> str: pass
    
    @property
    def is_connected(self) -> bool: return self._connected
    
    @abstractmethod
    def connect(self): pass
    
    @abstractmethod
    def disconnect(self): pass
    
    @abstractmethod
    def matmul(self, A: np.ndarray, B: np.ndarray) -> np.ndarray: pass
    
    def softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

class SimulatorBackend(Backend):
    @property
    def name(self) -> str: return "simulator"
    
    def connect(self): self._connected = True
    def disconnect(self): self._connected = False
    
    def matmul(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        return np.matmul(A.astype(np.int32), B.astype(np.int32))

class FPGABackend(Backend):
    def __init__(self, device: str, array_size: int = 16, baudrate: int = 3000000):
        super().__init__(array_size)
        self._device = device
        self._baudrate = baudrate
        self._serial = None
    
    @property
    def name(self) -> str: return "fpga"
    
    def connect(self):
        import serial
        self._serial = serial.Serial(port=self._device, baudrate=self._baudrate, timeout=5.0)
        self._connected = True
    
    def disconnect(self):
        if self._serial: self._serial.close()
        self._connected = False
    
    def matmul(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        # TODO: Implement FPGA protocol
        raise NotImplementedError("FPGA matmul not yet implemented")
