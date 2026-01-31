import numpy as np
from typing import Optional, Union, Literal
from .backends import SimulatorBackend, FPGABackend

class TinyTPU:
    def __init__(self, device: Optional[str] = None, backend: Literal["auto", "simulator", "fpga"] = "auto", array_size: int = 16):
        self.array_size = array_size
        if backend == "simulator":
            self._backend = SimulatorBackend(array_size=array_size)
        elif backend == "fpga":
            self._backend = FPGABackend(device=device, array_size=array_size)
        else:
            self._backend = SimulatorBackend(array_size=array_size)
        self._backend.connect()
    
    @property
    def is_connected(self) -> bool:
        return self._backend.is_connected
    
    @property
    def backend_name(self) -> str:
        return self._backend.name
    
    def _to_numpy(self, x) -> np.ndarray:
        """Convert input to numpy array."""
        if x is None:
            raise TypeError("Input cannot be None")
        if isinstance(x, np.ndarray):
            return x
        return np.array(x)
    
    def _validate_int8(self, x: np.ndarray, name: str) -> np.ndarray:
        """Validate and convert to INT8 with range checking."""
        if x.dtype in [np.float32, np.float64, np.float16]:
            if np.any(x < -128) or np.any(x > 127):
                raise ValueError(f"{name} has values outside INT8 range [-128, 127]")
        x = x.astype(np.int8)
        return x
    
    def matmul(self, A, B) -> np.ndarray:
        """Matrix multiplication with input validation."""
        # Convert to numpy
        A = self._to_numpy(A)
        B = self._to_numpy(B)
        
        # Validate dimensions
        if A.ndim != 2:
            raise ValueError(f"A must be 2D, got {A.ndim}D")
        if B.ndim != 2:
            raise ValueError(f"B must be 2D, got {B.ndim}D")
        if A.shape[1] != B.shape[0]:
            raise ValueError(f"Incompatible shapes: A{A.shape} @ B{B.shape}")
        
        # Handle empty matrices
        if A.size == 0 or B.size == 0:
            return np.zeros((A.shape[0], B.shape[1]), dtype=np.int32)
        
        # Convert to int8
        A = self._validate_int8(A, "A")
        B = self._validate_int8(B, "B")
        
        M, K = A.shape
        K2, N = B.shape
        C = np.zeros((M, N), dtype=np.int32)
        
        # Tiled matmul
        for i in range(0, M, self.array_size):
            for j in range(0, N, self.array_size):
                for k in range(0, K, self.array_size):
                    i_end = min(i + self.array_size, M)
                    j_end = min(j + self.array_size, N)
                    k_end = min(k + self.array_size, K)
                    A_tile = self._pad_tile(A[i:i_end, k:k_end])
                    B_tile = self._pad_tile(B[k:k_end, j:j_end])
                    C_tile = self._backend.matmul(A_tile, B_tile)
                    C[i:i_end, j:j_end] += C_tile[:i_end-i, :j_end-j]
        return C
    
    def _pad_tile(self, tile: np.ndarray) -> np.ndarray:
        padded = np.zeros((self.array_size, self.array_size), dtype=np.int8)
        h, w = tile.shape
        padded[:h, :w] = tile
        return padded
    
    def softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        return self._backend.softmax(x, axis)
    
    def benchmark(self, size: int = 128, iterations: int = 100) -> dict:
        import time
        A = np.random.randint(-128, 127, (size, size), dtype=np.int8)
        B = np.random.randint(-128, 127, (size, size), dtype=np.int8)
        for _ in range(5):
            self.matmul(A, B)
        start = time.perf_counter()
        for _ in range(iterations):
            self.matmul(A, B)
        elapsed = time.perf_counter() - start
        gops = (2 * size ** 3 * iterations) / elapsed / 1e9
        return {"backend": self.backend_name, "array_size": self.array_size, "matrix_size": size, "iterations": iterations, "total_time_s": elapsed, "time_per_matmul_ms": elapsed / iterations * 1000, "gops": gops}
    
    def close(self):
        self._backend.disconnect()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()
        return False
    
    def __repr__(self):
        return f"TinyTPU(backend={self.backend_name!r}, array_size={self.array_size})"
