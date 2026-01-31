import numpy as np
from typing import Optional, Literal
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
    
    def matmul(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        A = A.astype(np.int8)
        B = B.astype(np.int8)
        if A.shape[1] != B.shape[0]:
            raise ValueError(f"Incompatible shapes: A{A.shape} @ B{B.shape}")
        M, K = A.shape
        K2, N = B.shape
        C = np.zeros((M, N), dtype=np.int32)
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
