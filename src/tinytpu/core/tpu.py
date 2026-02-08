"""
TinyTPU Core - Systolic array simulation + optimized operations.
"""

import numpy as np
import logging

logger = logging.getLogger("tinytpu.core.tpu")


class TinyTPU:
    """Core TinyTPU engine with 4x4 systolic array and backend selection."""

    def __init__(self, array_size=4, backend="auto"):
        self.array_size = array_size
        self.backend = backend

    def matmul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Matrix multiply using systolic array tiling."""
        M, K1 = a.shape
        K2, N = b.shape
        assert K1 == K2, f"Dimension mismatch: {a.shape} x {b.shape}"
        c = np.zeros((M, N), dtype=a.dtype)
        s = self.array_size
        for i in range(0, M, s):
            for j in range(0, N, s):
                for k in range(0, K1, s):
                    a_tile = a[i:i+s, k:k+s]
                    b_tile = b[k:k+s, j:j+s]
                    c[i:i+s, j:j+s] += self._systolic_matmul(a_tile, b_tile)
        return c

    def _systolic_matmul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Weight-stationary systolic array simulation."""
        M, K = a.shape
        K2, N = b.shape
        c = np.zeros((M, N), dtype=a.dtype)
        for i in range(M):
            for j in range(N):
                for k in range(K):
                    c[i, j] += a[i, k] * b[k, j]
        return c

    def conv2d(self, x, w, stride=1, padding=0):
        """2D convolution using systolic array for inner product."""
        if padding > 0:
            x = np.pad(x, ((0,0), (0,0), (padding, padding), (padding, padding)))
        N, C, H, W = x.shape
        OC, IC, KH, KW = w.shape
        OH = (H - KH) // stride + 1
        OW = (W - KW) // stride + 1
        out = np.zeros((N, OC, OH, OW), dtype=x.dtype)
        w_flat = w.reshape(OC, -1)
        for n in range(N):
            for oh in range(OH):
                for ow in range(OW):
                    h_s, w_s = oh * stride, ow * stride
                    patch = x[n, :, h_s:h_s+KH, w_s:w_s+KW].reshape(-1)
                    out[n, :, oh, ow] = self.matmul(
                        w_flat, patch.reshape(-1, 1)
                    ).flatten()
        return out
