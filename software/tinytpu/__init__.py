"""
TinyTPU - Tensor Processing Unit
================================
A minimal TPU implementation for learning and fast inference.

Installation:
    pip install tinytpu
    pip install tinytpu[fast]   # With PyTorch backend
    pip install tinytpu[llm]    # With LLM support

Quick Start:
    from tinytpu import TinyTPU
    import numpy as np
    
    tpu = TinyTPU()  # Auto-selects best backend
    
    A = np.random.randn(64, 128).astype(np.float32)
    B = np.random.randn(128, 64).astype(np.float32)
    C = tpu.matmul_float(A, B)
"""

__version__ = "0.2.0"

from .unified_backend import (
    TinyTPU,
    get_best_backend,
    Backend,
    NumpyBackend,
    PyTorchBackend,
)

__all__ = [
    "TinyTPU",
    "get_best_backend",
    "Backend", 
    "NumpyBackend",
    "PyTorchBackend",
    "__version__",
]
