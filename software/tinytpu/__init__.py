"""
TinyTPU - Tensor Processing Unit
================================
Auto-selects the fastest available backend (PyTorch > Numba > NumPy).
"""

__version__ = "0.2.0"

from .unified_backend import TinyTPU, get_best_backend
from .unified_backend import Backend, NumpyBackend, PyTorchBackend, NumbaBackend

__all__ = [
    "TinyTPU",
    "get_best_backend", 
    "Backend",
    "NumpyBackend",
    "PyTorchBackend",
    "NumbaBackend",
]
