__version__ = "0.1.0"
from .core import TinyTPU
from .backends import SimulatorBackend, FPGABackend
__all__ = ["TinyTPU", "SimulatorBackend", "FPGABackend"]
