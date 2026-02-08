"""TinyTPU Inference - ONNX engine, model zoo, transforms."""

__all__ = ["Model", "ModelZoo", "TinyTPUEngine"]

def __getattr__(name):
    if name == "Model":
        from tinytpu.inference.model_zoo import Model
        return Model
    if name == "ModelZoo":
        from tinytpu.inference.model_zoo import ModelZoo
        return ModelZoo
    if name == "TinyTPUEngine":
        from tinytpu.inference.engine import TinyTPUEngine
        return TinyTPUEngine
    raise AttributeError(f"module 'tinytpu.inference' has no attribute {name!r}")
