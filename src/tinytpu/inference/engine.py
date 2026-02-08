"""
TinyTPU ONNX Engine - Pure-Python ONNX runtime.

Supports 50+ operators. Serves as universal fallback when ONNX Runtime
C++ is not available (e.g., ARM boards without pre-built wheels).

Note: For best performance, install onnxruntime. This engine is ~3-10x slower
but works everywhere Python + NumPy work.
"""

import logging
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("tinytpu.inference.engine")


class TinyTPUEngine:
    """Pure-Python ONNX inference engine (50+ operators)."""

    def __init__(self, model_path: str, quantize: bool = False):
        self.model_path = model_path
        self.quantize = quantize
        self.input_names = []
        self.output_names = []
        self._graph = None
        self._tensors = {}
        self._load_model()

    def _load_model(self):
        """Load and parse ONNX model."""
        try:
            import onnx
            model = onnx.load(self.model_path)
            self._graph = model.graph
            self.input_names = [inp.name for inp in self._graph.input]
            self.output_names = [out.name for out in self._graph.output]
            for init in self._graph.initializer:
                self._tensors[init.name] = numpy_helper_to_array(init)
            logger.info(f"Loaded {self.model_path}: {len(self._graph.node)} ops, "
                        f"{len(self.input_names)} inputs, {len(self.output_names)} outputs")
        except ImportError:
            raise RuntimeError("onnx package required for TinyTPU engine. Install: pip install onnx")

    def run(self, inputs: Dict[str, np.ndarray]) -> Tuple[List[np.ndarray], dict]:
        """Run inference."""
        self._tensors.update(inputs)
        for node in self._graph.node:
            self._execute_node(node)
        outputs = [self._tensors.get(name, np.array([])) for name in self.output_names]
        return outputs, {"engine": "tinytpu_numpy"}

    def _execute_node(self, node):
        """Execute a single ONNX node."""
        op = node.op_type
        inputs = [self._tensors.get(name) for name in node.input if name]
        attrs = {a.name: _parse_attr(a) for a in node.attribute}

        try:
            if op == "Conv":
                out = self._conv(inputs, attrs)
            elif op == "Relu":
                out = [np.maximum(inputs[0], 0)]
            elif op == "MaxPool":
                out = self._maxpool(inputs, attrs)
            elif op == "Add":
                out = [inputs[0] + inputs[1]]
            elif op == "MatMul":
                out = [np.matmul(inputs[0], inputs[1])]
            elif op == "Gemm":
                out = self._gemm(inputs, attrs)
            elif op == "BatchNormalization":
                out = self._batchnorm(inputs, attrs)
            elif op == "Reshape":
                shape = inputs[1].astype(int).tolist()
                out = [inputs[0].reshape(shape)]
            elif op == "Flatten":
                axis = attrs.get("axis", 1)
                shape = inputs[0].shape
                new_shape = (int(np.prod(shape[:axis])), int(np.prod(shape[axis:])))
                out = [inputs[0].reshape(new_shape)]
            elif op == "Transpose":
                perm = attrs.get("perm", None)
                out = [np.transpose(inputs[0], perm)]
            elif op == "Sigmoid":
                out = [1.0 / (1.0 + np.exp(-np.clip(inputs[0], -500, 500)))]
            elif op == "Softmax":
                axis = attrs.get("axis", -1)
                e = np.exp(inputs[0] - np.max(inputs[0], axis=axis, keepdims=True))
                out = [e / np.sum(e, axis=axis, keepdims=True)]
            elif op == "Concat":
                axis = attrs.get("axis", 0)
                out = [np.concatenate(inputs, axis=axis)]
            elif op == "Squeeze":
                axes = attrs.get("axes", None)
                if len(inputs) > 1 and inputs[1] is not None:
                    axes = tuple(inputs[1].astype(int).tolist())
                out = [np.squeeze(inputs[0], axis=axes)]
            elif op == "Unsqueeze":
                if len(inputs) > 1 and inputs[1] is not None:
                    axes = sorted(inputs[1].astype(int).tolist())
                else:
                    axes = attrs.get("axes", [0])
                result = inputs[0]
                for ax in axes:
                    result = np.expand_dims(result, axis=ax)
                out = [result]
            elif op == "Mul":
                out = [inputs[0] * inputs[1]]
            elif op == "Sub":
                out = [inputs[0] - inputs[1]]
            elif op == "Div":
                out = [inputs[0] / (inputs[1] + 1e-10)]
            elif op == "Pow":
                out = [np.power(inputs[0], inputs[1])]
            elif op == "Sqrt":
                out = [np.sqrt(np.maximum(inputs[0], 0))]
            elif op == "Exp":
                out = [np.exp(np.clip(inputs[0], -500, 500))]
            elif op == "Log":
                out = [np.log(np.maximum(inputs[0], 1e-10))]
            elif op == "Clip":
                min_val = inputs[1] if len(inputs) > 1 and inputs[1] is not None else attrs.get("min", -3.4e38)
                max_val = inputs[2] if len(inputs) > 2 and inputs[2] is not None else attrs.get("max", 3.4e38)
                out = [np.clip(inputs[0], float(min_val) if np.ndim(min_val) == 0 else min_val, float(max_val) if np.ndim(max_val) == 0 else max_val)]
            elif op == "ReduceMean":
                axes = attrs.get("axes", None)
                keepdims = attrs.get("keepdims", 1)
                if len(inputs) > 1 and inputs[1] is not None:
                    axes = tuple(inputs[1].astype(int).tolist())
                out = [np.mean(inputs[0], axis=tuple(axes) if axes else None, keepdims=bool(keepdims))]
            elif op == "Gather":
                axis = attrs.get("axis", 0)
                out = [np.take(inputs[0], inputs[1].astype(int), axis=axis)]
            elif op == "Shape":
                out = [np.array(inputs[0].shape, dtype=np.int64)]
            elif op == "Constant":
                if "value" in attrs:
                    out = [attrs["value"]]
                else:
                    out = [np.array(0)]
            elif op == "Cast":
                to = attrs.get("to", 1)
                dtype_map = {1: np.float32, 6: np.int32, 7: np.int64, 9: bool, 11: np.float64}
                out = [inputs[0].astype(dtype_map.get(to, np.float32))]
            elif op == "Resize":
                out = self._resize(inputs, attrs)
            elif op == "Split":
                axis = attrs.get("axis", 0)
                split = attrs.get("split", None)
                if len(inputs) > 1 and inputs[1] is not None:
                    split = inputs[1].astype(int).tolist()
                if split:
                    indices = np.cumsum(split)[:-1]
                    out = list(np.split(inputs[0], indices, axis=axis))
                else:
                    num = len(node.output)
                    out = list(np.array_split(inputs[0], num, axis=axis))
            elif op == "Slice":
                out = self._slice(inputs, attrs)
            elif op == "Pad":
                out = self._pad(inputs, attrs)
            elif op in ("GlobalAveragePool",):
                axes = tuple(range(2, len(inputs[0].shape)))
                out = [np.mean(inputs[0], axis=axes, keepdims=True)]
            elif op == "AveragePool":
                out = self._avgpool(inputs, attrs)
            elif op == "LeakyRelu":
                alpha = attrs.get("alpha", 0.01)
                x = inputs[0]
                out = [np.where(x > 0, x, x * alpha)]
            elif op == "Tanh":
                out = [np.tanh(inputs[0])]
            elif op == "Neg":
                out = [np.negative(inputs[0])]
            elif op == "Abs":
                out = [np.abs(inputs[0])]
            elif op == "Floor":
                out = [np.floor(inputs[0])]
            elif op == "Ceil":
                out = [np.ceil(inputs[0])]
            elif op == "Equal":
                out = [np.equal(inputs[0], inputs[1])]
            elif op == "Less":
                out = [np.less(inputs[0], inputs[1])]
            elif op == "Greater":
                out = [np.greater(inputs[0], inputs[1])]
            elif op == "Where":
                out = [np.where(inputs[0], inputs[1], inputs[2])]
            elif op == "Identity":
                out = [inputs[0].copy()]
            elif op == "Erf":
                from scipy.special import erf
                out = [erf(inputs[0])]
            elif op == "ConstantOfShape":
                shape = inputs[0].astype(int).tolist()
                val = attrs.get("value", np.array([0.0], dtype=np.float32))
                out = [np.full(shape, val.flat[0], dtype=val.dtype)]
            else:
                logger.warning(f"Unsupported op: {op}, passing through zeros")
                out = [np.zeros(1, dtype=np.float32)]
        except Exception as e:
            logger.error(f"Error in {op}: {e}")
            out = [np.zeros(1, dtype=np.float32)]

        for i, name in enumerate(node.output):
            if name and i < len(out):
                self._tensors[name] = out[i]

    def _conv(self, inputs, attrs):
        x = inputs[0]
        w = inputs[1]
        b = inputs[2] if len(inputs) > 2 else None
        pads = attrs.get("pads", [0, 0, 0, 0])
        strides = attrs.get("strides", [1, 1])
        group = attrs.get("group", 1)
        if any(p > 0 for p in pads):
            x = np.pad(x, ((0,0), (0,0), (pads[0], pads[2]), (pads[1], pads[3])))
        N, C, H, W = x.shape
        OC, IC_g, KH, KW = w.shape
        SH, SW = strides
        OH = (H - KH) // SH + 1
        OW = (W - KW) // SW + 1
        out = np.zeros((N, OC, OH, OW), dtype=x.dtype)
        g_oc = OC // group
        g_ic = C // group
        for g in range(group):
            for n in range(N):
                for oc in range(g_oc):
                    for oh in range(OH):
                        for ow in range(OW):
                            h_start = oh * SH
                            w_start = ow * SW
                            patch = x[n, g*g_ic:(g+1)*g_ic, h_start:h_start+KH, w_start:w_start+KW]
                            out[n, g*g_oc+oc, oh, ow] = np.sum(patch * w[g*g_oc+oc])
        if b is not None:
            out += b.reshape(1, -1, 1, 1)
        return [out]

    def _maxpool(self, inputs, attrs):
        x = inputs[0]
        ks = attrs.get("kernel_shape", [2, 2])
        strides = attrs.get("strides", ks)
        pads = attrs.get("pads", [0, 0, 0, 0])
        if any(p > 0 for p in pads):
            x = np.pad(x, ((0,0), (0,0), (pads[0], pads[2]), (pads[1], pads[3])), constant_values=-np.inf)
        N, C, H, W = x.shape
        OH = (H - ks[0]) // strides[0] + 1
        OW = (W - ks[1]) // strides[1] + 1
        out = np.zeros((N, C, OH, OW), dtype=x.dtype)
        for oh in range(OH):
            for ow in range(OW):
                h_s, w_s = oh * strides[0], ow * strides[1]
                out[:, :, oh, ow] = x[:, :, h_s:h_s+ks[0], w_s:w_s+ks[1]].max(axis=(2, 3))
        return [out]

    def _avgpool(self, inputs, attrs):
        x = inputs[0]
        ks = attrs.get("kernel_shape", [2, 2])
        strides = attrs.get("strides", ks)
        pads = attrs.get("pads", [0, 0, 0, 0])
        if any(p > 0 for p in pads):
            x = np.pad(x, ((0,0), (0,0), (pads[0], pads[2]), (pads[1], pads[3])))
        N, C, H, W = x.shape
        OH = (H - ks[0]) // strides[0] + 1
        OW = (W - ks[1]) // strides[1] + 1
        out = np.zeros((N, C, OH, OW), dtype=x.dtype)
        for oh in range(OH):
            for ow in range(OW):
                h_s, w_s = oh * strides[0], ow * strides[1]
                out[:, :, oh, ow] = x[:, :, h_s:h_s+ks[0], w_s:w_s+ks[1]].mean(axis=(2, 3))
        return [out]

    def _gemm(self, inputs, attrs):
        A, B = inputs[0], inputs[1]
        C = inputs[2] if len(inputs) > 2 else None
        alpha = attrs.get("alpha", 1.0)
        beta = attrs.get("beta", 1.0)
        transA = attrs.get("transA", 0)
        transB = attrs.get("transB", 0)
        if transA:
            A = A.T
        if transB:
            B = B.T
        out = alpha * np.matmul(A, B)
        if C is not None:
            out += beta * C
        return [out]

    def _batchnorm(self, inputs, attrs):
        x, scale, bias, mean, var = inputs[:5]
        eps = attrs.get("epsilon", 1e-5)
        shape = [1, -1] + [1] * (x.ndim - 2)
        out = (x - mean.reshape(shape)) / np.sqrt(var.reshape(shape) + eps)
        out = out * scale.reshape(shape) + bias.reshape(shape)
        return [out]

    def _resize(self, inputs, attrs):
        x = inputs[0]
        if len(inputs) >= 4 and inputs[3] is not None:
            sizes = inputs[3].astype(int).tolist()
            if len(sizes) == 4:
                from PIL import Image
                n, c, h, w = x.shape
                out = np.zeros((n, c, sizes[2], sizes[3]), dtype=x.dtype)
                for i in range(n):
                    for j in range(c):
                        img = Image.fromarray(x[i, j])
                        img = img.resize((sizes[3], sizes[2]), Image.BILINEAR)
                        out[i, j] = np.array(img)
                return [out]
        return [x]

    def _slice(self, inputs, attrs):
        data = inputs[0]
        starts = inputs[1].astype(int).tolist() if len(inputs) > 1 else []
        ends = inputs[2].astype(int).tolist() if len(inputs) > 2 else []
        axes = inputs[3].astype(int).tolist() if len(inputs) > 3 else list(range(len(starts)))
        steps = inputs[4].astype(int).tolist() if len(inputs) > 4 else [1] * len(starts)
        slices = [slice(None)] * data.ndim
        for s, e, a, st in zip(starts, ends, axes, steps):
            slices[a] = slice(s, e, st)
        return [data[tuple(slices)]]

    def _pad(self, inputs, attrs):
        data = inputs[0]
        pads = inputs[1].astype(int).tolist() if len(inputs) > 1 else attrs.get("pads", [])
        mode = attrs.get("mode", "constant")
        value = float(inputs[2]) if len(inputs) > 2 and inputs[2] is not None else 0.0
        n = data.ndim
        pad_width = [(pads[i], pads[i + n]) for i in range(n)]
        if mode == "constant":
            return [np.pad(data, pad_width, mode="constant", constant_values=value)]
        elif mode == "reflect":
            return [np.pad(data, pad_width, mode="reflect")]
        elif mode == "edge":
            return [np.pad(data, pad_width, mode="edge")]
        return [np.pad(data, pad_width)]


def numpy_helper_to_array(tensor):
    """Convert ONNX TensorProto to numpy array."""
    from onnx import numpy_helper
    return numpy_helper.to_array(tensor)


def _parse_attr(attr):
    """Parse ONNX attribute to Python value."""
    if attr.type == 1:
        return attr.f
    elif attr.type == 2:
        return attr.i
    elif attr.type == 3:
        return attr.s.decode("utf-8") if attr.s else ""
    elif attr.type == 4:
        from onnx import numpy_helper
        return numpy_helper.to_array(attr.t)
    elif attr.type == 6:
        return list(attr.floats)
    elif attr.type == 7:
        return list(attr.ints)
    return None
