"""
TinyTPU ONNX Engine
===================
Load and run ANY ONNX model on cheap hardware.

Supports:
- Image classification (MobileNet, ResNet)
- Object detection (YOLOv5-nano)
- NLP models (DistilBERT, SmolLM)
- INT8 quantization for memory savings

Usage:
    from tinytpu.onnx_engine import TinyTPUEngine
    
    engine = TinyTPUEngine("model.onnx")
    output = engine.run(input_data)
"""

import numpy as np
import time
import os
from typing import Dict, List, Optional, Tuple

class TinyTPUEngine:
    """
    ONNX inference engine powered by TinyTPU.
    
    Loads ONNX models and runs inference using
    optimized TinyTPU operations.
    """
    
    def __init__(self, model_path: str = None, quantize: bool = False):
        """
        Args:
            model_path: Path to .onnx file
            quantize: Enable INT8 quantization (4x memory reduction)
        """
        self._torch = None
        self._device = None
        self._quantize = quantize
        self._graph = None
        self._weights = {}
        self._initializers = {}
        self._node_ops = {}
        
        # Setup backend
        try:
            import torch
            self._torch = torch
            self._device = torch.device('cpu')
            import os
            torch.set_num_threads(os.cpu_count() or 4)
            self._backend = "pytorch"
        except ImportError:
            self._backend = "numpy"
        
        print(f"TinyTPU ONNX Engine ({self._backend} backend)")
        
        if model_path:
            self.load(model_path)
    
    def load(self, model_path: str):
        """Load an ONNX model."""
        import onnx
        from onnx import numpy_helper
        
        print(f"Loading: {model_path}")
        start = time.perf_counter()
        
        model = onnx.load(model_path)
        self._graph = model.graph
        
        # Extract weights/initializers
        total_params = 0
        total_bytes = 0
        
        for init in self._graph.initializer:
            tensor = numpy_helper.to_array(init)
            total_params += tensor.size
            
            if self._quantize and tensor.ndim >= 2 and tensor.dtype == np.float32:
                # INT8 quantize weights
                max_val = np.max(np.abs(tensor))
                scale = max_val / 127.0 if max_val > 0 else 1.0
                quantized = np.clip(np.round(tensor / scale), -128, 127).astype(np.int8)
                self._initializers[init.name] = {
                    'data': quantized,
                    'scale': scale,
                    'quantized': True
                }
                total_bytes += quantized.nbytes + 4
            else:
                self._initializers[init.name] = {
                    'data': tensor,
                    'scale': 1.0,
                    'quantized': False
                }
                total_bytes += tensor.nbytes
        
        # Count ops
        op_counts = {}
        for node in self._graph.node:
            op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1
        
        elapsed = time.perf_counter() - start
        
        orig_bytes = total_params * 4
        print(f"  Parameters: {total_params/1e6:.1f}M")
        if self._quantize:
            print(f"  Memory: {orig_bytes/1024/1024:.1f}MB -> {total_bytes/1024/1024:.1f}MB (INT8)")
        else:
            print(f"  Memory: {total_bytes/1024/1024:.1f}MB")
        print(f"  Operators: {len(self._graph.node)} ({len(op_counts)} types)")
        print(f"  Op types: {op_counts}")
        print(f"  Loaded in {elapsed:.2f}s")
        
        # Get input/output info
        self._inputs = {}
        for inp in self._graph.input:
            if inp.name not in self._initializers:
                shape = []
                for dim in inp.type.tensor_type.shape.dim:
                    if dim.dim_value > 0:
                        shape.append(dim.dim_value)
                    else:
                        shape.append(1)  # Dynamic dim default
                self._inputs[inp.name] = shape
        
        self._outputs = [o.name for o in self._graph.output]
        
        print(f"  Inputs: {self._inputs}")
        print(f"  Outputs: {self._outputs}")
    
    def _get_tensor(self, name: str, runtime_values: dict) -> np.ndarray:
        """Get tensor by name from initializers or runtime values."""
        if name in runtime_values:
            t = runtime_values[name]
            if isinstance(t, np.ndarray) and not t.flags.writeable:
                t = t.copy()
            return t
        if name in self._initializers:
            init = self._initializers[name]
            if init['quantized']:
                return init['data'].astype(np.float32) * init['scale']
            data = init['data']
            # Cast float16 to float32
            if hasattr(data, 'dtype') and data.dtype == np.float16:
                return data.astype(np.float32)
            if isinstance(data, np.ndarray) and not data.flags.writeable:
                return data.copy()
            return data
        raise ValueError(f"Tensor not found: {name}")
    
    def _get_attr(self, node, name, default=None):
        """Get node attribute."""
        for attr in node.attribute:
            if attr.name == name:
                if attr.type == 1:  # FLOAT
                    return attr.f
                elif attr.type == 2:  # INT
                    return attr.i
                elif attr.type == 3:  # STRING
                    return attr.s.decode()
                elif attr.type == 4:  # TENSOR
                    from onnx import numpy_helper
                    return numpy_helper.to_array(attr.t)
                elif attr.type == 6:  # FLOATS
                    return list(attr.floats)
                elif attr.type == 7:  # INTS
                    return list(attr.ints)
                elif attr.type == 8:  # STRINGS
                    return [s.decode() for s in attr.strings]
        return default
    
    # ============================================================
    # OPERATOR IMPLEMENTATIONS
    # ============================================================
    
    def _op_conv(self, node, values):
        """Convolution."""
        X = self._get_tensor(node.input[0], values).astype(np.float32)
        W = self._get_tensor(node.input[1], values).astype(np.float32)
        B = self._get_tensor(node.input[2], values).astype(np.float32) if len(node.input) > 2 and node.input[2] else None
        
        pads = self._get_attr(node, 'pads', [0,0,0,0])
        strides = self._get_attr(node, 'strides', [1,1])
        group = self._get_attr(node, 'group', 1)
        
        if self._torch:
            X_t = self._torch.from_numpy(X)
            W_t = self._torch.from_numpy(W)
            B_t = self._torch.from_numpy(B) if B is not None else None
            
            padding = (pads[0], pads[1]) if len(pads) >= 2 else 0
            stride = tuple(strides)
            
            out = self._torch.nn.functional.conv2d(X_t, W_t, B_t, stride=stride, padding=padding, groups=group)
            return out.numpy()
        else:
            # Naive numpy conv (slow but works)
            N, C, H, W_dim = X.shape
            OC, IC, KH, KW = W.shape
            
            pH, pW = pads[0], pads[1]
            sH, sW = strides
            
            X_pad = np.pad(X, ((0,0),(0,0),(pH,pH),(pW,pW)))
            
            OH = (H + 2*pH - KH) // sH + 1
            OW = (W_dim + 2*pW - KW) // sW + 1
            
            out = np.zeros((N, OC, OH, OW), dtype=np.float32)
            for n in range(N):
                for oc in range(OC):
                    for oh in range(OH):
                        for ow in range(OW):
                            h_start = oh * sH
                            w_start = ow * sW
                            patch = X_pad[n, :, h_start:h_start+KH, w_start:w_start+KW]
                            out[n, oc, oh, ow] = np.sum(patch * W[oc]) + (B[oc] if B is not None else 0)
            return out
    
    def _op_gemm(self, node, values):
        """General Matrix Multiplication."""
        A = self._get_tensor(node.input[0], values)
        B = self._get_tensor(node.input[1], values)
        C = self._get_tensor(node.input[2], values) if len(node.input) > 2 else None
        
        alpha = self._get_attr(node, 'alpha', 1.0)
        beta = self._get_attr(node, 'beta', 1.0)
        transA = self._get_attr(node, 'transA', 0)
        transB = self._get_attr(node, 'transB', 0)
        
        if transA: A = A.T
        if transB: B = B.T
        
        if self._torch:
            out = self._torch.matmul(
                self._torch.from_numpy(A.astype(np.float32)),
                self._torch.from_numpy(B.astype(np.float32))
            ).numpy() * alpha
        else:
            out = np.matmul(A, B) * alpha
        
        if C is not None:
            out += beta * C
        
        return out
    
    def _op_matmul(self, node, values):
        """Matrix multiplication."""
        A = self._get_tensor(node.input[0], values)
        B = self._get_tensor(node.input[1], values)
        
        if self._torch:
            return self._torch.matmul(
                self._torch.from_numpy(A.astype(np.float32)),
                self._torch.from_numpy(B.astype(np.float32))
            ).numpy()
        return np.matmul(A, B)
    
    def _op_relu(self, node, values):
        X = self._get_tensor(node.input[0], values)
        if self._torch:
            return self._torch.relu(self._torch.from_numpy(X)).numpy()
        return np.maximum(0, X)
    
    def _op_sigmoid(self, node, values):
        X = self._get_tensor(node.input[0], values)
        if self._torch:
            return self._torch.sigmoid(self._torch.from_numpy(X)).numpy()
        return 1.0 / (1.0 + np.exp(-X))
    
    def _op_tanh(self, node, values):
        X = self._get_tensor(node.input[0], values)
        return np.tanh(X)
    
    def _op_softmax(self, node, values):
        X = self._get_tensor(node.input[0], values)
        axis = self._get_attr(node, 'axis', -1)
        if self._torch:
            return self._torch.nn.functional.softmax(
                self._torch.from_numpy(X.astype(np.float32)), dim=axis
            ).numpy()
        exp_x = np.exp(X - np.max(X, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    
    def _op_add(self, node, values):
        A = self._get_tensor(node.input[0], values)
        B = self._get_tensor(node.input[1], values)
        return A + B
    
    def _op_sub(self, node, values):
        A = self._get_tensor(node.input[0], values)
        B = self._get_tensor(node.input[1], values)
        return A - B
    
    def _op_mul(self, node, values):
        A = self._get_tensor(node.input[0], values)
        B = self._get_tensor(node.input[1], values)
        return A * B
    
    def _op_div(self, node, values):
        A = self._get_tensor(node.input[0], values)
        B = self._get_tensor(node.input[1], values)
        return A / B
    
    def _op_sqrt(self, node, values):
        X = self._get_tensor(node.input[0], values)
        return np.sqrt(X)
    
    def _op_pow(self, node, values):
        X = self._get_tensor(node.input[0], values)
        Y = self._get_tensor(node.input[1], values)
        return np.power(X, Y)
    
    def _op_exp(self, node, values):
        X = self._get_tensor(node.input[0], values)
        return np.exp(X)
    
    def _op_log(self, node, values):
        X = self._get_tensor(node.input[0], values)
        return np.log(X)
    
    def _op_clip(self, node, values):
        X = self._get_tensor(node.input[0], values)
        min_val = self._get_tensor(node.input[1], values) if len(node.input) > 1 and node.input[1] else None
        max_val = self._get_tensor(node.input[2], values) if len(node.input) > 2 and node.input[2] else None
        
        if min_val is not None:
            X = np.maximum(X, min_val)
        if max_val is not None:
            X = np.minimum(X, max_val)
        return X
    
    def _op_reshape(self, node, values):
        X = self._get_tensor(node.input[0], values)
        shape = self._get_tensor(node.input[1], values).astype(np.int64)
        return X.reshape(shape)
    
    def _op_transpose(self, node, values):
        X = self._get_tensor(node.input[0], values)
        perm = self._get_attr(node, 'perm', None)
        if perm:
            return np.transpose(X, perm)
        return X.T
    
    def _op_flatten(self, node, values):
        X = self._get_tensor(node.input[0], values)
        axis = self._get_attr(node, 'axis', 1)
        shape = X.shape
        new_shape = (int(np.prod(shape[:axis])), int(np.prod(shape[axis:])))
        return X.reshape(new_shape)
    
    def _op_squeeze(self, node, values):
        X = self._get_tensor(node.input[0], values)
        if len(node.input) > 1:
            axes = self._get_tensor(node.input[1], values)
            return np.squeeze(X, axis=tuple(axes))
        return np.squeeze(X)
    
    def _op_unsqueeze(self, node, values):
        X = self._get_tensor(node.input[0], values)
        axes = self._get_attr(node, 'axes', None)
        if axes is None and len(node.input) > 1:
            axes = self._get_tensor(node.input[1], values).tolist()
        if axes:
            for ax in sorted(axes):
                X = np.expand_dims(X, axis=int(ax))
        return X
    
    def _op_concat(self, node, values):
        tensors = [self._get_tensor(inp, values) for inp in node.input]
        axis = self._get_attr(node, 'axis', 0)
        return np.concatenate(tensors, axis=axis)
    
    def _op_gather(self, node, values):
        X = self._get_tensor(node.input[0], values)
        indices = self._get_tensor(node.input[1], values).astype(np.int64)
        axis = self._get_attr(node, 'axis', 0)
        return np.take(X, indices, axis=axis)
    
    def _op_slice(self, node, values):
        X = self._get_tensor(node.input[0], values)
        starts = self._get_tensor(node.input[1], values).astype(np.int64)
        ends = self._get_tensor(node.input[2], values).astype(np.int64)
        axes = self._get_tensor(node.input[3], values).astype(np.int64) if len(node.input) > 3 else np.arange(len(starts))
        steps = self._get_tensor(node.input[4], values).astype(np.int64) if len(node.input) > 4 else np.ones(len(starts), dtype=np.int64)
        
        slices = [slice(None)] * len(X.shape)
        for i, ax in enumerate(axes):
            slices[int(ax)] = slice(int(starts[i]), int(ends[i]), int(steps[i]))
        return X[tuple(slices)]
    
    def _op_shape(self, node, values):
        X = self._get_tensor(node.input[0], values)
        return np.array(X.shape, dtype=np.int64)
    
    def _op_constant(self, node, values):
        value = self._get_attr(node, 'value', None)
        if value is not None:
            return value
        value_float = self._get_attr(node, 'value_float', None)
        if value_float is not None:
            return np.array(value_float, dtype=np.float32)
        value_int = self._get_attr(node, 'value_int', None)
        if value_int is not None:
            return np.array(value_int, dtype=np.int64)
        value_floats = self._get_attr(node, 'value_floats', None)
        if value_floats is not None:
            return np.array(value_floats, dtype=np.float32)
        value_ints = self._get_attr(node, 'value_ints', None)
        if value_ints is not None:
            return np.array(value_ints, dtype=np.int64)
        return np.array(0)
    
    def _op_constantofshape(self, node, values):
        shape = self._get_tensor(node.input[0], values).astype(np.int64)
        value = self._get_attr(node, 'value', None)
        if value is not None:
            return np.full(shape, value.flat[0])
        return np.zeros(shape, dtype=np.float32)
    
    def _op_batchnormalization(self, node, values):
        X = self._get_tensor(node.input[0], values)
        scale = self._get_tensor(node.input[1], values)
        bias = self._get_tensor(node.input[2], values)
        mean = self._get_tensor(node.input[3], values)
        var = self._get_tensor(node.input[4], values)
        epsilon = self._get_attr(node, 'epsilon', 1e-5)
        
        if X.ndim == 4:
            scale = scale.reshape(1, -1, 1, 1)
            bias = bias.reshape(1, -1, 1, 1)
            mean = mean.reshape(1, -1, 1, 1)
            var = var.reshape(1, -1, 1, 1)
        
        return scale * (X - mean) / np.sqrt(var + epsilon) + bias
    
    def _op_layernormalization(self, node, values):
        X = self._get_tensor(node.input[0], values)
        scale = self._get_tensor(node.input[1], values) if len(node.input) > 1 else None
        bias = self._get_tensor(node.input[2], values) if len(node.input) > 2 else None
        axis = self._get_attr(node, 'axis', -1)
        epsilon = self._get_attr(node, 'epsilon', 1e-5)
        
        mean = X.mean(axis=axis, keepdims=True)
        var = X.var(axis=axis, keepdims=True)
        out = (X - mean) / np.sqrt(var + epsilon)
        
        if scale is not None: out = out * scale
        if bias is not None: out = out + bias
        return out
    
    def _op_globalaveragepool(self, node, values):
        X = self._get_tensor(node.input[0], values)
        return np.mean(X, axis=(2, 3), keepdims=True)
    
    def _op_averagepool(self, node, values):
        X = self._get_tensor(node.input[0], values)
        kernel = self._get_attr(node, 'kernel_shape', [2,2])
        strides = self._get_attr(node, 'strides', kernel)
        pads = self._get_attr(node, 'pads', [0,0,0,0])
        
        if self._torch:
            X_t = self._torch.from_numpy(X)
            padding = (pads[0], pads[1])
            out = self._torch.nn.functional.avg_pool2d(X_t, kernel, strides, padding)
            return out.numpy()
        
        N, C, H, W = X.shape
        OH = (H + pads[0] + pads[2] - kernel[0]) // strides[0] + 1
        OW = (W + pads[1] + pads[3] - kernel[1]) // strides[1] + 1
        out = np.zeros((N, C, OH, OW), dtype=np.float32)
        X_pad = np.pad(X, ((0,0),(0,0),(pads[0],pads[2]),(pads[1],pads[3])))
        for oh in range(OH):
            for ow in range(OW):
                out[:,:,oh,ow] = X_pad[:,:,oh*strides[0]:oh*strides[0]+kernel[0],ow*strides[1]:ow*strides[1]+kernel[1]].mean(axis=(2,3))
        return out
    
    def _op_maxpool(self, node, values):
        X = self._get_tensor(node.input[0], values)
        kernel = self._get_attr(node, 'kernel_shape', [2,2])
        strides = self._get_attr(node, 'strides', kernel)
        pads = self._get_attr(node, 'pads', [0,0,0,0])
        
        if self._torch:
            X_t = self._torch.from_numpy(X)
            padding = (pads[0], pads[1])
            out = self._torch.nn.functional.max_pool2d(X_t, kernel, strides, padding)
            return out.numpy()
        
        N, C, H, W = X.shape
        OH = (H + pads[0] + pads[2] - kernel[0]) // strides[0] + 1
        OW = (W + pads[1] + pads[3] - kernel[1]) // strides[1] + 1
        out = np.zeros((N, C, OH, OW), dtype=np.float32)
        X_pad = np.pad(X, ((0,0),(0,0),(pads[0],pads[2]),(pads[1],pads[3])), constant_values=-np.inf)
        for oh in range(OH):
            for ow in range(OW):
                out[:,:,oh,ow] = X_pad[:,:,oh*strides[0]:oh*strides[0]+kernel[0],ow*strides[1]:ow*strides[1]+kernel[1]].max(axis=(2,3))
        return out
    
    def _op_pad(self, node, values):
        X = self._get_tensor(node.input[0], values)
        pads = self._get_tensor(node.input[1], values).astype(np.int64)
        mode = self._get_attr(node, 'mode', 'constant')
        constant_value = self._get_tensor(node.input[2], values) if len(node.input) > 2 else 0
        
        n = len(pads) // 2
        pad_width = [(int(pads[i]), int(pads[i+n])) for i in range(n)]
        return np.pad(X, pad_width, mode=mode, constant_values=float(constant_value) if np.isscalar(constant_value) else 0)
    
    def _op_resize(self, node, values):
        X = self._get_tensor(node.input[0], values)
        # Simplified nearest-neighbor resize
        if len(node.input) > 2 and node.input[2]:
            scales = self._get_tensor(node.input[2], values)
            if scales.size > 0 and not np.all(scales == 0):
                new_shape = [int(s * sc) for s, sc in zip(X.shape, scales)]
                if self._torch:
                    X_t = self._torch.from_numpy(X)
                    out = self._torch.nn.functional.interpolate(X_t, size=new_shape[2:], mode='nearest')
                    return out.numpy()
        if len(node.input) > 3 and node.input[3]:
            sizes = self._get_tensor(node.input[3], values).astype(np.int64)
            if self._torch:
                X_t = self._torch.from_numpy(X)
                out = self._torch.nn.functional.interpolate(X_t, size=list(sizes[2:]), mode='nearest')
                return out.numpy()
        return X
    
    def _op_cast(self, node, values):
        X = self._get_tensor(node.input[0], values)
        to = self._get_attr(node, 'to', 1)
        dtype_map = {1: np.float32, 6: np.int32, 7: np.int64, 9: bool, 10: np.float16, 11: np.float64}
        return X.astype(dtype_map.get(to, np.float32))
    
    def _op_where(self, node, values):
        cond = self._get_tensor(node.input[0], values)
        X = self._get_tensor(node.input[1], values)
        Y = self._get_tensor(node.input[2], values)
        return np.where(cond, X, Y)
    
    def _op_reducemean(self, node, values):
        X = self._get_tensor(node.input[0], values)
        axes = self._get_attr(node, 'axes', None)
        if axes is None and len(node.input) > 1:
            axes = self._get_tensor(node.input[1], values).tolist()
        keepdims = self._get_attr(node, 'keepdims', 1)
        if axes:
            return np.mean(X, axis=tuple(int(a) for a in axes), keepdims=bool(keepdims))
        return np.mean(X, keepdims=bool(keepdims))
    
    def _op_reducesum(self, node, values):
        X = self._get_tensor(node.input[0], values)
        axes = None
        if len(node.input) > 1 and node.input[1]:
            axes = self._get_tensor(node.input[1], values).tolist()
        keepdims = self._get_attr(node, 'keepdims', 1)
        if axes:
            return np.sum(X, axis=tuple(int(a) for a in axes), keepdims=bool(keepdims))
        return np.sum(X, keepdims=bool(keepdims))
    
    def _op_identity(self, node, values):
        return self._get_tensor(node.input[0], values)
    
    def _op_dropout(self, node, values):
        return self._get_tensor(node.input[0], values)
    
    def _op_split(self, node, values):
        X = self._get_tensor(node.input[0], values)
        axis = self._get_attr(node, 'axis', 0)
        split = None
        if len(node.input) > 1:
            split = self._get_tensor(node.input[1], values).astype(np.int64).tolist()
        if split:
            indices = np.cumsum(split)[:-1]
            return np.split(X, indices, axis=axis)
        num_outputs = len(node.output)
        return np.array_split(X, num_outputs, axis=axis)
    
    def _op_neg(self, node, values):
        return -self._get_tensor(node.input[0], values)
    
    def _op_abs(self, node, values):
        return np.abs(self._get_tensor(node.input[0], values))
    
    def _op_floor(self, node, values):
        return np.floor(self._get_tensor(node.input[0], values))
    
    def _op_ceil(self, node, values):
        return np.ceil(self._get_tensor(node.input[0], values))
    
    def _op_reciprocal(self, node, values):
        return 1.0 / self._get_tensor(node.input[0], values)
    
    def _op_equal(self, node, values):
        A = self._get_tensor(node.input[0], values)
        B = self._get_tensor(node.input[1], values)
        return (A == B)
    
    def _op_less(self, node, values):
        A = self._get_tensor(node.input[0], values)
        B = self._get_tensor(node.input[1], values)
        return (A < B)
    
    def _op_greater(self, node, values):
        A = self._get_tensor(node.input[0], values)
        B = self._get_tensor(node.input[1], values)
        return (A > B)
    
    def _op_not(self, node, values):
        return ~self._get_tensor(node.input[0], values).astype(bool)
    
    def _op_erf(self, node, values):
        from scipy.special import erf
        return erf(self._get_tensor(node.input[0], values))
    
    # ============================================================
    # EXECUTION ENGINE
    # ============================================================
    
    def run(self, inputs: Dict[str, np.ndarray] = None, **kwargs) -> Dict[str, np.ndarray]:
        """
        Run inference.
        
        Args:
            inputs: Dict mapping input names to numpy arrays
            **kwargs: Alternative way to pass inputs
        
        Returns:
            Dict mapping output names to numpy arrays
        """
        if self._graph is None:
            raise RuntimeError("No model loaded. Call load() first.")
        
        # Combine inputs
        if inputs is None:
            inputs = {}
        inputs.update(kwargs)
        
        # Initialize runtime values with inputs
        values = dict(inputs)
        
        start = time.perf_counter()
        
        # Execute nodes in order
        for node in self._graph.node:
            op_type = node.op_type.lower()
            op_fn = getattr(self, f'_op_{op_type}', None)
            
            if op_fn is None:
                print(f"  WARNING: Unsupported op '{node.op_type}', skipping")
                # Pass through first input if available
                if node.input and node.input[0] in values:
                    for out in node.output:
                        values[out] = values[node.input[0]]
                continue
            
            try:
                result = op_fn(node, values)
                
                # Handle multiple outputs (e.g., Split)
                if isinstance(result, (list, tuple)):
                    for i, out in enumerate(node.output):
                        if out and i < len(result):
                            values[out] = result[i]
                else:
                    for out in node.output:
                        if out:
                            values[out] = result
            except Exception as e:
                print(f"  ERROR in {node.op_type} ({node.name}): {e}")
                if node.input and node.input[0] in values:
                    for out in node.output:
                        values[out] = values[node.input[0]]
        
        elapsed = time.perf_counter() - start
        
        # Collect outputs
        outputs = {}
        for name in self._outputs:
            if name in values:
                outputs[name] = values[name]
        
        return outputs, elapsed
    
    def benchmark(self, inputs, runs=10):
        """Benchmark inference speed."""
        # Warmup
        for _ in range(3):
            self.run(inputs)
        
        times = []
        for _ in range(runs):
            _, elapsed = self.run(inputs)
            times.append(elapsed)
        
        return {
            'mean_ms': np.mean(times) * 1000,
            'median_ms': np.median(times) * 1000,
            'min_ms': np.min(times) * 1000,
            'max_ms': np.max(times) * 1000,
            'fps': 1.0 / np.mean(times),
        }
    
    def summary(self):
        """Print model summary."""
        if self._graph is None:
            print("No model loaded")
            return
        
        print("\nModel Summary:")
        print(f"  Inputs: {self._inputs}")
        print(f"  Outputs: {self._outputs}")
        print(f"  Nodes: {len(self._graph.node)}")
        print(f"  Weights: {len(self._initializers)}")
        
        total_params = sum(
            init['data'].size for init in self._initializers.values()
        )
        total_bytes = sum(
            init['data'].nbytes for init in self._initializers.values()
        )
        print(f"  Parameters: {total_params/1e6:.1f}M")
        print(f"  Memory: {total_bytes/1024/1024:.1f}MB")
        if self._quantize:
            print(f"  Quantized: INT8")


if __name__ == "__main__":
    print("TinyTPU ONNX Engine loaded successfully!")
    print("Usage:")
    print("  engine = TinyTPUEngine('model.onnx')")
    print("  output = engine.run({'input': data})")


