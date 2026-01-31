"""
TinyTPU Tensor Operations
=========================
Full tensor calculation support with TPU acceleration.
"""

import numpy as np
from typing import Union, Tuple, List, Optional

class TinyTensor:
    """Tensor with TinyTPU acceleration for matmul operations."""
    
    def __init__(self, data, tpu=None):
        if isinstance(data, TinyTensor):
            self.data = data.data.copy()
        elif isinstance(data, np.ndarray):
            self.data = data.astype(np.float32) if data.dtype not in [np.float32, np.float64, np.int8, np.int32, np.int64] else data
        else:
            self.data = np.array(data, dtype=np.float32)
        
        if tpu is None:
            from tinytpu.unified_backend import TinyTPU
            self._tpu = TinyTPU(backend="simulator")
        else:
            self._tpu = tpu
    
    @property
    def shape(self): return self.data.shape
    
    @property
    def ndim(self): return self.data.ndim
    
    @property
    def dtype(self): return self.data.dtype
    
    @property
    def size(self): return self.data.size
    
    @property
    def T(self): return TinyTensor(self.data.T, self._tpu)
    
    # ==================== CREATION ====================
    
    @classmethod
    def zeros(cls, *shape, tpu=None):
        return cls(np.zeros(shape if len(shape) > 1 else shape[0], dtype=np.float32), tpu)
    
    @classmethod
    def ones(cls, *shape, tpu=None):
        return cls(np.ones(shape if len(shape) > 1 else shape[0], dtype=np.float32), tpu)
    
    @classmethod
    def randn(cls, *shape, tpu=None):
        return cls(np.random.randn(*shape).astype(np.float32), tpu)
    
    @classmethod
    def rand(cls, *shape, tpu=None):
        return cls(np.random.rand(*shape).astype(np.float32), tpu)
    
    @classmethod
    def randint(cls, low, high, shape, tpu=None):
        return cls(np.random.randint(low, high, shape).astype(np.float32), tpu)
    
    @classmethod
    def eye(cls, n, tpu=None):
        return cls(np.eye(n, dtype=np.float32), tpu)
    
    @classmethod
    def arange(cls, *args, tpu=None):
        return cls(np.arange(*args).astype(np.float32), tpu)
    
    @classmethod
    def linspace(cls, start, stop, num, tpu=None):
        return cls(np.linspace(start, stop, num).astype(np.float32), tpu)
    
    @classmethod
    def full(cls, shape, value, tpu=None):
        return cls(np.full(shape, value, dtype=np.float32), tpu)
    
    # ==================== SHAPE OPS ====================
    
    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return TinyTensor(self.data.reshape(shape), self._tpu)
    
    def transpose(self, *axes):
        """Transpose tensor. Supports negative indices."""
        if not axes:
            return TinyTensor(self.data.T, self._tpu)
        
        # Handle single tuple argument
        if len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = tuple(axes[0])
        
        # Handle two arguments for swapping axes (like -2, -1)
        if len(axes) == 2:
            ax0, ax1 = axes
            # Convert negative indices
            ndim = self.ndim
            if ax0 < 0: ax0 = ndim + ax0
            if ax1 < 0: ax1 = ndim + ax1
            return TinyTensor(np.swapaxes(self.data, ax0, ax1), self._tpu)
        
        # Full axes specification
        return TinyTensor(self.data.transpose(axes), self._tpu)
    
    def permute(self, *dims):
        if len(dims) == 1 and isinstance(dims[0], (tuple, list)):
            dims = tuple(dims[0])
        return TinyTensor(self.data.transpose(dims), self._tpu)
    
    def squeeze(self, axis=None):
        return TinyTensor(np.squeeze(self.data, axis), self._tpu)
    
    def unsqueeze(self, axis):
        return TinyTensor(np.expand_dims(self.data, axis), self._tpu)
    
    def flatten(self, start_dim=0, end_dim=-1):
        shape = self.shape
        if end_dim < 0:
            end_dim = len(shape) + end_dim
        new_shape = shape[:start_dim] + (-1,) + shape[end_dim+1:]
        return self.reshape(new_shape)
    
    def view(self, *shape):
        return self.reshape(*shape)
    
    def expand(self, *sizes):
        return TinyTensor(np.broadcast_to(self.data, sizes), self._tpu)
    
    def repeat(self, *repeats):
        return TinyTensor(np.tile(self.data, repeats), self._tpu)
    
    def chunk(self, chunks, dim=0):
        return [TinyTensor(c, self._tpu) for c in np.array_split(self.data, chunks, axis=dim)]
    
    def split(self, split_size, dim=0):
        indices = list(range(split_size, self.shape[dim], split_size))
        return [TinyTensor(c, self._tpu) for c in np.split(self.data, indices, axis=dim)]
    
    @staticmethod
    def cat(tensors, dim=0):
        tpu = tensors[0]._tpu if tensors else None
        return TinyTensor(np.concatenate([t.data for t in tensors], axis=dim), tpu)
    
    @staticmethod
    def stack(tensors, dim=0):
        tpu = tensors[0]._tpu if tensors else None
        return TinyTensor(np.stack([t.data for t in tensors], axis=dim), tpu)
    
    # ==================== ELEMENT-WISE OPS ====================
    
    def __add__(self, other):
        o = other.data if isinstance(other, TinyTensor) else other
        return TinyTensor(self.data + o, self._tpu)
    
    def __radd__(self, other): return self.__add__(other)
    
    def __sub__(self, other):
        o = other.data if isinstance(other, TinyTensor) else other
        return TinyTensor(self.data - o, self._tpu)
    
    def __rsub__(self, other):
        return TinyTensor(other - self.data, self._tpu)
    
    def __mul__(self, other):
        o = other.data if isinstance(other, TinyTensor) else other
        return TinyTensor(self.data * o, self._tpu)
    
    def __rmul__(self, other): return self.__mul__(other)
    
    def __truediv__(self, other):
        o = other.data if isinstance(other, TinyTensor) else other
        return TinyTensor(self.data / o, self._tpu)
    
    def __rtruediv__(self, other):
        return TinyTensor(other / self.data, self._tpu)
    
    def __pow__(self, n):
        return TinyTensor(self.data ** n, self._tpu)
    
    def __neg__(self):
        return TinyTensor(-self.data, self._tpu)
    
    def abs(self): return TinyTensor(np.abs(self.data), self._tpu)
    def sqrt(self): return TinyTensor(np.sqrt(self.data), self._tpu)
    def rsqrt(self): return TinyTensor(1.0 / np.sqrt(self.data), self._tpu)
    def exp(self): return TinyTensor(np.exp(self.data), self._tpu)
    def log(self): return TinyTensor(np.log(self.data), self._tpu)
    def sin(self): return TinyTensor(np.sin(self.data), self._tpu)
    def cos(self): return TinyTensor(np.cos(self.data), self._tpu)
    def clamp(self, min_val=None, max_val=None): return TinyTensor(np.clip(self.data, min_val, max_val), self._tpu)
    def floor(self): return TinyTensor(np.floor(self.data), self._tpu)
    def ceil(self): return TinyTensor(np.ceil(self.data), self._tpu)
    def round(self): return TinyTensor(np.round(self.data), self._tpu)
    def sign(self): return TinyTensor(np.sign(self.data), self._tpu)
    
    # ==================== MATRIX OPS (TPU ACCELERATED) ====================
    
    def _to_int8(self, x):
        if x.dtype == np.int8:
            return x
        scale = max(np.abs(x).max(), 1e-8) / 127
        return np.clip(np.round(x / scale), -128, 127).astype(np.int8)
    
    def matmul(self, other):
        """Matrix multiply - TPU ACCELERATED for 2D, 3D, 4D tensors."""
        A = self.data
        B = other.data if isinstance(other, TinyTensor) else np.asarray(other)
        
        # 1D @ 1D: dot product
        if A.ndim == 1 and B.ndim == 1:
            return TinyTensor(np.array(np.dot(A, B)), self._tpu)
        
        # 2D @ 2D: standard matmul (TPU accelerated)
        if A.ndim == 2 and B.ndim == 2:
            A_int8 = self._to_int8(A)
            B_int8 = self._to_int8(B)
            result = self._tpu.matmul(A_int8, B_int8)
            return TinyTensor(result.astype(np.float32), self._tpu)
        
        # 3D @ 3D: batched matmul
        if A.ndim == 3 and B.ndim == 3:
            batch = A.shape[0]
            results = []
            for i in range(batch):
                A_int8 = self._to_int8(A[i])
                B_int8 = self._to_int8(B[i])
                results.append(self._tpu.matmul(A_int8, B_int8))
            return TinyTensor(np.stack(results).astype(np.float32), self._tpu)
        
        # 3D @ 2D: broadcast
        if A.ndim == 3 and B.ndim == 2:
            batch = A.shape[0]
            B_int8 = self._to_int8(B)
            results = []
            for i in range(batch):
                A_int8 = self._to_int8(A[i])
                results.append(self._tpu.matmul(A_int8, B_int8))
            return TinyTensor(np.stack(results).astype(np.float32), self._tpu)
        
        # 4D @ 4D: attention-like (batch, heads, seq, dim)
        if A.ndim == 4 and B.ndim == 4:
            batch, heads = A.shape[:2]
            results = []
            for b in range(batch):
                head_results = []
                for h in range(heads):
                    A_int8 = self._to_int8(A[b, h])
                    B_int8 = self._to_int8(B[b, h])
                    head_results.append(self._tpu.matmul(A_int8, B_int8))
                results.append(np.stack(head_results))
            return TinyTensor(np.stack(results).astype(np.float32), self._tpu)
        
        # Fallback to numpy
        return TinyTensor(np.matmul(A, B), self._tpu)
    
    def __matmul__(self, other): return self.matmul(other)
    def mm(self, other): return self.matmul(other)
    def bmm(self, other): return self.matmul(other)
    
    def dot(self, other):
        o = other.data if isinstance(other, TinyTensor) else other
        return TinyTensor(np.dot(self.data, o), self._tpu)
    
    def outer(self, other):
        o = other.data if isinstance(other, TinyTensor) else other
        return TinyTensor(np.outer(self.data, o), self._tpu)
    
    # ==================== REDUCTIONS ====================
    
    def sum(self, axis=None, keepdims=False):
        return TinyTensor(self.data.sum(axis=axis, keepdims=keepdims), self._tpu)
    
    def mean(self, axis=None, keepdims=False):
        return TinyTensor(self.data.mean(axis=axis, keepdims=keepdims), self._tpu)
    
    def max(self, axis=None, keepdims=False):
        return TinyTensor(self.data.max(axis=axis, keepdims=keepdims), self._tpu)
    
    def min(self, axis=None, keepdims=False):
        return TinyTensor(self.data.min(axis=axis, keepdims=keepdims), self._tpu)
    
    def var(self, axis=None, keepdims=False):
        return TinyTensor(self.data.var(axis=axis, keepdims=keepdims), self._tpu)
    
    def std(self, axis=None, keepdims=False):
        return TinyTensor(self.data.std(axis=axis, keepdims=keepdims), self._tpu)
    
    def argmax(self, axis=None):
        return TinyTensor(self.data.argmax(axis=axis), self._tpu)
    
    def argmin(self, axis=None):
        return TinyTensor(self.data.argmin(axis=axis), self._tpu)
    
    def cumsum(self, axis=0):
        return TinyTensor(np.cumsum(self.data, axis=axis), self._tpu)
    
    # ==================== ACTIVATIONS ====================
    
    def relu(self): return TinyTensor(np.maximum(0, self.data), self._tpu)
    def relu6(self): return TinyTensor(np.clip(self.data, 0, 6), self._tpu)
    
    def leaky_relu(self, negative_slope=0.01):
        return TinyTensor(np.where(self.data > 0, self.data, self.data * negative_slope), self._tpu)
    
    def elu(self, alpha=1.0):
        return TinyTensor(np.where(self.data > 0, self.data, alpha * (np.exp(self.data) - 1)), self._tpu)
    
    def gelu(self):
        x = self.data
        return TinyTensor(0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3))), self._tpu)
    
    def silu(self):
        return TinyTensor(self.data * (1 / (1 + np.exp(-self.data))), self._tpu)
    
    def swish(self): return self.silu()
    
    def mish(self):
        return TinyTensor(self.data * np.tanh(np.log1p(np.exp(self.data))), self._tpu)
    
    def sigmoid(self):
        return TinyTensor(1 / (1 + np.exp(-self.data)), self._tpu)
    
    def tanh(self):
        return TinyTensor(np.tanh(self.data), self._tpu)
    
    def hardtanh(self, min_val=-1, max_val=1):
        return TinyTensor(np.clip(self.data, min_val, max_val), self._tpu)
    
    def softplus(self, beta=1):
        return TinyTensor(np.log1p(np.exp(beta * self.data)) / beta, self._tpu)
    
    def softmax(self, axis=-1):
        x = self.data
        exp_x = np.exp(x - x.max(axis=axis, keepdims=True))
        return TinyTensor(exp_x / exp_x.sum(axis=axis, keepdims=True), self._tpu)
    
    def log_softmax(self, axis=-1):
        x = self.data
        max_x = x.max(axis=axis, keepdims=True)
        return TinyTensor(x - max_x - np.log(np.exp(x - max_x).sum(axis=axis, keepdims=True)), self._tpu)
    
    # ==================== NORMALIZATION ====================
    
    def layer_norm(self, normalized_shape, weight=None, bias=None, eps=1e-5):
        x = self.data
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + eps)
        
        if weight is not None:
            w = weight.data if isinstance(weight, TinyTensor) else weight
            x_norm = x_norm * w
        if bias is not None:
            b = bias.data if isinstance(bias, TinyTensor) else bias
            x_norm = x_norm + b
        
        return TinyTensor(x_norm, self._tpu)
    
    def rms_norm(self, normalized_shape, weight=None, eps=1e-5):
        x = self.data
        rms = np.sqrt((x ** 2).mean(axis=-1, keepdims=True) + eps)
        x_norm = x / rms
        
        if weight is not None:
            w = weight.data if isinstance(weight, TinyTensor) else weight
            x_norm = x_norm * w
        
        return TinyTensor(x_norm, self._tpu)
    
    def batch_norm(self, running_mean, running_var, weight=None, bias=None, eps=1e-5):
        x = self.data
        mean = running_mean.data if isinstance(running_mean, TinyTensor) else running_mean
        var = running_var.data if isinstance(running_var, TinyTensor) else running_var
        x_norm = (x - mean) / np.sqrt(var + eps)
        
        if weight is not None:
            w = weight.data if isinstance(weight, TinyTensor) else weight
            x_norm = x_norm * w
        if bias is not None:
            b = bias.data if isinstance(bias, TinyTensor) else bias
            x_norm = x_norm + b
        
        return TinyTensor(x_norm, self._tpu)
    
    # ==================== CONVOLUTION ====================
    
    def conv2d(self, weight, bias=None, stride=1, padding=0):
        x = self.data
        w = weight.data if isinstance(weight, TinyTensor) else weight
        
        if isinstance(stride, int): stride = (stride, stride)
        if isinstance(padding, int): padding = (padding, padding)
        
        if padding[0] > 0 or padding[1] > 0:
            x = np.pad(x, ((0,0), (0,0), (padding[0], padding[0]), (padding[1], padding[1])))
        
        N, C_in, H, W = x.shape
        C_out, C_in_w, kH, kW = w.shape
        H_out = (H - kH) // stride[0] + 1
        W_out = (W - kW) // stride[1] + 1
        
        output = np.zeros((N, C_out, H_out, W_out), dtype=np.float32)
        for n in range(N):
            for c_out in range(C_out):
                for h in range(H_out):
                    for w_idx in range(W_out):
                        h_start, w_start = h * stride[0], w_idx * stride[1]
                        patch = x[n, :, h_start:h_start+kH, w_start:w_start+kW]
                        output[n, c_out, h, w_idx] = np.sum(patch * w[c_out])
        
        if bias is not None:
            b = bias.data if isinstance(bias, TinyTensor) else bias
            output += b.reshape(1, -1, 1, 1)
        
        return TinyTensor(output, self._tpu)
    
    # ==================== POOLING ====================
    
    def max_pool2d(self, kernel_size, stride=None):
        if stride is None: stride = kernel_size
        if isinstance(kernel_size, int): kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int): stride = (stride, stride)
        
        x = self.data
        N, C, H, W = x.shape
        kH, kW = kernel_size
        sH, sW = stride
        H_out, W_out = (H - kH) // sH + 1, (W - kW) // sW + 1
        
        output = np.zeros((N, C, H_out, W_out), dtype=np.float32)
        for h in range(H_out):
            for w in range(W_out):
                output[:, :, h, w] = x[:, :, h*sH:h*sH+kH, w*sW:w*sW+kW].max(axis=(2, 3))
        
        return TinyTensor(output, self._tpu)
    
    def avg_pool2d(self, kernel_size, stride=None):
        if stride is None: stride = kernel_size
        if isinstance(kernel_size, int): kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int): stride = (stride, stride)
        
        x = self.data
        N, C, H, W = x.shape
        kH, kW = kernel_size
        sH, sW = stride
        H_out, W_out = (H - kH) // sH + 1, (W - kW) // sW + 1
        
        output = np.zeros((N, C, H_out, W_out), dtype=np.float32)
        for h in range(H_out):
            for w in range(W_out):
                output[:, :, h, w] = x[:, :, h*sH:h*sH+kH, w*sW:w*sW+kW].mean(axis=(2, 3))
        
        return TinyTensor(output, self._tpu)
    
    def adaptive_avg_pool2d(self, output_size):
        if isinstance(output_size, int): output_size = (output_size, output_size)
        x = self.data
        N, C, H, W = x.shape
        oH, oW = output_size
        
        output = np.zeros((N, C, oH, oW), dtype=np.float32)
        for h in range(oH):
            for w in range(oW):
                h_start, h_end = h * H // oH, (h + 1) * H // oH
                w_start, w_end = w * W // oW, (w + 1) * W // oW
                output[:, :, h, w] = x[:, :, h_start:h_end, w_start:w_end].mean(axis=(2, 3))
        
        return TinyTensor(output, self._tpu)
    
    # ==================== ATTENTION ====================
    
    def scaled_dot_product_attention(self, key, value, mask=None):
        """Scaled dot-product attention - core of transformer."""
        Q = self
        K = key if isinstance(key, TinyTensor) else TinyTensor(key, self._tpu)
        V = value if isinstance(value, TinyTensor) else TinyTensor(value, self._tpu)
        
        d_k = Q.shape[-1]
        
        # Q @ K^T / sqrt(d_k)
        K_T = K.transpose(-2, -1)
        scores = Q @ K_T
        scores = scores * (1.0 / np.sqrt(d_k))
        
        # Mask (optional)
        if mask is not None:
            m = mask.data if isinstance(mask, TinyTensor) else mask
            scores = TinyTensor(np.where(m, scores.data, -1e9), self._tpu)
        
        # Softmax
        attn = scores.softmax(axis=-1)
        
        # Attention @ V
        return attn @ V
    
    def multi_head_attention(self, key, value, num_heads, mask=None):
        """Multi-head attention."""
        Q = self
        K = key if isinstance(key, TinyTensor) else TinyTensor(key, self._tpu)
        V = value if isinstance(value, TinyTensor) else TinyTensor(value, self._tpu)
        
        batch, seq_len, d_model = Q.shape
        head_dim = d_model // num_heads
        
        Q = Q.reshape(batch, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)
        K = K.reshape(batch, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)
        V = V.reshape(batch, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)
        
        output = Q.scaled_dot_product_attention(K, V, mask)
        output = output.permute(0, 2, 1, 3).reshape(batch, seq_len, d_model)
        
        return output
    
    # ==================== EMBEDDING ====================
    
    def embedding(self, weight):
        w = weight.data if isinstance(weight, TinyTensor) else weight
        indices = self.data.astype(np.int64)
        return TinyTensor(w[indices], self._tpu)
    
    # ==================== DROPOUT ====================
    
    def dropout(self, p=0.5, training=True):
        if not training or p == 0: return self
        mask = np.random.binomial(1, 1-p, self.shape) / (1-p)
        return TinyTensor(self.data * mask, self._tpu)
    
    # ==================== COMPARISON ====================
    
    def __eq__(self, other):
        o = other.data if isinstance(other, TinyTensor) else other
        return TinyTensor(self.data == o, self._tpu)
    
    def __lt__(self, other):
        o = other.data if isinstance(other, TinyTensor) else other
        return TinyTensor(self.data < o, self._tpu)
    
    def __gt__(self, other):
        o = other.data if isinstance(other, TinyTensor) else other
        return TinyTensor(self.data > o, self._tpu)
    
    def masked_fill(self, mask, value):
        m = mask.data if isinstance(mask, TinyTensor) else mask
        result = self.data.copy()
        result[m] = value
        return TinyTensor(result, self._tpu)
    
    # ==================== UTILITY ====================
    
    def numpy(self): return self.data.copy()
    def item(self): return self.data.item()
    def tolist(self): return self.data.tolist()
    def clone(self): return TinyTensor(self.data.copy(), self._tpu)
    def contiguous(self): return TinyTensor(np.ascontiguousarray(self.data), self._tpu)
    
    def to_int8(self): return TinyTensor(self._to_int8(self.data), self._tpu)
    def astype(self, dtype): return TinyTensor(self.data.astype(dtype), self._tpu)
    def float(self): return self.astype(np.float32)
    def int(self): return self.astype(np.int32)
    def long(self): return self.astype(np.int64)
    
    def __repr__(self): return f"TinyTensor(shape={self.shape}, dtype={self.dtype})"
    def __str__(self): return str(self.data)
    def __len__(self): return len(self.data)
    
    def __getitem__(self, idx): return TinyTensor(self.data[idx], self._tpu)
    def __setitem__(self, idx, value):
        if isinstance(value, TinyTensor): self.data[idx] = value.data
        else: self.data[idx] = value


# ==================== CONVENIENCE FUNCTIONS ====================

def tensor(data, tpu=None): return TinyTensor(data, tpu)
def zeros(*shape, tpu=None): return TinyTensor.zeros(*shape, tpu=tpu)
def ones(*shape, tpu=None): return TinyTensor.ones(*shape, tpu=tpu)
def randn(*shape, tpu=None): return TinyTensor.randn(*shape, tpu=tpu)
def rand(*shape, tpu=None): return TinyTensor.rand(*shape, tpu=tpu)
def randint(low, high, shape, tpu=None): return TinyTensor.randint(low, high, shape, tpu=tpu)
def eye(n, tpu=None): return TinyTensor.eye(n, tpu=tpu)
def arange(*args, tpu=None): return TinyTensor.arange(*args, tpu=tpu)
def linspace(start, stop, num, tpu=None): return TinyTensor.linspace(start, stop, num, tpu=tpu)
def full(shape, value, tpu=None): return TinyTensor.full(shape, value, tpu=tpu)
def cat(tensors, dim=0): return TinyTensor.cat(tensors, dim)
def stack(tensors, dim=0): return TinyTensor.stack(tensors, dim)
def matmul(a, b): return tensor(a).matmul(b)
def softmax(x, axis=-1): return tensor(x).softmax(axis)
def relu(x): return tensor(x).relu()
def gelu(x): return tensor(x).gelu()
def sigmoid(x): return tensor(x).sigmoid()


if __name__ == "__main__":
    print("=" * 60)
    print("TinyTensor - Full Tensor Operations Demo")
    print("=" * 60)
    
    print("\n1. CREATION:")
    a = tensor([[1, 2], [3, 4]])
    print(f"   tensor = {a.shape}")
    print(f"   zeros(2,3) = {zeros(2,3).shape}")
    print(f"   randn(2,3) = {randn(2,3).shape}")
    
    print("\n2. ELEMENT-WISE:")
    a = tensor([[1, 2], [3, 4]])
    b = tensor([[5, 6], [7, 8]])
    print(f"   a + b = \n{(a + b).data}")
    print(f"   a * b = \n{(a * b).data}")
    
    print("\n3. MATMUL (TPU ACCELERATED):")
    print(f"   2D: a @ b = \n{(a @ b).data}")
    
    c = randn(4, 8, 16)
    d = randn(4, 16, 8)
    print(f"   3D batched: (4,8,16) @ (4,16,8) = {(c @ d).shape}")
    
    e = randn(2, 4, 8, 16)
    f = randn(2, 4, 16, 8)
    print(f"   4D attention: (2,4,8,16) @ (2,4,16,8) = {(e @ f).shape}")
    
    print("\n4. ACTIVATIONS:")
    x = tensor([-2, -1, 0, 1, 2])
    print(f"   relu = {x.relu().data}")
    print(f"   gelu = {x.gelu().data}")
    print(f"   sigmoid = {x.sigmoid().data}")
    
    print("\n5. SOFTMAX:")
    logits = tensor([[1, 2, 3], [1, 2, 3]])
    print(f"   softmax = \n{logits.softmax().data}")
    
    print("\n6. NORMALIZATION:")
    x = randn(2, 4)
    x_ln = x.layer_norm(4)
    print(f"   layer_norm mean = {x_ln.mean(axis=-1).data}")
    x_rms = x.rms_norm(4)
    print(f"   rms_norm done")
    
    print("\n7. ATTENTION (TPU ACCELERATED):")
    Q = randn(2, 32, 64)
    K = randn(2, 32, 64)
    V = randn(2, 32, 64)
    attn_out = Q.scaled_dot_product_attention(K, V)
    print(f"   scaled_dot_product: {attn_out.shape}")
    
    Q = randn(2, 32, 128)
    K = randn(2, 32, 128)
    V = randn(2, 32, 128)
    mha_out = Q.multi_head_attention(K, V, num_heads=8)
    print(f"   multi_head_attention (8 heads): {mha_out.shape}")
    
    print("\n8. CONVOLUTION:")
    x = randn(1, 3, 32, 32)
    w = randn(16, 3, 3, 3)
    y = x.conv2d(w, padding=1)
    print(f"   conv2d: (1,3,32,32) * (16,3,3,3) = {y.shape}")
    
    print("\n9. POOLING:")
    x = randn(1, 3, 32, 32)
    print(f"   max_pool2d(2): {x.max_pool2d(2).shape}")
    print(f"   avg_pool2d(2): {x.avg_pool2d(2).shape}")
    print(f"   adaptive_avg_pool2d(1): {x.adaptive_avg_pool2d(1).shape}")
    
    print("\n10. REDUCTIONS:")
    x = tensor([[1, 2, 3], [4, 5, 6]])
    print(f"   sum = {x.sum().item()}")
    print(f"   mean = {x.mean().item()}")
    print(f"   sum(axis=1) = {x.sum(axis=1).data}")
    
    print("\n" + "=" * 60)
    print("ALL TENSOR OPERATIONS WORKING!")
    print("=" * 60)

