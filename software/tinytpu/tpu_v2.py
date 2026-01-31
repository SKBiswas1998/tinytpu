"""
TinyTPU v2 - Zero-Copy Tensor Operations
========================================
Optimized to avoid numpy<->torch conversion overhead.
"""

import numpy as np
import time

class TinyTPU:
    """
    TinyTPU with native tensor support.
    
    Usage:
        tpu = TinyTPU()
        
        # NumPy (with conversion overhead)
        C = tpu.matmul(A_numpy, B_numpy)
        
        # Native PyTorch (zero overhead)
        C = tpu.matmul(A_torch, B_torch)
    """
    
    def __init__(self, backend="auto", device=None):
        self._torch = None
        self._device = None
        self._backend = "numpy"
        
        # Try PyTorch
        try:
            import torch
            self._torch = torch
            
            if device:
                self._device = torch.device(device)
            elif torch.cuda.is_available():
                self._device = torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self._device = torch.device('mps')
            else:
                self._device = torch.device('cpu')
            
            if self._device.type == 'cpu':
                import os
                torch.set_num_threads(os.cpu_count() or 4)
            
            self._backend = "pytorch"
            print(f"TinyTPU: PyTorch backend ({self._device})")
            
        except ImportError:
            print("TinyTPU: NumPy backend (install torch for 10x speedup)")
    
    @property
    def backend(self): return self._backend
    
    @property 
    def device(self): return str(self._device) if self._device else "cpu"
    
    def _to_tensor(self, x):
        """Convert to native tensor if needed."""
        if self._torch is None:
            return np.asarray(x, dtype=np.float32)
        
        torch = self._torch
        if isinstance(x, torch.Tensor):
            return x.to(self._device)
        else:
            return torch.from_numpy(np.asarray(x, dtype=np.float32)).to(self._device)
    
    def _to_output(self, x, like=None):
        """Convert output to match input type."""
        if self._torch is None:
            return x
        
        # If input was numpy, return numpy
        if like is not None and isinstance(like, np.ndarray):
            if self._device.type != 'cpu':
                return x.cpu().numpy()
            return x.numpy()
        return x
    
    def tensor(self, data, dtype=None):
        """Create a native tensor (avoids conversion overhead)."""
        if self._torch:
            return self._torch.tensor(data, dtype=dtype or self._torch.float32, device=self._device)
        return np.array(data, dtype=dtype or np.float32)
    
    def randn(self, *shape):
        """Create random tensor."""
        if self._torch:
            return self._torch.randn(*shape, device=self._device)
        return np.random.randn(*shape).astype(np.float32)
    
    def zeros(self, *shape):
        if self._torch:
            return self._torch.zeros(*shape, device=self._device)
        return np.zeros(shape, dtype=np.float32)
    
    # ============================================================
    # CORE OPERATIONS
    # ============================================================
    
    def matmul(self, A, B):
        """Matrix multiplication."""
        is_numpy = isinstance(A, np.ndarray)
        A, B = self._to_tensor(A), self._to_tensor(B)
        
        if self._torch:
            C = self._torch.matmul(A, B)
        else:
            C = np.matmul(A, B)
        
        return self._to_output(C, A if is_numpy else None)
    
    def softmax(self, x, axis=-1):
        is_numpy = isinstance(x, np.ndarray)
        x = self._to_tensor(x)
        
        if self._torch:
            y = self._torch.nn.functional.softmax(x, dim=axis)
        else:
            exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
            y = exp_x / np.sum(exp_x, axis=axis, keepdims=True)
        
        return self._to_output(y, x if is_numpy else None)
    
    def relu(self, x):
        is_numpy = isinstance(x, np.ndarray)
        x = self._to_tensor(x)
        
        if self._torch:
            y = self._torch.relu(x)
        else:
            y = np.maximum(0, x)
        
        return self._to_output(y, x if is_numpy else None)
    
    def gelu(self, x):
        is_numpy = isinstance(x, np.ndarray)
        x = self._to_tensor(x)
        
        if self._torch:
            y = self._torch.nn.functional.gelu(x)
        else:
            y = 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))
        
        return self._to_output(y, x if is_numpy else None)
    
    def layer_norm(self, x, normalized_shape=None, weight=None, bias=None, eps=1e-5):
        is_numpy = isinstance(x, np.ndarray)
        x = self._to_tensor(x)
        
        if self._torch:
            if normalized_shape is None:
                normalized_shape = (x.shape[-1],)
            y = self._torch.nn.functional.layer_norm(x, normalized_shape, weight, bias, eps)
        else:
            mean = x.mean(axis=-1, keepdims=True)
            var = x.var(axis=-1, keepdims=True)
            y = (x - mean) / np.sqrt(var + eps)
            if weight is not None: y = y * weight
            if bias is not None: y = y + bias
        
        return self._to_output(y, x if is_numpy else None)
    
    def embedding(self, weight, indices):
        """Embedding lookup."""
        is_numpy = isinstance(weight, np.ndarray)
        weight = self._to_tensor(weight)
        
        if self._torch:
            if not isinstance(indices, self._torch.Tensor):
                indices = self._torch.tensor(indices, dtype=self._torch.long, device=self._device)
            y = self._torch.nn.functional.embedding(indices, weight)
        else:
            y = weight[indices]
        
        return self._to_output(y, weight if is_numpy else None)


# ============================================================
# BENCHMARK
# ============================================================

def bench(fn, warmup=3, runs=10):
    for _ in range(warmup): fn()
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return np.median(times) * 1000


if __name__ == "__main__":
    print("=" * 70)
    print("TINYTPU v2 BENCHMARK - Native Tensor Support")
    print("=" * 70)
    
    tpu = TinyTPU()
    
    sizes = [(256, 256), (512, 512), (1024, 1024), (2048, 2048)]
    
    print("\n" + "=" * 70)
    print("MATMUL: NumPy input vs Native tensor input")
    print("=" * 70)
    
    print(f"\n{'Size':<15} {'NumPy→TPU':<15} {'Native TPU':<15} {'Speedup':<10}")
    print("-" * 55)
    
    for M, N in sizes:
        # NumPy input (with conversion)
        A_np = np.random.randn(M, N).astype(np.float32)
        B_np = np.random.randn(N, M).astype(np.float32)
        t_numpy = bench(lambda: tpu.matmul(A_np, B_np))
        
        # Native tensor input (no conversion)
        A_native = tpu.randn(M, N)
        B_native = tpu.randn(N, M)
        t_native = bench(lambda: tpu.matmul(A_native, B_native))
        
        speedup = t_numpy / t_native
        print(f"{M}x{N:<10} {t_numpy:<15.2f} {t_native:<15.2f} {speedup:<10.2f}x")
    
    print("\n" + "=" * 70)
    print("NEURAL OPS: NumPy input vs Native tensor")
    print("=" * 70)
    
    x_np = np.random.randn(1000, 768).astype(np.float32)
    x_native = tpu.randn(1000, 768)
    
    ops = ['relu', 'softmax', 'gelu', 'layer_norm']
    
    print(f"\n{'Op':<15} {'NumPy→TPU':<15} {'Native TPU':<15} {'Speedup':<10}")
    print("-" * 55)
    
    for op in ops:
        fn = getattr(tpu, op)
        t_numpy = bench(lambda: fn(x_np))
        t_native = bench(lambda: fn(x_native))
        speedup = t_numpy / t_native
        print(f"{op:<15} {t_numpy:<15.3f} {t_native:<15.3f} {speedup:<10.2f}x")
    
    print("\n" + "=" * 70)
    print("COMPARISON: TinyTPU Native vs PyTorch Direct")
    print("=" * 70)
    
    try:
        import torch
        
        print(f"\n{'Size':<15} {'TinyTPU':<15} {'PyTorch':<15} {'Ratio':<10}")
        print("-" * 55)
        
        for M, N in sizes:
            A_tpu = tpu.randn(M, N)
            B_tpu = tpu.randn(N, M)
            t_tpu = bench(lambda: tpu.matmul(A_tpu, B_tpu))
            
            A_pt = torch.randn(M, N)
            B_pt = torch.randn(N, M)
            t_pt = bench(lambda: torch.matmul(A_pt, B_pt))
            
            ratio = t_tpu / t_pt
            print(f"{M}x{N:<10} {t_tpu:<15.2f} {t_pt:<15.2f} {ratio:<10.2f}x")
        
        print("\n[Neural Ops]")
        print(f"{'Op':<15} {'TinyTPU':<15} {'PyTorch':<15} {'Ratio':<10}")
        print("-" * 55)
        
        x_tpu = tpu.randn(1000, 768)
        x_pt = torch.randn(1000, 768)
        
        for op, pt_fn in [
            ('relu', lambda: torch.relu(x_pt)),
            ('softmax', lambda: torch.nn.functional.softmax(x_pt, dim=-1)),
            ('gelu', lambda: torch.nn.functional.gelu(x_pt)),
            ('layer_norm', lambda: torch.nn.functional.layer_norm(x_pt, (768,))),
        ]:
            fn = getattr(tpu, op)
            t_tpu = bench(lambda: fn(x_tpu))
            t_pt = bench(pt_fn)
            ratio = t_tpu / t_pt
            print(f"{op:<15} {t_tpu:<15.3f} {t_pt:<15.3f} {ratio:<10.2f}x")
            
    except ImportError:
        print("PyTorch not installed")
    
    print("\n" + "=" * 70)
    print("DONE!")
    print("=" * 70)
