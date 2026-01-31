"""
TinyTPU Benchmark Suite
=======================
Compare TinyTPU against other tensor libraries.

Libraries tested:
- NumPy (baseline)
- TinyTPU (our library)
- PyTorch
- TensorFlow (if available)
- JAX (if available)
"""

import numpy as np
import time
import sys
from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class BenchmarkResult:
    name: str
    operation: str
    size: Tuple[int, ...]
    time_ms: float
    gflops: float
    speedup: float = 1.0

def benchmark_fn(fn, warmup=3, iterations=10):
    """Benchmark a function."""
    # Warmup
    for _ in range(warmup):
        fn()
    
    # Benchmark
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    
    return np.median(times) * 1000  # ms

# ============================================================
# NUMPY BASELINE
# ============================================================

def benchmark_numpy(sizes):
    print("\n[NumPy]")
    results = []
    
    for M, K, N in sizes:
        A = np.random.randn(M, K).astype(np.float32)
        B = np.random.randn(K, N).astype(np.float32)
        
        time_ms = benchmark_fn(lambda: np.matmul(A, B))
        gflops = (2 * M * K * N) / (time_ms / 1000) / 1e9
        
        results.append(BenchmarkResult("NumPy", "matmul", (M, K, N), time_ms, gflops))
        print(f"  {M}x{K} @ {K}x{N}: {time_ms:.2f}ms, {gflops:.2f} GFLOPS")
    
    return results

# ============================================================
# TINYTPU
# ============================================================

def benchmark_tinytpu(sizes):
    print("\n[TinyTPU]")
    
    try:
        from tinytpu import TinyTPU
        tpu = TinyTPU(backend="auto")
        print(f"  Backend: {tpu.backend_name} ({tpu.device})")
    except ImportError:
        # Try local import
        sys.path.insert(0, 'software')
        from tinytpu import TinyTPU
        tpu = TinyTPU(backend="auto")
        print(f"  Backend: {tpu.backend_name} ({tpu.device})")
    
    results = []
    
    for M, K, N in sizes:
        A = np.random.randn(M, K).astype(np.float32)
        B = np.random.randn(K, N).astype(np.float32)
        
        time_ms = benchmark_fn(lambda: tpu.matmul_float(A, B))
        gflops = (2 * M * K * N) / (time_ms / 1000) / 1e9
        
        results.append(BenchmarkResult("TinyTPU", "matmul", (M, K, N), time_ms, gflops))
        print(f"  {M}x{K} @ {K}x{N}: {time_ms:.2f}ms, {gflops:.2f} GFLOPS")
    
    return results

# ============================================================
# PYTORCH
# ============================================================

def benchmark_pytorch(sizes):
    print("\n[PyTorch]")
    
    try:
        import torch
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"  Device: {device}")
        if device.type == 'cpu':
            torch.set_num_threads(torch.get_num_threads())
            print(f"  Threads: {torch.get_num_threads()}")
    except ImportError:
        print("  Not installed (pip install torch)")
        return []
    
    results = []
    
    for M, K, N in sizes:
        A = torch.randn(M, K, device=device)
        B = torch.randn(K, N, device=device)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        def run():
            C = torch.matmul(A, B)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            return C
        
        time_ms = benchmark_fn(run)
        gflops = (2 * M * K * N) / (time_ms / 1000) / 1e9
        
        results.append(BenchmarkResult("PyTorch", "matmul", (M, K, N), time_ms, gflops))
        print(f"  {M}x{K} @ {K}x{N}: {time_ms:.2f}ms, {gflops:.2f} GFLOPS")
    
    return results

# ============================================================
# TENSORFLOW
# ============================================================

def benchmark_tensorflow(sizes):
    print("\n[TensorFlow]")
    
    try:
        import os
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        device = 'GPU' if gpus else 'CPU'
        print(f"  Device: {device}")
    except ImportError:
        print("  Not installed (pip install tensorflow)")
        return []
    
    results = []
    
    for M, K, N in sizes:
        A = tf.random.normal((M, K))
        B = tf.random.normal((K, N))
        
        @tf.function
        def matmul():
            return tf.matmul(A, B)
        
        # Warmup
        matmul()
        
        time_ms = benchmark_fn(lambda: matmul().numpy())
        gflops = (2 * M * K * N) / (time_ms / 1000) / 1e9
        
        results.append(BenchmarkResult("TensorFlow", "matmul", (M, K, N), time_ms, gflops))
        print(f"  {M}x{K} @ {K}x{N}: {time_ms:.2f}ms, {gflops:.2f} GFLOPS")
    
    return results

# ============================================================
# JAX
# ============================================================

def benchmark_jax(sizes):
    print("\n[JAX]")
    
    try:
        import jax
        import jax.numpy as jnp
        devices = jax.devices()
        print(f"  Devices: {[d.platform for d in devices]}")
    except ImportError:
        print("  Not installed (pip install jax)")
        return []
    
    results = []
    
    for M, K, N in sizes:
        A = jnp.array(np.random.randn(M, K).astype(np.float32))
        B = jnp.array(np.random.randn(K, N).astype(np.float32))
        
        @jax.jit
        def matmul(a, b):
            return jnp.matmul(a, b)
        
        # Warmup (JIT compile)
        matmul(A, B).block_until_ready()
        
        def run():
            return matmul(A, B).block_until_ready()
        
        time_ms = benchmark_fn(run)
        gflops = (2 * M * K * N) / (time_ms / 1000) / 1e9
        
        results.append(BenchmarkResult("JAX", "matmul", (M, K, N), time_ms, gflops))
        print(f"  {M}x{K} @ {K}x{N}: {time_ms:.2f}ms, {gflops:.2f} GFLOPS")
    
    return results

# ============================================================
# ADDITIONAL OPS BENCHMARK
# ============================================================

def benchmark_ops():
    """Benchmark additional operations."""
    print("\n" + "=" * 70)
    print("OPERATION BENCHMARKS (1000x768 tensor)")
    print("=" * 70)
    
    size = (1000, 768)
    iterations = 100
    
    # NumPy
    print("\n[NumPy]")
    x_np = np.random.randn(*size).astype(np.float32)
    
    ops_np = {
        'relu': lambda: np.maximum(0, x_np),
        'softmax': lambda: np.exp(x_np - x_np.max(-1, keepdims=True)) / np.exp(x_np - x_np.max(-1, keepdims=True)).sum(-1, keepdims=True),
        'layer_norm': lambda: (x_np - x_np.mean(-1, keepdims=True)) / np.sqrt(x_np.var(-1, keepdims=True) + 1e-5),
        'gelu': lambda: 0.5 * x_np * (1 + np.tanh(np.sqrt(2/np.pi) * (x_np + 0.044715 * x_np**3))),
    }
    
    np_times = {}
    for name, fn in ops_np.items():
        t = benchmark_fn(fn, warmup=5, iterations=iterations)
        np_times[name] = t
        print(f"  {name}: {t:.3f}ms")
    
    # TinyTPU
    print("\n[TinyTPU]")
    try:
        sys.path.insert(0, 'software')
        from tinytpu import TinyTPU
        tpu = TinyTPU(backend="auto")
        
        ops_tpu = {
            'relu': lambda: tpu.relu(x_np),
            'softmax': lambda: tpu.softmax(x_np),
            'layer_norm': lambda: tpu.layer_norm(x_np),
            'gelu': lambda: tpu.gelu(x_np),
        }
        
        for name, fn in ops_tpu.items():
            t = benchmark_fn(fn, warmup=5, iterations=iterations)
            speedup = np_times[name] / t
            print(f"  {name}: {t:.3f}ms ({speedup:.2f}x vs NumPy)")
    except Exception as e:
        print(f"  Error: {e}")
    
    # PyTorch
    print("\n[PyTorch]")
    try:
        import torch
        import torch.nn.functional as F
        x_pt = torch.from_numpy(x_np)
        
        ops_pt = {
            'relu': lambda: F.relu(x_pt),
            'softmax': lambda: F.softmax(x_pt, dim=-1),
            'layer_norm': lambda: F.layer_norm(x_pt, (768,)),
            'gelu': lambda: F.gelu(x_pt),
        }
        
        for name, fn in ops_pt.items():
            t = benchmark_fn(fn, warmup=5, iterations=iterations)
            speedup = np_times[name] / t
            print(f"  {name}: {t:.3f}ms ({speedup:.2f}x vs NumPy)")
    except ImportError:
        print("  Not installed")

# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("TINYTPU BENCHMARK SUITE")
    print("=" * 70)
    print("Comparing tensor libraries for matrix multiplication")
    print("=" * 70)
    
    # Matrix sizes to test
    sizes = [
        (128, 128, 128),
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
    ]
    
    all_results = []
    
    # Run benchmarks
    all_results.extend(benchmark_numpy(sizes))
    all_results.extend(benchmark_tinytpu(sizes))
    all_results.extend(benchmark_pytorch(sizes))
    all_results.extend(benchmark_tensorflow(sizes))
    all_results.extend(benchmark_jax(sizes))
    
    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY: Matrix Multiplication (GFLOPS, higher is better)")
    print("=" * 70)
    
    # Group by size
    print(f"\n{'Size':<20}", end="")
    libs = ["NumPy", "TinyTPU", "PyTorch", "TensorFlow", "JAX"]
    for lib in libs:
        print(f"{lib:<12}", end="")
    print()
    print("-" * 80)
    
    for size in sizes:
        print(f"{str(size):<20}", end="")
        for lib in libs:
            result = next((r for r in all_results if r.name == lib and r.size == size), None)
            if result:
                print(f"{result.gflops:<12.2f}", end="")
            else:
                print(f"{'N/A':<12}", end="")
        print()
    
    # Find baseline (NumPy) for speedup calculation
    print("\n" + "=" * 70)
    print("SPEEDUP vs NumPy")
    print("=" * 70)
    
    print(f"\n{'Size':<20}", end="")
    for lib in libs[1:]:  # Skip NumPy
        print(f"{lib:<12}", end="")
    print()
    print("-" * 70)
    
    for size in sizes:
        numpy_result = next((r for r in all_results if r.name == "NumPy" and r.size == size), None)
        if not numpy_result:
            continue
        
        print(f"{str(size):<20}", end="")
        for lib in libs[1:]:
            result = next((r for r in all_results if r.name == lib and r.size == size), None)
            if result:
                speedup = result.gflops / numpy_result.gflops
                print(f"{speedup:<12.2f}x", end="")
            else:
                print(f"{'N/A':<12}", end="")
        print()
    
    # Additional ops
    benchmark_ops()
    
    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()
