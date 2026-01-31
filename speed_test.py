import numpy as np
import time
import sys
sys.path.insert(0, 'software')
from tinytpu.tpu_v2 import TinyTPU

print('=' * 70)
print('TINYTPU SPEED TEST')
print('=' * 70)

tpu = TinyTPU()

def bench(fn, warmup=3, runs=10):
    for _ in range(warmup): fn()
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return np.median(times) * 1000

# Matmul
print('\n[MATMUL]')
print('Size            Time         GFLOPS')
print('-' * 40)
for N in [256, 512, 1024, 2048]:
    A = tpu.randn(N, N)
    B = tpu.randn(N, N)
    t = bench(lambda: tpu.matmul(A, B))
    gflops = (2 * N**3) / (t/1000) / 1e9
    print(f'{N}x{N}          {t:.2f}ms       {gflops:.1f}')

# Neural ops
print('\n[NEURAL OPS] (1000x768 tensor)')
print('Op              Time')
print('-' * 30)
x = tpu.randn(1000, 768)
for op in ['relu', 'gelu', 'softmax', 'layer_norm']:
    fn = getattr(tpu, op)
    t = bench(lambda: fn(x))
    print(f'{op:15} {t:.3f}ms')

# LLM estimate
print('\n[LLM ESTIMATE]')
x = tpu.randn(128, 768)
W1 = tpu.randn(768, 3072)
W2 = tpu.randn(3072, 768)
t_layer = bench(lambda: tpu.matmul(tpu.gelu(tpu.matmul(x, W1)), W2))
print(f'Single transformer MLP: {t_layer:.2f}ms')
print(f'Estimated GPT-2 (12 layers): {12 * t_layer:.0f}ms/token')
print(f'Estimated speed: {1000 / (12 * t_layer):.1f} tok/s')

print('\n' + '=' * 70)
