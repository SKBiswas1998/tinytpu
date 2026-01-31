import numpy as np
from tinytpu import TinyTPU

def main():
    print("TinyTPU Basic Example")
    print("=" * 40)
    
    tpu = TinyTPU(backend="simulator")
    print(f"Backend: {tpu.backend_name}")
    
    A = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]], dtype=np.int8)
    B = np.eye(4, dtype=np.int8)
    C = tpu.matmul(A, B)
    
    print(f"\nA @ I = A:")
    print(C)
    
    results = tpu.benchmark(size=64, iterations=50)
    print(f"\nBenchmark: {results['gops']:.2f} GOPS")
    
    tpu.close()

if __name__ == "__main__":
    main()
