"""TINYTPU INTERACTIVE TESTER"""
import numpy as np
import time
import sys

try:
    from tinytpu import TinyTPU
    TPU = TinyTPU(backend="simulator", array_size=16)
except ImportError as e:
    print(f"ERROR: {e}")
    sys.exit(1)

def parse_matrix(s):
    s = s.strip()
    if s.startswith("random"):
        p = s.split(); return np.random.randint(-128, 127, (int(p[1]), int(p[2])), dtype=np.int8)
    if s.startswith("zeros"):
        p = s.split(); return np.zeros((int(p[1]), int(p[2])), dtype=np.int8)
    if s.startswith("ones"):
        p = s.split(); return np.ones((int(p[1]), int(p[2])), dtype=np.int8)
    if s.startswith("eye"):
        p = s.split(); return np.eye(int(p[1]), dtype=np.int8)
    if s.startswith("full"):
        p = s.split(); return np.full((int(p[1]), int(p[2])), int(p[3]), dtype=np.int8)
    if s.startswith("[["):
        return np.array(eval(s), dtype=np.int8)
    if ";" in s:
        return np.array([[int(x) for x in r.split()] for r in s.split(";")], dtype=np.int8)
    raise ValueError(f"Cannot parse: {s}")

def print_matrix(name, m):
    print(f"\n{name} {m.shape}:")
    if m.size <= 64:
        for row in m: print("  [" + " ".join(f"{x:4d}" for x in row) + "]")
    else:
        print(f"  (large matrix, corners: [{m[0,0]}...{m[0,-1]}] ... [{m[-1,0]}...{m[-1,-1]}])")

def main():
    print("="*60)
    print("  TINYTPU INTERACTIVE TESTER")
    print("  Commands: matmul, random, stress, boundary, custom, info, help, quit")
    print("="*60)
    
    while True:
        try:
            line = input("\ntinytpu> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        
        if not line: continue
        if line in ["quit","exit","q"]: break
        
        parts = line.split()
        cmd = parts[0]
        
        try:
            if cmd == "matmul":
                A = parse_matrix(parts[1])
                B = parse_matrix(parts[2])
                print_matrix("A", A)
                print_matrix("B", B)
                t0 = time.perf_counter()
                C_tpu = TPU.matmul(A, B)
                t1 = time.perf_counter()
                C_np = np.matmul(A.astype(np.int64), B.astype(np.int64))
                print_matrix("Result", C_tpu)
                match = np.array_equal(C_tpu, C_np)
                print(f"\n{'='*50}")
                print(f"Match: {'YES' if match else 'NO'}")
                print(f"Time: {(t1-t0)*1000:.3f} ms")
            
            elif cmd == "random":
                m,k,n = int(parts[1]), int(parts[2]), int(parts[3])
                A = np.random.randint(-128, 127, (m,k), dtype=np.int8)
                B = np.random.randint(-128, 127, (k,n), dtype=np.int8)
                t0 = time.perf_counter()
                C_tpu = TPU.matmul(A, B)
                t1 = time.perf_counter()
                C_np = np.matmul(A.astype(np.int64), B.astype(np.int64))
                match = np.array_equal(C_tpu, C_np)
                print(f"({m},{k}) @ ({k},{n}): {'MATCH' if match else 'MISMATCH'} in {(t1-t0)*1000:.1f}ms")
            
            elif cmd == "stress":
                m,k,n,r = int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])
                passed = 0
                t_start = time.perf_counter()
                for i in range(r):
                    A = np.random.randint(-128, 127, (m,k), dtype=np.int8)
                    B = np.random.randint(-128, 127, (k,n), dtype=np.int8)
                    if np.array_equal(TPU.matmul(A,B), np.matmul(A.astype(np.int64), B.astype(np.int64))):
                        passed += 1
                    if (i+1) % max(1, r//10) == 0:
                        print(f"  {i+1}/{r}: {passed} passed")
                t_total = time.perf_counter() - t_start
                print(f"\nResult: {passed}/{r} passed in {t_total:.2f}s")
            
            elif cmd == "boundary":
                bounds = [-128, -1, 0, 1, 127]
                print("\nBoundary tests (4x4 matrices):")
                for a in bounds:
                    for b in bounds:
                        A = np.full((4,4), a, dtype=np.int8)
                        B = np.full((4,4), b, dtype=np.int8)
                        C = TPU.matmul(A, B)
                        exp = a * b * 4
                        status = "OK" if C[0,0] == exp else "FAIL"
                        print(f"  {a:4d} x {b:4d} x 4 = {C[0,0]:8d} (exp {exp:8d}) {status}")
            
            elif cmd == "custom":
                print("Enter A rows (empty line to finish):")
                rows_a = []
                while True:
                    r = input("  A> ").strip()
                    if not r: break
                    rows_a.append([int(x) for x in r.split()])
                print("Enter B rows (empty line to finish):")
                rows_b = []
                while True:
                    r = input("  B> ").strip()
                    if not r: break
                    rows_b.append([int(x) for x in r.split()])
                A = np.array(rows_a, dtype=np.int8)
                B = np.array(rows_b, dtype=np.int8)
                print_matrix("A", A)
                print_matrix("B", B)
                C = TPU.matmul(A, B)
                print_matrix("Result", C)
                C_np = np.matmul(A.astype(np.int64), B.astype(np.int64))
                print(f"Match: {'YES' if np.array_equal(C, C_np) else 'NO'}")
            
            elif cmd == "info":
                print(f"Backend: {TPU.backend_name}")
                print(f"Array size: {TPU.array_size}")
                A = np.random.randint(-128, 127, (128,128), dtype=np.int8)
                B = np.random.randint(-128, 127, (128,128), dtype=np.int8)
                t0 = time.perf_counter()
                for _ in range(10): TPU.matmul(A,B)
                print(f"128x128 avg: {(time.perf_counter()-t0)/10*1000:.1f}ms")
            
            elif cmd == "help":
                print("""
Commands:
  matmul A B     - Matrix multiply (e.g., matmul [[1,2],[3,4]] [[5,6],[7,8]])
  random M K N   - Test random MxK @ KxN
  stress M K N R - R random tests
  boundary       - INT8 boundary tests
  custom         - Enter matrices manually
  info           - Show TPU info
  quit           - Exit

Matrix formats:
  [[1,2],[3,4]]  - Python list
  random M N     - Random matrix
  zeros M N      - Zero matrix
  eye N          - Identity
  full M N V     - Filled with V
""")
            else:
                print(f"Unknown: {cmd}. Try 'help'")
        
        except Exception as e:
            print(f"Error: {e}")
    
    TPU.close()
    print("Bye!")

if __name__ == "__main__":
    main()
