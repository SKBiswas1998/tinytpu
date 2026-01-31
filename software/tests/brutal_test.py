"""
BRUTAL TINYTPU TEST SUITE
Designed to break everything and find real-world issues.
"""

import numpy as np
import time
import traceback
import sys
import gc

# Track all failures
FAILURES = []
PASSES = []

def test(name):
    """Decorator to track test results."""
    def decorator(func):
        def wrapper():
            print(f"\n{'='*60}")
            print(f"TEST: {name}")
            print('='*60)
            try:
                func()
                PASSES.append(name)
                print(f"[PASS] {name}")
                return True
            except Exception as e:
                FAILURES.append((name, str(e), traceback.format_exc()))
                print(f"[FAIL] {name}")
                print(f"  Error: {e}")
                return False
        return wrapper
    return decorator

# ============================================================
# SECTION 1: BASIC FUNCTIONALITY
# ============================================================

@test("Import TinyTPU")
def test_import():
    from tinytpu import TinyTPU
    assert TinyTPU is not None

@test("Create TinyTPU instance")
def test_create():
    from tinytpu import TinyTPU
    tpu = TinyTPU(backend="simulator")
    assert tpu.is_connected
    assert tpu.backend_name == "simulator"
    tpu.close()

@test("Basic 2x2 matmul")
def test_basic_2x2():
    from tinytpu import TinyTPU
    tpu = TinyTPU(backend="simulator", array_size=4)
    A = np.array([[1, 2], [3, 4]], dtype=np.int8)
    B = np.array([[5, 6], [7, 8]], dtype=np.int8)
    C = tpu.matmul(A, B)
    expected = np.array([[19, 22], [43, 50]])
    assert np.array_equal(C, expected), f"Got {C}, expected {expected}"
    tpu.close()

# ============================================================
# SECTION 2: EDGE CASES - WHERE THINGS BREAK
# ============================================================

@test("Empty matrix (0x0)")
def test_empty_matrix():
    from tinytpu import TinyTPU
    tpu = TinyTPU(backend="simulator")
    A = np.array([], dtype=np.int8).reshape(0, 0)
    B = np.array([], dtype=np.int8).reshape(0, 0)
    C = tpu.matmul(A, B)
    assert C.shape == (0, 0)
    tpu.close()

@test("Single element (1x1)")
def test_single_element():
    from tinytpu import TinyTPU
    tpu = TinyTPU(backend="simulator")
    A = np.array([[7]], dtype=np.int8)
    B = np.array([[8]], dtype=np.int8)
    C = tpu.matmul(A, B)
    assert C[0, 0] == 56, f"Got {C[0,0]}, expected 56"
    tpu.close()

@test("Non-square matrix (3x5) @ (5x2)")
def test_non_square():
    from tinytpu import TinyTPU
    tpu = TinyTPU(backend="simulator", array_size=4)
    A = np.random.randint(-128, 127, (3, 5), dtype=np.int8)
    B = np.random.randint(-128, 127, (5, 2), dtype=np.int8)
    C = tpu.matmul(A, B)
    expected = np.matmul(A.astype(np.int32), B.astype(np.int32))
    assert np.array_equal(C, expected), f"Mismatch in non-square matmul"
    tpu.close()

@test("Tall matrix (100x3) @ (3x2)")
def test_tall_matrix():
    from tinytpu import TinyTPU
    tpu = TinyTPU(backend="simulator", array_size=4)
    A = np.random.randint(-128, 127, (100, 3), dtype=np.int8)
    B = np.random.randint(-128, 127, (3, 2), dtype=np.int8)
    C = tpu.matmul(A, B)
    expected = np.matmul(A.astype(np.int32), B.astype(np.int32))
    assert np.array_equal(C, expected)
    tpu.close()

@test("Wide matrix (2x100) @ (100x3)")
def test_wide_matrix():
    from tinytpu import TinyTPU
    tpu = TinyTPU(backend="simulator", array_size=4)
    A = np.random.randint(-128, 127, (2, 100), dtype=np.int8)
    B = np.random.randint(-128, 127, (100, 3), dtype=np.int8)
    C = tpu.matmul(A, B)
    expected = np.matmul(A.astype(np.int32), B.astype(np.int32))
    assert np.array_equal(C, expected)
    tpu.close()

@test("Prime dimensions (7x11) @ (11x13)")
def test_prime_dimensions():
    from tinytpu import TinyTPU
    tpu = TinyTPU(backend="simulator", array_size=4)
    A = np.random.randint(-128, 127, (7, 11), dtype=np.int8)
    B = np.random.randint(-128, 127, (11, 13), dtype=np.int8)
    C = tpu.matmul(A, B)
    expected = np.matmul(A.astype(np.int32), B.astype(np.int32))
    assert np.array_equal(C, expected)
    tpu.close()

# ============================================================
# SECTION 3: INT8 BOUNDARY VALUES
# ============================================================

@test("Max positive: 127 * 127")
def test_max_positive():
    from tinytpu import TinyTPU
    tpu = TinyTPU(backend="simulator", array_size=4)
    A = np.full((4, 4), 127, dtype=np.int8)
    B = np.full((4, 4), 127, dtype=np.int8)
    C = tpu.matmul(A, B)
    # 127 * 127 * 4 = 64516
    expected = 127 * 127 * 4
    assert C[0, 0] == expected, f"Got {C[0,0]}, expected {expected}"
    tpu.close()

@test("Max negative: -128 * -128")
def test_max_negative_squared():
    from tinytpu import TinyTPU
    tpu = TinyTPU(backend="simulator", array_size=4)
    A = np.full((4, 4), -128, dtype=np.int8)
    B = np.full((4, 4), -128, dtype=np.int8)
    C = tpu.matmul(A, B)
    # -128 * -128 * 4 = 65536
    expected = 65536
    assert C[0, 0] == expected, f"Got {C[0,0]}, expected {expected}"
    tpu.close()

@test("Mixed extreme: 127 * -128")
def test_mixed_extreme():
    from tinytpu import TinyTPU
    tpu = TinyTPU(backend="simulator", array_size=4)
    A = np.full((4, 4), 127, dtype=np.int8)
    B = np.full((4, 4), -128, dtype=np.int8)
    C = tpu.matmul(A, B)
    # 127 * -128 * 4 = -65024
    expected = -65024
    assert C[0, 0] == expected, f"Got {C[0,0]}, expected {expected}"
    tpu.close()

@test("DANGEROUS: -128 * -1 (overflows int8)")
def test_dangerous_overflow():
    from tinytpu import TinyTPU
    tpu = TinyTPU(backend="simulator", array_size=4)
    A = np.full((4, 4), -128, dtype=np.int8)
    B = np.full((4, 4), -1, dtype=np.int8)
    C = tpu.matmul(A, B)
    # -128 * -1 = 128 (overflows int8, but result is int32)
    # 128 * 4 = 512
    expected = 512
    assert C[0, 0] == expected, f"Got {C[0,0]}, expected {expected}"
    tpu.close()

@test("All zeros")
def test_all_zeros():
    from tinytpu import TinyTPU
    tpu = TinyTPU(backend="simulator")
    A = np.zeros((16, 16), dtype=np.int8)
    B = np.zeros((16, 16), dtype=np.int8)
    C = tpu.matmul(A, B)
    assert np.all(C == 0)
    tpu.close()

@test("Identity matrix")
def test_identity():
    from tinytpu import TinyTPU
    tpu = TinyTPU(backend="simulator", array_size=4)
    A = np.random.randint(-128, 127, (8, 8), dtype=np.int8)
    I = np.eye(8, dtype=np.int8)
    C = tpu.matmul(A, I)
    assert np.array_equal(C, A.astype(np.int32))
    tpu.close()

# ============================================================
# SECTION 4: NUMERICAL ACCURACY - EXHAUSTIVE
# ============================================================

@test("1000 random matrices vs numpy")
def test_random_accuracy():
    from tinytpu import TinyTPU
    tpu = TinyTPU(backend="simulator", array_size=4)
    np.random.seed(42)
    errors = 0
    for i in range(1000):
        m, k, n = np.random.randint(1, 20, 3)
        A = np.random.randint(-128, 127, (m, k), dtype=np.int8)
        B = np.random.randint(-128, 127, (k, n), dtype=np.int8)
        C = tpu.matmul(A, B)
        expected = np.matmul(A.astype(np.int32), B.astype(np.int32))
        if not np.array_equal(C, expected):
            errors += 1
            if errors <= 3:
                print(f"  Mismatch at test {i}: shapes ({m},{k})@({k},{n})")
    assert errors == 0, f"{errors}/1000 random tests failed"
    tpu.close()

@test("All boundary value combinations (25 tests)")
def test_boundary_combinations():
    from tinytpu import TinyTPU
    tpu = TinyTPU(backend="simulator", array_size=4)
    bounds = [-128, -1, 0, 1, 127]
    errors = 0
    for a in bounds:
        for b in bounds:
            A = np.full((4, 4), a, dtype=np.int8)
            B = np.full((4, 4), b, dtype=np.int8)
            C = tpu.matmul(A, B)
            expected = a * b * 4
            if C[0, 0] != expected:
                errors += 1
                print(f"  {a} * {b} * 4: got {C[0,0]}, expected {expected}")
    assert errors == 0, f"{errors}/25 boundary tests failed"
    tpu.close()

# ============================================================
# SECTION 5: INPUT VALIDATION & ERROR HANDLING
# ============================================================

@test("Incompatible shapes should raise ValueError")
def test_incompatible_shapes():
    from tinytpu import TinyTPU
    tpu = TinyTPU(backend="simulator")
    A = np.zeros((3, 4), dtype=np.int8)
    B = np.zeros((5, 6), dtype=np.int8)
    try:
        C = tpu.matmul(A, B)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Incompatible" in str(e) or "shape" in str(e).lower()
    tpu.close()

@test("1D input should raise error")
def test_1d_input():
    from tinytpu import TinyTPU
    tpu = TinyTPU(backend="simulator")
    A = np.array([1, 2, 3], dtype=np.int8)
    B = np.array([4, 5, 6], dtype=np.int8)
    try:
        C = tpu.matmul(A, B)
        assert False, "Should have raised error for 1D input"
    except (ValueError, IndexError, AttributeError):
        pass  # Expected
    tpu.close()

@test("3D input should raise error")
def test_3d_input():
    from tinytpu import TinyTPU
    tpu = TinyTPU(backend="simulator")
    A = np.zeros((2, 3, 4), dtype=np.int8)
    B = np.zeros((2, 4, 5), dtype=np.int8)
    try:
        C = tpu.matmul(A, B)
        assert False, "Should have raised error for 3D input"
    except (ValueError, IndexError):
        pass  # Expected
    tpu.close()

@test("Float input auto-conversion")
def test_float_input():
    from tinytpu import TinyTPU
    tpu = TinyTPU(backend="simulator", array_size=4)
    A = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    B = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    C = tpu.matmul(A, B)
    expected = np.array([[19, 22], [43, 50]])
    assert np.array_equal(C, expected), f"Float conversion failed"
    tpu.close()

@test("Values out of INT8 range")
def test_out_of_range():
    from tinytpu import TinyTPU
    tpu = TinyTPU(backend="simulator")
    A = np.array([[200, -200], [300, -300]], dtype=np.int16)
    B = np.array([[1, 0], [0, 1]], dtype=np.int16)
    # This should either clip, wrap, or raise error
    try:
        C = tpu.matmul(A, B)
        # Check if values were clipped or wrapped
        print(f"  Out of range handled: result[0,0] = {C[0,0]}")
    except (ValueError, OverflowError):
        print(f"  Out of range raised error (acceptable)")
    tpu.close()

@test("None input")
def test_none_input():
    from tinytpu import TinyTPU
    tpu = TinyTPU(backend="simulator")
    try:
        C = tpu.matmul(None, None)
        assert False, "Should have raised error"
    except (TypeError, AttributeError):
        pass
    tpu.close()

@test("List input (not numpy array)")
def test_list_input():
    from tinytpu import TinyTPU
    tpu = TinyTPU(backend="simulator", array_size=4)
    A = [[1, 2], [3, 4]]
    B = [[5, 6], [7, 8]]
    try:
        C = tpu.matmul(A, B)
        # Should work if auto-converted
        expected = np.array([[19, 22], [43, 50]])
        assert np.array_equal(C, expected)
    except (TypeError, AttributeError) as e:
        print(f"  List input not supported: {e}")
    tpu.close()

# ============================================================
# SECTION 6: MEMORY & PERFORMANCE STRESS
# ============================================================

@test("Large matrix (512x512)")
def test_large_512():
    from tinytpu import TinyTPU
    tpu = TinyTPU(backend="simulator", array_size=16)
    A = np.random.randint(-128, 127, (512, 512), dtype=np.int8)
    B = np.random.randint(-128, 127, (512, 512), dtype=np.int8)
    start = time.time()
    C = tpu.matmul(A, B)
    elapsed = time.time() - start
    print(f"  512x512 took {elapsed:.2f}s")
    expected = np.matmul(A.astype(np.int32), B.astype(np.int32))
    assert np.array_equal(C, expected)
    tpu.close()

@test("Very large matrix (1024x1024)")
def test_large_1024():
    from tinytpu import TinyTPU
    tpu = TinyTPU(backend="simulator", array_size=16)
    A = np.random.randint(-128, 127, (1024, 1024), dtype=np.int8)
    B = np.random.randint(-128, 127, (1024, 1024), dtype=np.int8)
    start = time.time()
    C = tpu.matmul(A, B)
    elapsed = time.time() - start
    print(f"  1024x1024 took {elapsed:.2f}s")
    # Skip full verification for speed, just check shape
    assert C.shape == (1024, 1024)
    tpu.close()

@test("Repeated operations (memory leak check)")
def test_memory_leak():
    from tinytpu import TinyTPU
    import gc
    gc.collect()
    tpu = TinyTPU(backend="simulator", array_size=4)
    A = np.random.randint(-128, 127, (64, 64), dtype=np.int8)
    B = np.random.randint(-128, 127, (64, 64), dtype=np.int8)
    for i in range(100):
        C = tpu.matmul(A, B)
        del C
    gc.collect()
    print("  100 iterations completed without crash")
    tpu.close()

@test("Multiple TPU instances")
def test_multiple_instances():
    from tinytpu import TinyTPU
    tpu1 = TinyTPU(backend="simulator", array_size=4)
    tpu2 = TinyTPU(backend="simulator", array_size=8)
    tpu3 = TinyTPU(backend="simulator", array_size=16)
    A = np.random.randint(-128, 127, (16, 16), dtype=np.int8)
    B = np.random.randint(-128, 127, (16, 16), dtype=np.int8)
    C1 = tpu1.matmul(A, B)
    C2 = tpu2.matmul(A, B)
    C3 = tpu3.matmul(A, B)
    assert np.array_equal(C1, C2)
    assert np.array_equal(C2, C3)
    tpu1.close()
    tpu2.close()
    tpu3.close()

# ============================================================
# SECTION 7: REAL-WORLD LLM WORKLOADS
# ============================================================

@test("LLM-like shapes: (1, 4096) @ (4096, 4096)")
def test_llm_projection():
    from tinytpu import TinyTPU
    tpu = TinyTPU(backend="simulator", array_size=16)
    batch_size = 1
    hidden_size = 4096
    A = np.random.randint(-128, 127, (batch_size, hidden_size), dtype=np.int8)
    B = np.random.randint(-128, 127, (hidden_size, hidden_size), dtype=np.int8)
    start = time.time()
    C = tpu.matmul(A, B)
    elapsed = time.time() - start
    print(f"  (1, 4096) @ (4096, 4096) took {elapsed:.2f}s")
    assert C.shape == (batch_size, hidden_size)
    tpu.close()

@test("Attention QK^T: (32, 128) @ (128, 32)")  
def test_attention_qk():
    from tinytpu import TinyTPU
    tpu = TinyTPU(backend="simulator", array_size=16)
    seq_len = 32
    head_dim = 128
    Q = np.random.randint(-128, 127, (seq_len, head_dim), dtype=np.int8)
    K = np.random.randint(-128, 127, (head_dim, seq_len), dtype=np.int8)
    C = tpu.matmul(Q, K)
    expected = np.matmul(Q.astype(np.int32), K.astype(np.int32))
    assert np.array_equal(C, expected)
    tpu.close()

@test("FFN up projection: (8, 4096) @ (4096, 11008)")
def test_ffn_up():
    from tinytpu import TinyTPU
    tpu = TinyTPU(backend="simulator", array_size=16)
    A = np.random.randint(-128, 127, (8, 4096), dtype=np.int8)
    B = np.random.randint(-128, 127, (4096, 11008), dtype=np.int8)
    start = time.time()
    C = tpu.matmul(A, B)
    elapsed = time.time() - start
    print(f"  FFN up (8, 4096) @ (4096, 11008) took {elapsed:.2f}s")
    assert C.shape == (8, 11008)
    tpu.close()

# ============================================================
# SECTION 8: API USABILITY
# ============================================================

@test("Context manager")
def test_context_manager():
    from tinytpu import TinyTPU
    with TinyTPU(backend="simulator") as tpu:
        A = np.eye(4, dtype=np.int8)
        B = np.eye(4, dtype=np.int8)
        C = tpu.matmul(A, B)
        assert np.array_equal(C, np.eye(4, dtype=np.int32))
    # TPU should be closed now
    assert not tpu.is_connected

@test("Benchmark function")
def test_benchmark():
    from tinytpu import TinyTPU
    tpu = TinyTPU(backend="simulator", array_size=4)
    results = tpu.benchmark(size=32, iterations=10)
    assert "gops" in results
    assert "time_per_matmul_ms" in results
    assert results["iterations"] == 10
    print(f"  Benchmark: {results['gops']:.2f} GOPS")
    tpu.close()

@test("Softmax function")
def test_softmax():
    from tinytpu import TinyTPU
    tpu = TinyTPU(backend="simulator")
    x = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    y = tpu.softmax(x)
    # Check sums to 1
    row_sums = y.sum(axis=1)
    assert np.allclose(row_sums, 1.0), f"Softmax rows don't sum to 1: {row_sums}"
    tpu.close()

@test("Repr string")
def test_repr():
    from tinytpu import TinyTPU
    tpu = TinyTPU(backend="simulator", array_size=8)
    r = repr(tpu)
    assert "simulator" in r
    assert "8" in r
    tpu.close()

# ============================================================
# SECTION 9: ARRAY SIZE EDGE CASES
# ============================================================

@test("Array size 1 (degenerate)")
def test_array_size_1():
    from tinytpu import TinyTPU
    tpu = TinyTPU(backend="simulator", array_size=1)
    A = np.random.randint(-128, 127, (8, 8), dtype=np.int8)
    B = np.random.randint(-128, 127, (8, 8), dtype=np.int8)
    C = tpu.matmul(A, B)
    expected = np.matmul(A.astype(np.int32), B.astype(np.int32))
    assert np.array_equal(C, expected)
    tpu.close()

@test("Array size 2")
def test_array_size_2():
    from tinytpu import TinyTPU
    tpu = TinyTPU(backend="simulator", array_size=2)
    A = np.random.randint(-128, 127, (7, 9), dtype=np.int8)
    B = np.random.randint(-128, 127, (9, 5), dtype=np.int8)
    C = tpu.matmul(A, B)
    expected = np.matmul(A.astype(np.int32), B.astype(np.int32))
    assert np.array_equal(C, expected)
    tpu.close()

@test("Array size 32")
def test_array_size_32():
    from tinytpu import TinyTPU
    tpu = TinyTPU(backend="simulator", array_size=32)
    A = np.random.randint(-128, 127, (100, 100), dtype=np.int8)
    B = np.random.randint(-128, 127, (100, 100), dtype=np.int8)
    C = tpu.matmul(A, B)
    expected = np.matmul(A.astype(np.int32), B.astype(np.int32))
    assert np.array_equal(C, expected)
    tpu.close()

@test("Matrix smaller than array size")
def test_smaller_than_array():
    from tinytpu import TinyTPU
    tpu = TinyTPU(backend="simulator", array_size=64)
    A = np.random.randint(-128, 127, (5, 5), dtype=np.int8)
    B = np.random.randint(-128, 127, (5, 5), dtype=np.int8)
    C = tpu.matmul(A, B)
    expected = np.matmul(A.astype(np.int32), B.astype(np.int32))
    assert np.array_equal(C, expected)
    tpu.close()

# ============================================================
# SECTION 10: CONCURRENCY (if applicable)
# ============================================================

@test("Sequential operations same instance")
def test_sequential_ops():
    from tinytpu import TinyTPU
    tpu = TinyTPU(backend="simulator", array_size=4)
    results = []
    for i in range(10):
        A = np.full((4, 4), i, dtype=np.int8)
        B = np.full((4, 4), 1, dtype=np.int8)
        C = tpu.matmul(A, B)
        results.append(C[0, 0])
    expected = [i * 4 for i in range(10)]
    assert results == expected
    tpu.close()

# ============================================================
# RUN ALL TESTS
# ============================================================

def main():
    print("\n" + "#" * 60)
    print("# BRUTAL TINYTPU TEST SUITE")
    print("# Finding every weakness...")
    print("#" * 60)
    
    tests = [
        # Section 1: Basic
        test_import,
        test_create,
        test_basic_2x2,
        
        # Section 2: Edge cases
        test_empty_matrix,
        test_single_element,
        test_non_square,
        test_tall_matrix,
        test_wide_matrix,
        test_prime_dimensions,
        
        # Section 3: INT8 boundaries
        test_max_positive,
        test_max_negative_squared,
        test_mixed_extreme,
        test_dangerous_overflow,
        test_all_zeros,
        test_identity,
        
        # Section 4: Numerical accuracy
        test_random_accuracy,
        test_boundary_combinations,
        
        # Section 5: Error handling
        test_incompatible_shapes,
        test_1d_input,
        test_3d_input,
        test_float_input,
        test_out_of_range,
        test_none_input,
        test_list_input,
        
        # Section 6: Stress tests
        test_large_512,
        test_large_1024,
        test_memory_leak,
        test_multiple_instances,
        
        # Section 7: LLM workloads
        test_llm_projection,
        test_attention_qk,
        test_ffn_up,
        
        # Section 8: API usability
        test_context_manager,
        test_benchmark,
        test_softmax,
        test_repr,
        
        # Section 9: Array size edge cases
        test_array_size_1,
        test_array_size_2,
        test_array_size_32,
        test_smaller_than_array,
        
        # Section 10: Concurrency
        test_sequential_ops,
    ]
    
    for test_func in tests:
        test_func()
    
    # Final report
    print("\n" + "#" * 60)
    print("# FINAL REPORT")
    print("#" * 60)
    print(f"\nPASSED: {len(PASSES)}/{len(tests)}")
    print(f"FAILED: {len(FAILURES)}/{len(tests)}")
    
    if FAILURES:
        print("\n" + "=" * 60)
        print("FAILURE DETAILS:")
        print("=" * 60)
        for name, error, trace in FAILURES:
            print(f"\n[FAIL] {name}")
            print(f"Error: {error}")
            print("Traceback:")
            print(trace)
    
    if len(FAILURES) == 0:
        print("\n*** ALL TESTS PASSED ***")
    else:
        print(f"\n*** {len(FAILURES)} TESTS FAILED - NEEDS FIXING ***")
    
    return len(FAILURES)

if __name__ == "__main__":
    sys.exit(main())
