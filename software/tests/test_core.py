import numpy as np
import pytest
from tinytpu import TinyTPU

def test_basic_matmul():
    tpu = TinyTPU(backend="simulator", array_size=4)
    A = np.array([[1, 2], [3, 4]], dtype=np.int8)
    B = np.array([[5, 6], [7, 8]], dtype=np.int8)
    C = tpu.matmul(A, B)
    expected = np.array([[19, 22], [43, 50]], dtype=np.int32)
    np.testing.assert_array_equal(C, expected)
    tpu.close()

def test_max_values():
    tpu = TinyTPU(backend="simulator", array_size=4)
    A = np.full((4, 4), -128, dtype=np.int8)
    B = np.full((4, 4), -128, dtype=np.int8)
    C = tpu.matmul(A, B)
    expected = np.full((4, 4), 65536, dtype=np.int32)
    np.testing.assert_array_equal(C, expected)
    tpu.close()

def test_large_matmul():
    tpu = TinyTPU(backend="simulator", array_size=4)
    np.random.seed(42)
    A = np.random.randint(-128, 127, (16, 16), dtype=np.int8)
    B = np.random.randint(-128, 127, (16, 16), dtype=np.int8)
    C = tpu.matmul(A, B)
    expected = np.matmul(A.astype(np.int32), B.astype(np.int32))
    np.testing.assert_array_equal(C, expected)
    tpu.close()

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
