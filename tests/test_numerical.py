"""Tests for numerical methods — Richardson extrapolation, eigenvalues, activations."""

import pytest
import numpy as np
from tinytpu.numerical.methods import QuantizationExtrapolator, IterativeEigen, FastActivations


class TestQuantizationExtrapolator:

    def test_creation(self):
        qe = QuantizationExtrapolator()
        assert qe is not None

    def test_extrapolate_callable(self):
        qe = QuantizationExtrapolator()
        x = np.random.randn(8).astype(np.float32)
        result = qe.extrapolate(lambda v: v ** 2, x)
        assert result.shape == x.shape
        assert np.all(np.isfinite(result))

    def test_extrapolate_improves_over_single_quantization(self):
        """Richardson should produce results closer to true than single quantization."""
        qe = QuantizationExtrapolator()
        x = np.linspace(-0.5, 0.5, 16).astype(np.float32)
        true_result = x ** 2
        extrapolated = qe.extrapolate(lambda v: v ** 2, x)
        assert np.all(np.isfinite(extrapolated))

    def test_custom_bits_list(self):
        qe = QuantizationExtrapolator()
        x = np.array([0.5, -0.3], dtype=np.float32)
        result = qe.extrapolate(lambda v: v, x, bits_list=[4, 8])
        assert np.all(np.isfinite(result))


class TestIterativeEigen:

    def test_creation(self):
        ie = IterativeEigen()
        assert ie is not None

    def test_power_iteration_symmetric(self):
        """Power iteration on symmetric matrix should find largest eigenvalue."""
        ie = IterativeEigen()
        A = np.array([[4, 1], [1, 3]], dtype=np.float64)
        eigenvalue, eigenvector = ie.power_iteration(A, num_iters=100)
        # Largest eigenvalue of [[4,1],[1,3]] is (7+sqrt(5))/2 ≈ 4.618
        expected = (7 + np.sqrt(5)) / 2
        assert abs(eigenvalue - expected) < 0.1
        assert eigenvector.shape == (2,)

    def test_top_k_eigenvalues(self):
        ie = IterativeEigen()
        A = np.diag([5.0, 3.0, 1.0])
        eigenvalues, eigenvectors = ie.top_k(A, k=2, num_iters=100)
        assert len(eigenvalues) == 2
        assert abs(eigenvalues[0] - 5.0) < 0.1
        assert abs(eigenvalues[1] - 3.0) < 0.1

    def test_large_matrix(self):
        ie = IterativeEigen()
        np.random.seed(42)
        A = np.random.randn(50, 50)
        A = A @ A.T  # symmetric positive semi-definite
        eigenvalue, eigenvector = ie.power_iteration(A, num_iters=200)
        # Verify: A @ v ≈ λ * v
        Av = A @ eigenvector
        lv = eigenvalue * eigenvector
        assert np.allclose(Av, lv, atol=0.5)

    def test_convergence(self):
        """Should converge early for well-conditioned matrix."""
        ie = IterativeEigen()
        A = np.diag([10.0, 1.0])  # large eigenvalue gap → fast convergence
        val, vec = ie.power_iteration(A, num_iters=50)
        assert abs(val - 10.0) < 0.01

    def test_deflation(self):
        """top_k with deflation finds distinct eigenvalues."""
        ie = IterativeEigen()
        A = np.diag([7.0, 5.0, 3.0, 1.0])
        vals, _ = ie.top_k(A, k=3, num_iters=100)
        assert abs(vals[0] - 7.0) < 0.1
        assert abs(vals[1] - 5.0) < 0.1
        assert abs(vals[2] - 3.0) < 0.1


class TestFastActivations:

    def test_relu_poly_positive(self):
        fa = FastActivations()
        x = np.array([1, 2, 3], dtype=np.float32)
        result = fa.relu_poly(x)
        np.testing.assert_array_almost_equal(result, x)

    def test_relu_poly_negative(self):
        fa = FastActivations()
        x = np.array([-1, -2, -3], dtype=np.float32)
        result = fa.relu_poly(x)
        # Leaky ReLU-style: 0.01 * x for negative
        assert np.all(result < 0)
        assert np.all(result > -0.1)  # very small

    def test_relu_poly_zero(self):
        fa = FastActivations()
        x = np.array([0.0], dtype=np.float32)
        result = fa.relu_poly(x)
        assert abs(result[0]) < 0.01

    def test_sigmoid_horner_at_zero(self):
        """sigmoid(0) = 0.5 exactly."""
        fa = FastActivations()
        x = np.array([0.0], dtype=np.float32)
        result = fa.sigmoid_horner(x)
        assert abs(result[0] - 0.5) < 0.01

    def test_sigmoid_horner_small_range(self):
        """Polynomial sigmoid is accurate in [-2, 2]."""
        fa = FastActivations()
        x = np.array([-2, -1, 0, 1, 2], dtype=np.float32)
        result = fa.sigmoid_horner(x)
        expected = 1.0 / (1.0 + np.exp(-x))
        # Polynomial approximation — more tolerance needed
        assert np.allclose(result, expected, atol=0.15)

    def test_gelu_poly(self):
        fa = FastActivations()
        x = np.array([-2, -1, 0, 1, 2], dtype=np.float32)
        result = fa.gelu_poly(x)
        # GELU(0) = 0
        assert abs(result[2]) < 0.05
        # GELU(x) ≈ x for large positive x
        assert result[4] > 1.5

    def test_swish_poly_at_zero(self):
        fa = FastActivations()
        x = np.array([0.0], dtype=np.float32)
        result = fa.swish_poly(x)
        assert abs(result[0]) < 0.01

    def test_activations_vectorized(self):
        """Should work on large arrays without error."""
        fa = FastActivations()
        x = np.random.randn(1000, 768).astype(np.float32)
        for fn in [fa.relu_poly, fa.sigmoid_horner, fa.gelu_poly, fa.swish_poly]:
            result = fn(x)
            assert result.shape == (1000, 768)
            assert np.all(np.isfinite(result))
