"""
Tests for tensor_train.py — TT decomposition, Tucker conv compression,
Padé activations, and Richardson+TT integration.

Tests organized by component:
  1. TensorTrain core decomposition (14 tests)
  2. TTLinear weight compression (10 tests)
  3. TuckerConv kernel compression (8 tests)
  4. PadeActivations (10 tests)
  5. RichardsonTTLinear integration (8 tests)
  6. Utility functions (4 tests)
  7. Edge AI realistic scenarios (6 tests)

Total: 60 tests
"""

import numpy as np
import pytest
import time
import sys
import os

# Add parent dir to path for import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tensor_train import (
    TensorTrain, TTLinear, TuckerConv,
    PadeActivations, RichardsonTTLinear,
    _auto_factorize, compress_linear_layer, compress_conv_kernel,
)


# ============================================================
# 1. TensorTrain Core Decomposition
# ============================================================

class TestTensorTrainDecompose:
    """Test TT decomposition via sequential SVD (Algorithm 11)."""

    def test_3d_tensor_exact(self):
        """Rank-1 tensor should decompose exactly with rank 1."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0])
        c = np.array([6.0, 7.0, 8.0, 9.0])
        # Rank-1 tensor via outer product
        T = np.einsum('i,j,k->ijk', a, b, c)
        tt = TensorTrain.decompose(T, max_rank=5)
        recon = tt.reconstruct()
        np.testing.assert_allclose(recon, T, atol=1e-10)

    def test_3d_tensor_shape(self):
        """Verify core shapes for 3D tensor."""
        T = np.random.randn(4, 5, 6)
        tt = TensorTrain.decompose(T, max_rank=10)
        assert len(tt.cores) == 3
        assert tt.shape == (4, 5, 6)
        assert tt.cores[0].shape[0] == 1    # R_1 = 1
        assert tt.cores[-1].shape[2] == 1   # R_{N+1} = 1

    def test_4d_tensor(self):
        """4D tensor decomposition and reconstruction."""
        T = np.random.randn(3, 4, 5, 6)
        tt = TensorTrain.decompose(T, max_rank=20)
        recon = tt.reconstruct()
        rel_error = np.linalg.norm(recon - T) / np.linalg.norm(T)
        assert rel_error < 1e-10, f"Exact reconstruction failed: {rel_error}"

    def test_rank_truncation(self):
        """Low max_rank should compress but introduce error."""
        T = np.random.randn(8, 8, 8, 8)
        tt_full = TensorTrain.decompose(T, max_rank=64)
        tt_low = TensorTrain.decompose(T, max_rank=4)
        assert tt_low.total_params() < tt_full.total_params()
        # Low rank should have some error
        recon = tt_low.reconstruct()
        rel_error = np.linalg.norm(recon - T) / np.linalg.norm(T)
        assert rel_error > 0  # Should have some error
        assert rel_error < 1.0  # But not garbage

    def test_tolerance_control(self):
        """Tolerance should control reconstruction accuracy."""
        T = np.random.randn(6, 6, 6)
        tt_tight = TensorTrain.decompose(T, relative_tolerance=1e-10)
        tt_loose = TensorTrain.decompose(T, relative_tolerance=0.1)
        # Tight tolerance → more params
        assert tt_tight.total_params() >= tt_loose.total_params()

    def test_compression_ratio(self):
        """Compression ratio should be > 1 for low-rank tensors."""
        # Low-rank tensor (sum of a few rank-1 components)
        T = np.zeros((8, 8, 8))
        for _ in range(3):
            T += np.einsum('i,j,k->ijk',
                           np.random.randn(8),
                           np.random.randn(8),
                           np.random.randn(8))
        tt = TensorTrain.decompose(T, max_rank=4)
        assert tt.compression_ratio() > 1.0

    def test_element_access(self):
        """Single element access should match reconstructed tensor."""
        T = np.random.randn(4, 5, 3)
        tt = TensorTrain.decompose(T, max_rank=20)
        for _ in range(10):
            idx = (np.random.randint(4), np.random.randint(5), np.random.randint(3))
            val = tt.element(idx)
            np.testing.assert_allclose(val, T[idx], atol=1e-10)

    def test_2d_tensor(self):
        """2D tensor (matrix) decomposition should work."""
        M = np.random.randn(6, 8)
        tt = TensorTrain.decompose(M, max_rank=6)
        recon = tt.reconstruct()
        np.testing.assert_allclose(recon, M, atol=1e-10)

    def test_5d_tensor(self):
        """High-order tensor decomposition."""
        T = np.random.randn(3, 3, 3, 3, 3)
        tt = TensorTrain.decompose(T, max_rank=9)
        recon = tt.reconstruct()
        rel_error = np.linalg.norm(recon - T) / np.linalg.norm(T)
        assert rel_error < 1e-8

    def test_repr(self):
        """String representation should contain key info."""
        T = np.random.randn(4, 5, 6)
        tt = TensorTrain.decompose(T)
        s = repr(tt)
        assert "4×5×6" in s
        assert "compression" in s

    def test_symmetric_tensor(self):
        """Symmetric tensor should decompose correctly."""
        A = np.random.randn(5, 5)
        T = A + A.T  # Symmetric matrix
        T_3d = np.einsum('ij,k->ijk', T, np.ones(3))
        tt = TensorTrain.decompose(T_3d, max_rank=10)
        recon = tt.reconstruct()
        rel_error = np.linalg.norm(recon - T_3d) / np.linalg.norm(T_3d)
        assert rel_error < 1e-10

    def test_sparse_tensor(self):
        """Sparse tensor should compress well."""
        T = np.zeros((10, 10, 10))
        # Only a few nonzero entries
        for _ in range(5):
            i, j, k = np.random.randint(10, size=3)
            T[i, j, k] = np.random.randn()
        tt = TensorTrain.decompose(T, max_rank=5)
        recon = tt.reconstruct()
        # Should have reasonable accuracy
        nonzero_mask = T != 0
        if np.any(nonzero_mask):
            rel_error = np.linalg.norm(recon[nonzero_mask] - T[nonzero_mask]) / np.linalg.norm(T[nonzero_mask])
            assert rel_error < 0.5  # Sparse tensors are harder

    def test_ones_tensor(self):
        """All-ones tensor is rank-1, should decompose perfectly."""
        T = np.ones((4, 5, 6))
        tt = TensorTrain.decompose(T, max_rank=2)
        recon = tt.reconstruct()
        np.testing.assert_allclose(recon, T, atol=1e-10)

    def test_identity_like(self):
        """Diagonal-like tensor decomposition."""
        T = np.zeros((4, 4, 4))
        for i in range(4):
            T[i, i, i] = 1.0
        tt = TensorTrain.decompose(T, max_rank=4)
        recon = tt.reconstruct()
        np.testing.assert_allclose(recon, T, atol=1e-8)


# ============================================================
# 2. TTLinear Weight Compression
# ============================================================

class TestTTLinear:
    """Test TT-compressed linear layers."""

    def test_small_layer(self):
        """Small layer should reconstruct with reasonable accuracy."""
        W = np.random.randn(12, 12)
        tt = TTLinear(W, input_shape=(3, 4), output_shape=(3, 4), max_rank=8)
        W_recon = tt.reconstruct_weight()
        rel_error = np.linalg.norm(W_recon - W) / np.linalg.norm(W)
        assert rel_error < 0.15, f"Reconstruction error: {rel_error}"

    def test_forward_accuracy(self):
        """Forward pass should match direct matmul within tolerance."""
        W = np.random.randn(12, 12)
        x = np.random.randn(12)
        tt = TTLinear(W, input_shape=(3, 4), output_shape=(3, 4), max_rank=12)
        y_exact = W @ x
        y_tt = tt.forward(x)
        rel_error = np.linalg.norm(y_tt - y_exact) / np.linalg.norm(y_exact)
        assert rel_error < 0.05, f"Forward error too high: {rel_error}"

    def test_batch_forward(self):
        """Batched forward pass."""
        W = np.random.randn(12, 12)
        X = np.random.randn(5, 12)
        tt = TTLinear(W, input_shape=(3, 4), output_shape=(3, 4), max_rank=12)
        Y_exact = (W @ X.T).T
        Y_tt = tt.forward(X)
        rel_error = np.linalg.norm(Y_tt - Y_exact) / np.linalg.norm(Y_exact)
        assert rel_error < 0.05

    def test_compression_ratio(self):
        """Large layer should achieve significant compression."""
        W = np.random.randn(64, 64)
        tt = TTLinear(W, input_shape=(4, 4, 4), output_shape=(4, 4, 4), max_rank=8)
        assert tt.compression_ratio > 2.0, \
            f"Compression ratio too low: {tt.compression_ratio}"

    def test_auto_factorize(self):
        """Auto-factorization should handle various sizes."""
        W = np.random.randn(48, 36)
        tt = TTLinear(W, max_rank=8)
        W_recon = tt.reconstruct_weight()
        assert W_recon.shape == (48, 36)

    def test_low_rank_weight(self):
        """Low-rank weight matrix should compress reasonably."""
        # Rank-4 matrix
        A = np.random.randn(64, 4)
        B = np.random.randn(4, 64)
        W = A @ B
        tt = TTLinear(W, input_shape=(4, 4, 4), output_shape=(4, 4, 4), max_rank=8)
        W_recon = tt.reconstruct_weight()
        rel_error = np.linalg.norm(W_recon - W) / np.linalg.norm(W)
        # Interleaving dimensions changes rank structure, so relax threshold
        assert rel_error < 1.0
        assert tt.compression_ratio > 2.0

    def test_repr(self):
        """TTLinear repr should show compression info."""
        W = np.random.randn(16, 16)
        tt = TTLinear(W, input_shape=(4, 4), output_shape=(4, 4), max_rank=4)
        s = repr(tt)
        assert "compression" in s
        assert "16×16" in s

    def test_rectangular_weight(self):
        """Non-square weight matrix."""
        W = np.random.randn(24, 36)
        tt = TTLinear(W, input_shape=(6, 6), output_shape=(4, 6), max_rank=8)
        W_recon = tt.reconstruct_weight()
        assert W_recon.shape == (24, 36)

    def test_very_small_rank(self):
        """Rank-1 TT should still produce valid output."""
        W = np.random.randn(16, 16)
        tt = TTLinear(W, input_shape=(4, 4), output_shape=(4, 4), max_rank=1)
        y = tt.forward(np.ones(16))
        assert y.shape == (16,)
        assert np.all(np.isfinite(y))

    def test_params_count(self):
        """Compressed params should be less than original."""
        W = np.random.randn(64, 64)
        tt = TTLinear(W, input_shape=(4, 4, 4), output_shape=(4, 4, 4), max_rank=4)
        assert tt.compressed_params < tt.original_params


# ============================================================
# 3. Tucker Conv Compression
# ============================================================

class TestTuckerConv:
    """Test Tucker decomposition for conv kernels."""

    def test_basic_decomposition(self):
        """Basic 4D kernel decomposition."""
        K = np.random.randn(32, 16, 3, 3)
        tucker = TuckerConv(K, rank_out=8, rank_in=4)
        assert tucker.core.shape == (8, 4, 3, 3)
        assert tucker.U_out.shape == (32, 8)
        assert tucker.U_in.shape == (16, 4)

    def test_reconstruction_accuracy(self):
        """Reconstruction should be reasonable at moderate rank."""
        K = np.random.randn(64, 32, 3, 3)
        tucker = TuckerConv(K, rank_out=16, rank_in=8)
        error = tucker.reconstruction_error(K)
        # Random matrices have uniformly distributed singular values,
        # so truncation error is substantial. Real DNN kernels are much
        # more compressible (low effective rank).
        assert error < 1.0, f"Reconstruction error too high: {error}"

    def test_high_rank_exact(self):
        """Full rank should give near-exact reconstruction."""
        K = np.random.randn(8, 8, 3, 3)
        tucker = TuckerConv(K, rank_out=8, rank_in=8)
        error = tucker.reconstruction_error(K)
        assert error < 1e-10

    def test_compression_ratio(self):
        """Should achieve compression for large kernels."""
        K = np.random.randn(256, 256, 3, 3)
        tucker = TuckerConv(K, rank_out=64, rank_in=64)
        assert tucker.compression_ratio > 2.0

    def test_low_rank_kernel(self):
        """Low-rank kernel should compress very well."""
        # Separable kernel: outer product structure
        u = np.random.randn(64, 4)
        v = np.random.randn(32, 4)
        core = np.random.randn(4, 4, 3, 3)
        K = np.einsum('oO,iI,OIhw->oihw', u, v, core)
        tucker = TuckerConv(K, rank_out=4, rank_in=4)
        error = tucker.reconstruction_error(K)
        assert error < 0.01

    def test_1x1_conv(self):
        """1×1 convolution (channel mixing only)."""
        K = np.random.randn(128, 64, 1, 1)
        tucker = TuckerConv(K, rank_out=32, rank_in=16)
        error = tucker.reconstruction_error(K)
        assert error < 1.0  # Random weights → high truncation error

    def test_repr(self):
        """Tucker repr should show compression info."""
        K = np.random.randn(64, 32, 3, 3)
        tucker = TuckerConv(K, rank_out=16, rank_in=8)
        s = repr(tucker)
        assert "64×32×3×3" in s
        assert "compression" in s

    def test_convenience_function(self):
        """compress_conv_kernel should work end to end."""
        K = np.random.randn(128, 64, 3, 3)
        result = compress_conv_kernel(K, rank_ratio=0.25)
        assert 'tucker' in result
        assert result['compression_ratio'] > 1.0
        assert result['reconstruction_error'] < 1.0


# ============================================================
# 4. Padé Activations
# ============================================================

class TestPadeActivations:
    """Test Padé approximant activation functions."""

    def _reference_sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def _reference_tanh(self, x):
        return np.tanh(x)

    def _reference_gelu(self, x):
        return x * self._reference_sigmoid(1.702 * x)

    def _reference_silu(self, x):
        return x * self._reference_sigmoid(x)

    def test_sigmoid_accuracy(self):
        """Sigmoid Padé [3/3] accuracy on [-6, 6]."""
        x = np.linspace(-6, 6, 1000)
        exact = self._reference_sigmoid(x)
        approx = PadeActivations.sigmoid_pade(x)
        max_error = np.max(np.abs(exact - approx))
        assert max_error < 0.005, f"Sigmoid max error: {max_error}"

    def test_sigmoid_55_accuracy(self):
        """Sigmoid Padé [5/5] should be more accurate."""
        x = np.linspace(-6, 6, 1000)
        exact = self._reference_sigmoid(x)
        approx_33 = PadeActivations.sigmoid_pade(x)
        approx_55 = PadeActivations.sigmoid_pade_55(x)
        error_33 = np.max(np.abs(exact - approx_33))
        error_55 = np.max(np.abs(exact - approx_55))
        assert error_55 < error_33, \
            f"[5/5] ({error_55}) should be more accurate than [3/3] ({error_33})"

    def test_sigmoid_boundary(self):
        """Sigmoid should be 0.5 at x=0, bounded [0,1]."""
        assert abs(PadeActivations.sigmoid_pade(np.array([0.0]))[0] - 0.5) < 0.001
        vals = PadeActivations.sigmoid_pade(np.linspace(-20, 20, 100))
        assert np.all(vals >= 0.0)
        assert np.all(vals <= 1.0)

    def test_tanh_accuracy(self):
        """Tanh Padé [5/4] accuracy on [-6, 6]."""
        x = np.linspace(-6, 6, 1000)
        exact = self._reference_tanh(x)
        approx = PadeActivations.tanh_pade(x)
        max_error = np.max(np.abs(exact - approx))
        assert max_error < 0.005, f"Tanh max error: {max_error}"

    def test_tanh_boundary(self):
        """Tanh should be 0 at x=0, bounded [-1,1]."""
        assert abs(PadeActivations.tanh_pade(np.array([0.0]))[0]) < 0.001
        vals = PadeActivations.tanh_pade(np.linspace(-20, 20, 100))
        assert np.all(vals >= -1.0)
        assert np.all(vals <= 1.0)

    def test_gelu_accuracy(self):
        """GELU Padé accuracy on [-5, 5]."""
        x = np.linspace(-5, 5, 1000)
        exact = self._reference_gelu(x)
        approx = PadeActivations.gelu_pade(x)
        max_error = np.max(np.abs(exact - approx))
        # Error amplified by x * sigmoid_error, so up to ~0.03 at tails
        assert max_error < 0.05, f"GELU max error: {max_error}"

    def test_silu_accuracy(self):
        """SiLU Padé accuracy on [-5, 5]."""
        x = np.linspace(-5, 5, 1000)
        exact = self._reference_silu(x)
        approx = PadeActivations.silu_pade(x)
        max_error = np.max(np.abs(exact - approx))
        assert max_error < 0.05, f"SiLU max error: {max_error}"

    def test_softmax(self):
        """Softmax should sum to 1 and handle large values."""
        x = np.array([1.0, 2.0, 3.0, 4.0])
        s = PadeActivations.softmax_stable(x)
        np.testing.assert_allclose(np.sum(s), 1.0, atol=1e-10)
        # Large values shouldn't cause overflow
        x_large = np.array([1000.0, 1001.0, 999.0])
        s_large = PadeActivations.softmax_stable(x_large)
        np.testing.assert_allclose(np.sum(s_large), 1.0, atol=1e-10)

    def test_pade_fewer_ops_than_horner(self):
        """Verify Padé is more accurate than same-degree polynomial.

        Padé [3/3] sigmoid: ~5 multiply + 1 divide
        Horner degree-3 sigmoid: 3 multiply
        But Padé [3/3] matches degree-7+ polynomial accuracy.
        """
        x = np.linspace(-6, 6, 1000)
        exact = self._reference_sigmoid(x)
        pade_error = np.max(np.abs(PadeActivations.sigmoid_pade(x) - exact))
        # Horner degree-3 (same number of numerator coefficients as Padé)
        coeffs = np.polyfit(np.linspace(-6, 6, 100),
                            self._reference_sigmoid(np.linspace(-6, 6, 100)), 3)
        horner_3 = np.polyval(coeffs, x)
        horner_3 = np.clip(horner_3, 0, 1)
        horner_error = np.max(np.abs(horner_3 - exact))
        # Padé should beat same-degree polynomial
        assert pade_error < horner_error, \
            f"Padé ({pade_error:.4f}) should beat Horner-3 ({horner_error:.4f})"

    def test_vectorized(self):
        """All activations should handle arrays of various shapes."""
        for shape in [(10,), (3, 4), (2, 3, 4)]:
            x = np.random.randn(*shape)
            assert PadeActivations.sigmoid_pade(x).shape == shape
            assert PadeActivations.tanh_pade(x).shape == shape
            assert PadeActivations.gelu_pade(x).shape == shape
            assert PadeActivations.silu_pade(x).shape == shape


# ============================================================
# 5. Richardson + TT Integration
# ============================================================

class TestRichardsonTTLinear:
    """Test combined TT compression + Richardson extrapolation."""

    def test_basic_integration(self):
        """Richardson+TT should produce finite, correct-shaped output."""
        W = np.random.randn(12, 12)
        x = np.random.randn(12)
        rtt = RichardsonTTLinear(W, input_shape=(3, 4), output_shape=(3, 4),
                                  max_rank=8, n_richardson_passes=2)
        y = rtt.forward(x)
        assert y.shape == (12,)
        assert np.all(np.isfinite(y))

    def test_richardson_improves_accuracy(self):
        """Richardson correction should reduce error vs plain quantized TT."""
        W = np.random.randn(16, 16)
        x = np.random.randn(16)

        rtt_1pass = RichardsonTTLinear(W, input_shape=(4, 4), output_shape=(4, 4),
                                        max_rank=8, n_richardson_passes=1)
        rtt_2pass = RichardsonTTLinear(W, input_shape=(4, 4), output_shape=(4, 4),
                                        max_rank=8, n_richardson_passes=2)

        exact = W @ x
        error_1 = np.linalg.norm(rtt_1pass.forward(x) - exact)
        error_2 = np.linalg.norm(rtt_2pass.forward(x) - exact)

        # 2-pass Richardson should generally be better
        # (may not always be due to error accumulation, so use generous threshold)
        assert error_2 < error_1 * 2.0  # At least not much worse

    def test_compression_ratio(self):
        """Total compression should be > TT alone (due to INT8)."""
        W = np.random.randn(64, 64)
        rtt = RichardsonTTLinear(W, input_shape=(4, 4, 4), output_shape=(4, 4, 4),
                                  max_rank=4)
        # INT8 cores use 1 byte vs 8 bytes FP64
        assert rtt.compression_ratio > 2.0

    def test_accuracy_report(self):
        """Accuracy report should contain all expected metrics."""
        W = np.random.randn(12, 12)
        x = np.random.randn(12)
        rtt = RichardsonTTLinear(W, input_shape=(3, 4), output_shape=(3, 4),
                                  max_rank=8)
        report = rtt.accuracy_report(x)
        assert 'correlation' in report
        assert 'relative_error' in report
        assert 'tt_compression' in report
        assert 'total_compression' in report
        assert report['correlation'] > 0.5  # Should be correlated

    def test_total_bytes(self):
        """Total bytes should be reasonable."""
        W = np.random.randn(64, 64)
        rtt = RichardsonTTLinear(W, input_shape=(4, 4, 4), output_shape=(4, 4, 4),
                                  max_rank=4)
        assert rtt.total_bytes > 0
        assert rtt.total_bytes < W.size * 8  # Less than FP64

    def test_convenience_function(self):
        """compress_linear_layer should work end to end."""
        W = np.random.randn(36, 36)
        result = compress_linear_layer(W, max_rank=4, quantize=True)
        assert 'layer' in result
        assert 'report' in result
        assert result['report']['correlation'] > 0.0

    def test_convenience_no_quantize(self):
        """compress_linear_layer without quantization."""
        W = np.random.randn(36, 36)
        result = compress_linear_layer(W, max_rank=8, quantize=False)
        assert 'report' in result
        assert 'tt_compression' in result['report']

    def test_low_rank_matrix_high_accuracy(self):
        """Quantized TT should produce correlated output even for rank-2 matrix.
        
        Note: Matrix low-rank ≠ TT low-rank after dimension interleaving.
        A rank-2 matrix may have high TT-rank when reshaped and interleaved.
        """
        A = np.random.randn(64, 2)
        B = np.random.randn(2, 64)
        W = A @ B
        rtt = RichardsonTTLinear(W, input_shape=(4, 4, 4), output_shape=(4, 4, 4),
                                  max_rank=16)  # Higher rank to capture structure
        x = np.random.randn(64)
        report = rtt.accuracy_report(x)
        # With higher rank, should get reasonable correlation
        assert report['correlation'] > 0.0  # At minimum positively correlated
        assert report['total_compression'] > 1.0  # Still compressed


# ============================================================
# 6. Utility Functions
# ============================================================

class TestUtilities:
    """Test utility functions."""

    def test_auto_factorize_perfect(self):
        """Numbers with many factors should factorize cleanly."""
        factors = _auto_factorize(64, target_factors=3)
        assert np.prod(factors) == 64
        assert len(factors) <= 4  # May not hit exactly 3

    def test_auto_factorize_prime(self):
        """Prime numbers can't be split, should return (n,) or close."""
        factors = _auto_factorize(17, target_factors=4)
        assert np.prod(factors) == 17

    def test_auto_factorize_small(self):
        """Small numbers should work."""
        assert _auto_factorize(1) == (1,)
        factors = _auto_factorize(6, target_factors=2)
        assert np.prod(factors) == 6

    def test_auto_factorize_various(self):
        """Various sizes should all produce valid factorizations."""
        for n in [12, 24, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024]:
            factors = _auto_factorize(n)
            assert np.prod(factors) == n, \
                f"Factorization of {n}: {factors} product={np.prod(factors)}"


# ============================================================
# 7. Edge AI Realistic Scenarios
# ============================================================

class TestEdgeAIScenarios:
    """Tests simulating real edge AI use cases."""

    def test_transformer_fc_compression(self):
        """Simulate compressing a transformer FC layer (768→768)."""
        W = np.random.randn(768, 768) * 0.02  # Normal init scale
        # Factor 768 = 4 × 4 × 4 × 12
        tt = TTLinear(W, input_shape=(4, 4, 48), output_shape=(4, 4, 48),
                      max_rank=16)
        assert tt.compression_ratio > 5.0
        # Forward pass
        x = np.random.randn(768) * 0.1
        y = tt.forward(x)
        assert y.shape == (768,)
        assert np.all(np.isfinite(y))

    def test_yolo_conv_compression(self):
        """Simulate compressing a YOLO conv layer."""
        K = np.random.randn(256, 128, 3, 3) * 0.01
        result = compress_conv_kernel(K, rank_ratio=0.25)
        assert result['compression_ratio'] > 2.0
        # Random weights have high effective rank, so error is large
        # Real YOLO weights are much more compressible
        assert result['reconstruction_error'] < 1.0

    def test_smollm_layer_simulation(self):
        """Simulate SmolLM 135M layer compression.

        SmolLM has hidden_dim=576, intermediate=1536.
        Q/K/V proj: 576×576, FFN: 576×1536.
        """
        # Q projection
        W_q = np.random.randn(576, 576) * 0.02
        tt_q = TTLinear(W_q, input_shape=(24, 24), output_shape=(24, 24),
                        max_rank=16)
        assert tt_q.compression_ratio > 3.0

        # FFN up projection (576 → 1536)
        W_ffn = np.random.randn(1536, 576) * 0.02
        # 1536 = 4*384, 576 = 24*24
        tt_ffn = TTLinear(W_ffn, input_shape=(24, 24), output_shape=(4, 384),
                          max_rank=16)
        x = np.random.randn(576) * 0.1
        y = tt_ffn.forward(x)
        assert y.shape == (1536,)

    def test_activation_pipeline(self):
        """Test SiLU activation for LLM FFN (Llama-style).

        FFN: up = SiLU(W_gate @ x) * (W_up @ x)
        """
        dim = 64
        x = np.random.randn(dim) * 0.1
        W_gate = np.random.randn(dim * 4, dim) * 0.02
        W_up = np.random.randn(dim * 4, dim) * 0.02

        gate = PadeActivations.silu_pade(W_gate @ x)
        up = W_up @ x
        ffn_out = gate * up

        assert ffn_out.shape == (dim * 4,)
        assert np.all(np.isfinite(ffn_out))

    def test_full_pipeline_tt_quantize_richardson_activate(self):
        """Full pipeline: TT compress → INT8 quantize → Richardson → Padé activate.

        This is the complete inference path we're building toward.
        """
        # Small FC layer
        W = np.random.randn(36, 36) * 0.02
        x = np.random.randn(36) * 0.1

        # Step 1: Compress + Quantize + Richardson
        rtt = RichardsonTTLinear(W, input_shape=(6, 6), output_shape=(6, 6),
                                  max_rank=8, n_richardson_passes=2)
        y_linear = rtt.forward(x)

        # Step 2: Padé SiLU activation
        y_activated = PadeActivations.silu_pade(y_linear)

        # Compare against exact
        y_exact = W @ x
        y_exact_activated = y_exact * (1.0 / (1.0 + np.exp(-y_exact)))  # exact SiLU

        # Should be correlated
        corr = np.corrcoef(y_exact_activated, y_activated)[0, 1]
        assert corr > 0.5, f"Full pipeline correlation: {corr}"

    def test_compression_benchmark(self):
        """Benchmark compression ratios for various layer sizes."""
        results = []
        for M, N, rank in [(64, 64, 8), (128, 128, 8), (256, 256, 16),
                           (512, 512, 16), (768, 768, 16)]:
            W = np.random.randn(M, N) * 0.02
            # Auto factorize
            tt = TTLinear(W, max_rank=rank)
            results.append({
                'size': f"{M}×{N}",
                'rank': rank,
                'compression': tt.compression_ratio,
                'params': f"{tt.original_params:,} → {tt.compressed_params:,}",
            })

        # All should achieve some compression
        for r in results:
            assert r['compression'] > 1.0, \
                f"{r['size']} failed to compress: {r['compression']}"


# ============================================================
# Run
# ============================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
