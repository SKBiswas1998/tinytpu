"""
Numerical methods from Killingbeck Microcomputer Algorithms (1991).
Adapted for quantized edge inference.
"""
import numpy as np


class QuantizationExtrapolator:
    """Richardson extrapolation for INT8 quantization error reduction."""

    def extrapolate(self, func, x, bits_list=None):
        if bits_list is None:
            bits_list = [4, 6, 8]
        results = []
        for bits in bits_list:
            scale = (2 ** (bits - 1)) - 1
            q = np.clip(np.round(x * scale), -scale, scale)
            dq = q / scale
            results.append(func(dq))
        if len(results) >= 3:
            return (4 * results[2] - results[1]) / 3
        return results[-1]


class IterativeEigen:
    """HITTER iterative eigenvalue decomposition for on-device PCA."""

    def power_iteration(self, A, num_iters=100, tol=1e-6):
        n = A.shape[0]
        v = np.random.randn(n)
        v = v / np.linalg.norm(v)
        for _ in range(num_iters):
            Av = A @ v
            eigenvalue = v @ Av
            v_new = Av / (np.linalg.norm(Av) + 1e-10)
            if np.abs(np.abs(v_new @ v) - 1.0) < tol:
                v = v_new
                break
            v = v_new
        return eigenvalue, v

    def top_k(self, A, k=3, num_iters=100):
        eigenvalues = []
        eigenvectors = []
        A_deflated = A.copy()
        for _ in range(k):
            val, vec = self.power_iteration(A_deflated, num_iters)
            eigenvalues.append(val)
            eigenvectors.append(vec)
            A_deflated = A_deflated - val * np.outer(vec, vec)
        return np.array(eigenvalues), np.array(eigenvectors).T


class FastActivations:
    """Horner-form polynomial activations for INT8 inference."""

    @staticmethod
    def relu_poly(x, order=3):
        return np.where(x > 0, x, 0.01 * x)

    @staticmethod
    def sigmoid_horner(x):
        x_clip = np.clip(x, -5, 5)
        return 0.5 + x_clip * (0.25 + x_clip * x_clip * (-0.0208333))

    @staticmethod
    def gelu_poly(x):
        return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))

    @staticmethod
    def swish_poly(x):
        return x * FastActivations.sigmoid_horner(x)
