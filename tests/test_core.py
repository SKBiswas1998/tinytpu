"""Tests for TinyTPU core — systolic array simulation and HAL."""

import pytest
import numpy as np
from tinytpu.core.tpu import TinyTPU
from tinytpu.hal.detect import detect_hardware


class TestTinyTPU:

    def test_creation(self):
        tpu = TinyTPU()
        assert tpu is not None

    def test_matmul_identity(self):
        tpu = TinyTPU()
        A = np.eye(4, dtype=np.int8)
        B = np.array([[1], [2], [3], [4]], dtype=np.int8)
        result = tpu.matmul(A, B)
        np.testing.assert_array_equal(result.flatten()[:4], [1, 2, 3, 4])

    def test_matmul_2x2(self):
        tpu = TinyTPU()
        A = np.array([[1, 2], [3, 4]], dtype=np.int8)
        B = np.array([[5, 6], [7, 8]], dtype=np.int8)
        result = tpu.matmul(A, B)
        expected = np.array([[19, 22], [43, 50]])
        np.testing.assert_array_equal(result, expected)

    def test_matmul_larger(self):
        tpu = TinyTPU()
        np.random.seed(42)
        # Use int32 to avoid overflow
        A = np.random.randint(-10, 10, (8, 8)).astype(np.int32)
        B = np.random.randint(-10, 10, (8, 8)).astype(np.int32)
        result = tpu.matmul(A, B)
        expected = A @ B
        np.testing.assert_array_equal(result, expected)

    def test_matmul_non_square(self):
        tpu = TinyTPU()
        A = np.array([[1, 2, 3]], dtype=np.int32)       # 1x3
        B = np.array([[1], [2], [3]], dtype=np.int32)    # 3x1
        result = tpu.matmul(A, B)
        assert result.shape == (1, 1)
        assert result[0, 0] == 14  # 1*1 + 2*2 + 3*3

    def test_conv2d(self):
        tpu = TinyTPU()
        # conv2d expects 4D: (N, C, H, W) input, (OC, IC, KH, KW) kernel
        x = np.ones((1, 1, 4, 4), dtype=np.int32)
        kernel = np.ones((1, 1, 2, 2), dtype=np.int32)
        result = tpu.conv2d(x, kernel)
        # Each output is sum of 2x2 patch of ones = 4
        assert result[0, 0, 0, 0] == 4


class TestHardwareDetection:

    def test_returns_info(self):
        hw = detect_hardware()
        assert hw is not None
        assert hasattr(hw, "devices")
        assert hasattr(hw, "recommended")

    def test_cpu_always_present(self):
        hw = detect_hardware()
        cpu_devices = [d for d in hw.devices if "cpu" in d.backend.lower() or "cpu" in d.name.lower()]
        assert len(cpu_devices) >= 1

    def test_devices_have_structure(self):
        hw = detect_hardware()
        for device in hw.devices:
            assert hasattr(device, "name")
            assert hasattr(device, "backend")
            assert hasattr(device, "available")
            assert isinstance(device.available, bool)

    def test_recommendation_is_string(self):
        hw = detect_hardware()
        assert isinstance(hw.recommended, str)
        assert len(hw.recommended) > 0
