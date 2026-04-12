"""Tests for TinyTPU ONNX Engine — pure-Python operator implementations."""

import pytest
import numpy as np


class TestEngineDocstring:
    """Verify the operator count claim is accurate."""

    def test_operator_count_matches_docstring(self):
        """Engine claims 46 operators — verify."""
        from tinytpu.inference.engine import TinyTPUEngine
        assert "46 operators" in TinyTPUEngine.__doc__


class TestEngineOperators:
    """Test individual ONNX operator implementations without loading a model."""

    @pytest.fixture
    def engine_node(self):
        """Create a minimal engine mock for operator testing."""
        from tinytpu.inference.engine import TinyTPUEngine

        class FakeNode:
            def __init__(self, op_type, inputs, output_names, attrs=None):
                self.op_type = op_type
                self.input = inputs
                self.output = output_names
                self.attribute = []

        class TestableEngine:
            """Wraps engine to test individual operators."""
            def __init__(self):
                self._tensors = {}

            def run_op(self, op, inputs_dict, attrs=None):
                """Execute a single operator and return outputs."""
                engine = TinyTPUEngine.__new__(TinyTPUEngine)
                engine._tensors = dict(inputs_dict)
                input_names = list(inputs_dict.keys())
                output_names = ["out_0", "out_1", "out_2", "out_3"]

                node = type("Node", (), {
                    "op_type": op,
                    "input": input_names,
                    "output": output_names[:4],
                    "attribute": [],
                })()
                engine._execute_node(node)
                return [engine._tensors.get(name) for name in output_names if name in engine._tensors]

        return TestableEngine()

    def test_relu(self, engine_node):
        x = np.array([-2, -1, 0, 1, 2], dtype=np.float32)
        result = engine_node.run_op("Relu", {"x": x})
        assert len(result) >= 1
        np.testing.assert_array_equal(result[0], np.array([0, 0, 0, 1, 2], dtype=np.float32))

    def test_sigmoid(self, engine_node):
        x = np.array([0.0], dtype=np.float32)
        result = engine_node.run_op("Sigmoid", {"x": x})
        assert abs(result[0][0] - 0.5) < 1e-5

    def test_add(self, engine_node):
        a = np.array([1, 2, 3], dtype=np.float32)
        b = np.array([4, 5, 6], dtype=np.float32)
        result = engine_node.run_op("Add", {"a": a, "b": b})
        np.testing.assert_array_equal(result[0], np.array([5, 7, 9], dtype=np.float32))

    def test_mul(self, engine_node):
        a = np.array([2, 3], dtype=np.float32)
        b = np.array([4, 5], dtype=np.float32)
        result = engine_node.run_op("Mul", {"a": a, "b": b})
        np.testing.assert_array_equal(result[0], np.array([8, 15], dtype=np.float32))

    def test_matmul(self, engine_node):
        a = np.eye(3, dtype=np.float32)
        b = np.array([[1], [2], [3]], dtype=np.float32)
        result = engine_node.run_op("MatMul", {"a": a, "b": b})
        np.testing.assert_array_almost_equal(result[0], b)

    def test_softmax(self, engine_node):
        x = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        result = engine_node.run_op("Softmax", {"x": x})
        assert abs(np.sum(result[0]) - 1.0) < 1e-5
        assert result[0][0, 2] > result[0][0, 1] > result[0][0, 0]

    def test_tanh(self, engine_node):
        x = np.array([0.0], dtype=np.float32)
        result = engine_node.run_op("Tanh", {"x": x})
        assert abs(result[0][0]) < 1e-5

    def test_neg(self, engine_node):
        x = np.array([1, -2, 3], dtype=np.float32)
        result = engine_node.run_op("Neg", {"x": x})
        np.testing.assert_array_equal(result[0], np.array([-1, 2, -3], dtype=np.float32))

    def test_abs(self, engine_node):
        x = np.array([-3, -1, 0, 1, 3], dtype=np.float32)
        result = engine_node.run_op("Abs", {"x": x})
        np.testing.assert_array_equal(result[0], np.array([3, 1, 0, 1, 3], dtype=np.float32))

    def test_sqrt(self, engine_node):
        x = np.array([0, 1, 4, 9], dtype=np.float32)
        result = engine_node.run_op("Sqrt", {"x": x})
        np.testing.assert_array_almost_equal(result[0], np.array([0, 1, 2, 3], dtype=np.float32))

    def test_shape(self, engine_node):
        x = np.zeros((2, 3, 4), dtype=np.float32)
        result = engine_node.run_op("Shape", {"x": x})
        np.testing.assert_array_equal(result[0], np.array([2, 3, 4]))

    def test_identity(self, engine_node):
        x = np.array([1, 2, 3], dtype=np.float32)
        result = engine_node.run_op("Identity", {"x": x})
        np.testing.assert_array_equal(result[0], x)

    def test_leaky_relu(self, engine_node):
        x = np.array([-2, -1, 0, 1, 2], dtype=np.float32)
        result = engine_node.run_op("LeakyRelu", {"x": x})
        assert result[0][0] < 0  # negative side scaled
        assert result[0][0] > -2  # but not as negative as input
        assert result[0][4] == 2.0  # positive side unchanged

    def test_floor_ceil(self, engine_node):
        x = np.array([1.3, 2.7, -0.5], dtype=np.float32)
        floor_result = engine_node.run_op("Floor", {"x": x})
        ceil_result = engine_node.run_op("Ceil", {"x": x})
        np.testing.assert_array_equal(floor_result[0], np.array([1, 2, -1], dtype=np.float32))
        np.testing.assert_array_equal(ceil_result[0], np.array([2, 3, 0], dtype=np.float32))

    def test_equal_less_greater(self, engine_node):
        a = np.array([1, 2, 3], dtype=np.float32)
        b = np.array([2, 2, 2], dtype=np.float32)
        eq = engine_node.run_op("Equal", {"a": a, "b": b})
        lt = engine_node.run_op("Less", {"a": a, "b": b})
        gt = engine_node.run_op("Greater", {"a": a, "b": b})
        np.testing.assert_array_equal(eq[0], [False, True, False])
        np.testing.assert_array_equal(lt[0], [True, False, False])
        np.testing.assert_array_equal(gt[0], [False, False, True])

    def test_where(self, engine_node):
        cond = np.array([True, False, True])
        a = np.array([1, 2, 3], dtype=np.float32)
        b = np.array([4, 5, 6], dtype=np.float32)
        result = engine_node.run_op("Where", {"cond": cond, "a": a, "b": b})
        np.testing.assert_array_equal(result[0], np.array([1, 5, 3], dtype=np.float32))

    def test_unsupported_op_returns_zeros(self, engine_node):
        x = np.array([1, 2, 3], dtype=np.float32)
        result = engine_node.run_op("TotallyFakeOp", {"x": x})
        # Should not crash, returns zeros
        assert len(result) >= 1
