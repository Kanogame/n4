import math
from typing import Sequence, Tuple
from n4.numeric import PyFloat
from n4.tensor import Tensor
from n4.nn.tanh_layer import TanhLayer
from n4.nn.softmax_layer import SoftmaxLayer
from n4.nn.dence_layer import DenseLayer


def make_tensor_from_floats(
    vals: Sequence[float], shape: Tuple[int, ...]
) -> Tensor[PyFloat]:
    from n4.core import Value

    data = [Value.from_float(v, PyFloat) for v in vals]
    return Tensor(data, shape)


def test_tanh_layer_forward() -> None:
    vals = [-1.0, 0.0, 1.0]
    x = make_tensor_from_floats(vals, (1, 3))

    layer = TanhLayer(PyFloat)
    out = layer.forward_pass(x)

    for i, v in enumerate(out._data):
        assert math.isclose(v.data.v, math.tanh(vals[i]), rel_tol=1e-6)


def test_softmax_layer_probs_sum_to_one() -> None:
    # two rows, three classes
    vals = [1.0, 2.0, 3.0, 0.0, 0.0, 0.0]
    x = make_tensor_from_floats(vals, (2, 3))

    layer = SoftmaxLayer(PyFloat)
    out = layer.forward_pass(x)

    # each row sums to 1
    for r in range(2):
        row = [out._data[r * 3 + c] for c in range(3)]
        s = sum([v.data.v for v in row])
        assert abs(s - 1.0) < 1e-6


def test_dense_layer_shape_and_parameters() -> None:
    in_f = 4
    out_f = 2
    layer = DenseLayer(in_f, out_f, PyFloat)

    x = Tensor.ones((3, in_f), backend=PyFloat)
    out = layer.forward_pass(x)

    assert out.shape == (3, out_f)
    # parameters length should match weights + bias
    assert len(layer.parameters()) == in_f * out_f + out_f
