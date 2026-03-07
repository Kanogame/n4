from n4.numeric import PyFloat
from n4.nn.dence_layer import DenseLayer
from n4.tensor import Tensor
import pytest


def test_dense_init_shapes_and_parameters() -> None:
    layer = DenseLayer(3, 2, PyFloat)

    # weights shape (out_features, in_features)
    assert layer.weights.shape == (2, 3)
    assert layer.bias.shape == (2,)

    # parameters length = out*in + out
    params = layer.parameters()
    assert len(params) == 2 * 3 + 2


def test_forward_types_and_sizes() -> None:
    in_f = 3
    out_f = 2
    batch = 4

    layer = DenseLayer(in_f, out_f, PyFloat)

    # make weights all ones and bias zero for deterministic behaviour
    layer.weights = Tensor.ones((out_f, in_f), backend=PyFloat)
    layer.bias = Tensor.zeros((out_f,), backend=PyFloat)

    x = Tensor.ones((batch, in_f), backend=PyFloat)
    out = layer(x)

    assert out.shape == (batch, out_f)
    # each output element is dot(ones, ones) = in_f
    for v in out._data:
        assert pytest.approx(v.data.v) == float(in_f)


def test_backward_single_layer() -> None:
    in_f = 3
    out_f = 2
    batch = 5

    layer = DenseLayer(in_f, out_f, PyFloat)
    # deterministic params
    layer.weights = Tensor.ones((out_f, in_f), backend=PyFloat)
    layer.bias = Tensor.zeros((out_f,), backend=PyFloat)

    x = Tensor.ones((batch, in_f), backend=PyFloat)
    out = layer(x)
    loss = out.sum()

    # backward
    loss.backward()

    # Each weight gradient should equal sum_i x_{i,k} which is batch (x are ones)
    for p in layer.parameters():
        assert pytest.approx(p.grad.v) == float(batch)
