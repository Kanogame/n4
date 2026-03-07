from n4.numeric import PyFloat
from n4.nn.dence_layer import DenseLayer
from n4.nn.sequential import Sequential
from n4.tensor import Tensor
import pytest


def test_sequential_forward_and_backward_chain() -> None:
    batch = 2
    in_f = 2
    hidden = 3
    out_f = 1

    l1 = DenseLayer(in_f, hidden, PyFloat)
    l2 = DenseLayer(hidden, out_f, PyFloat)

    l1.weights = Tensor.ones((hidden, in_f), backend=PyFloat)
    l1.bias = Tensor.zeros((hidden,), backend=PyFloat)

    l2.weights = Tensor.ones((out_f, hidden), backend=PyFloat)
    l2.bias = Tensor.zeros((out_f,), backend=PyFloat)

    seq = Sequential(l1, l2)

    x = Tensor.ones((batch, in_f), backend=PyFloat)
    out = seq.forward_pass(x)

    expected_single_out = float(hidden * in_f)
    for v in out._data:
        assert pytest.approx(v.data.v) == expected_single_out

    loss = out.sum()
    loss.backward()

    expected_l2_w_grad = float(batch * in_f)
    expected_l1_w_grad = float(batch)

    l2_params = l2.parameters()
    for i, p in enumerate(l2_params):
        if i < hidden:
            assert pytest.approx(p.grad.v) == expected_l2_w_grad
        else:
            assert pytest.approx(p.grad.v) == float(batch)

    l1_params = l1.parameters()
    for i, p in enumerate(l1_params):
        if i < hidden * in_f:
            assert pytest.approx(p.grad.v) == expected_l1_w_grad
        else:
            assert pytest.approx(p.grad.v) == float(batch)
