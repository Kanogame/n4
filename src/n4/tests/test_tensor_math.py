from n4.numeric import PyFloat
from n4.tensor.tensor import Tensor
from n4.core import Value
from .helpers import new_value
import pytest


def test_tensor_add() -> None:
    a = Tensor([new_value(1), new_value(2)], (2,))
    b = Tensor([new_value(3), new_value(4)], (2,))

    c = a + b

    assert c._data[0].data.v == 4
    assert c._data[1].data.v == 6


def test_tensor_scalar_add() -> None:
    a = Tensor([new_value(1), new_value(2)], (2,))
    s = new_value(5)

    c = a + s

    assert c._data[0].data.v == 6
    assert c._data[1].data.v == 7


def test_tensor_shape_mismatch() -> None:
    a = Tensor([new_value(1), new_value(2)], (2,))
    b = Tensor([new_value(1), new_value(2), new_value(3)], (3,))

    with pytest.raises(RuntimeError):
        _ = a + b


def test_neg() -> None:
    t = Tensor([new_value(2), new_value(-3)], (2,))
    r = -t

    assert r._data[0].data.v == -2
    assert r._data[1].data.v == 3


def test_matmul_basic() -> None:
    a = Tensor(
        [new_value(1), new_value(2), new_value(3), new_value(4)],
        (2, 2),
    )

    # a:
    # 1 2
    # 3 4

    b = Tensor(
        [new_value(5), new_value(6), new_value(7), new_value(8)],
        (2, 2),
    )

    # b:
    # 5 6
    # 7 8

    c = a @ b

    assert c.shape == (2, 2)
    assert c._data[0].data.v == 19
    assert c._data[1].data.v == 22
    assert c._data[2].data.v == 43
    assert c._data[3].data.v == 50


def test_matmul_shape_error() -> None:
    a = Tensor([new_value(1)], (1, 1))
    b = Tensor([new_value(1)], (1,))

    with pytest.raises(ValueError):
        _ = a @ b


def test_sum_all() -> None:
    t = Tensor([new_value(1), new_value(2), new_value(3)], (3,))

    s: Value[PyFloat] = t.sum()

    assert s.data.v == 6


def test_sum_dim() -> None:
    data = [new_value(i) for i in range(6)]
    t = Tensor(data, (2, 3))

    s = t.sum_dim(dim=0)

    assert s.shape == (3,)
    assert s._data[0].data.v == 3
    assert s._data[1].data.v == 5
    assert s._data[2].data.v == 7


def test_sum_invalid_dim() -> None:
    t = Tensor([new_value(1)], (1,))

    with pytest.raises(ValueError):
        t.sum_dim(dim=3)


def test_mean_all() -> None:
    t = Tensor([new_value(1), new_value(3)], (2,))

    m = t.mean()

    assert abs(m.data.v - 2.0) < 1e-6


def test_mean_dim() -> None:
    data = [new_value(1), new_value(3), new_value(5), new_value(7)]
    t = Tensor(data, (2, 2))

    m = t.mean_dim(dim=0)

    assert m._data[0].data.v == 3
    assert m._data[1].data.v == 5
