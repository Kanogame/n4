from n4.core.numeric import PyFloat
from n4.tensor.tensor import Tensor
from n4.core import Value
import pytest


def test_init_correct_shape() -> None:
    data = [Value.from_int(i) for i in range(6)]
    t = Tensor(data, (2, 3))

    assert t.shape == (2, 3)
    assert t.ndim == 2
    assert t.size == 6


def test_init_wrong_shape() -> None:
    data = [Value.from_int(i) for i in range(5)]
    with pytest.raises(ValueError):
        Tensor(data, (2, 3))


def test_zeros() -> None:
    t = Tensor.zeros((2, 2), backend=PyFloat)

    for v in t._data:
        assert v.data.v == 0.0


def test_ones() -> None:
    t = Tensor.ones((2, 2), backend=PyFloat)

    for v in t._data:
        assert v.data.v == 1


def test_random_uniform_range() -> None:
    t = Tensor.random_uniform((10,), backend=PyFloat, low=-2, high=2)

    for v in t._data:
        assert -2 <= v.data.v <= 2


def test_to_list_2d() -> None:
    data = [Value.from_int(i) for i in range(6)]
    t = Tensor(data, (2, 3))

    lst = t.to_list()

    assert len(lst) == 2
    assert len(lst[0]) == 3


def test_get_single_element() -> None:
    data = [Value.from_int(i) for i in range(6)]
    t = Tensor(data, (2, 3))

    res = t[1, 2]
    assert type(res) is Value

    v: Value[PyFloat] = res

    assert isinstance(v, Value)
    assert v.data.v == 5.0


def test_get_slice() -> None:
    data: list[Value[PyFloat]] = [Value.from_int(i) for i in range(6)]
    t = Tensor(data, (2, 3))

    res = t[1]
    assert type(res) is Tensor

    row: Tensor[PyFloat] = res

    assert isinstance(row, Tensor)
    assert row.shape == (3,)
    assert row._data[0].data.v == 1


def test_get_out_of_bounds() -> None:
    data = [Value.from_int(i) for i in range(6)]
    t = Tensor(data, (2, 3))

    with pytest.raises(IndexError):
        t[3]


def test_reshape_valid() -> None:
    data = [Value.from_int(i) for i in range(6)]
    t = Tensor(data, (2, 3))

    t2 = t.reshape((3, 2))

    assert t2.shape == (3, 2)
    assert t2.size == 6


def test_reshape_invalid() -> None:
    data = [Value.from_int(i) for i in range(6)]
    t = Tensor(data, (2, 3))

    with pytest.raises(ValueError):
        t.reshape((4, 4))
