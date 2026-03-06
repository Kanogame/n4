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


def test_tensor_add() -> None:
    a = Tensor([Value.from_int(1), Value.from_int(2)], (2,))
    b = Tensor([Value.from_int(3), Value.from_int(4)], (2,))

    c = a + b

    assert c._data[0].data.v == 4
    assert c._data[1].data.v == 6


def test_tensor_scalar_add() -> None:
    a = Tensor([Value.from_int(1), Value.from_int(2)], (2,))
    s = Value.from_int(5)

    c = a + s

    assert c._data[0].data.v == 6
    assert c._data[1].data.v == 7


def test_tensor_shape_mismatch() -> None:
    a = Tensor([Value.from_int(1), Value.from_int(2)], (2,))
    b = Tensor([Value.from_int(1), Value.from_int(2), Value.from_int(3)], (3,))

    with pytest.raises(RuntimeError):
        a + b


def test_neg() -> None:
    t = Tensor([Value.from_int(2), Value.from_int(-3)], (2,))
    r = -t

    assert r._data[0].data.v == -2
    assert r._data[1].data.v == 3


def test_matmul_basic() -> None:
    a = Tensor(
        [Value.from_int(1), Value.from_int(2), Value.from_int(3), Value.from_int(4)],
        (2, 2),
    )

    # a:
    # 1 2
    # 3 4

    b = Tensor(
        [Value.from_int(5), Value.from_int(6), Value.from_int(7), Value.from_int(8)],
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
    a = Tensor([Value.from_int(1)], (1, 1))
    b = Tensor([Value.from_int(1)], (1,))

    with pytest.raises(ValueError):
        a @ b


def test_sum_all() -> None:
    t = Tensor([Value.from_int(1), Value.from_int(2), Value.from_int(3)], (3,))

    s: Value[PyFloat] = t.sum()

    assert s.data.v == 6


def test_sum_dim() -> None:
    data = [Value.from_int(i) for i in range(6)]
    t = Tensor(data, (2, 3))

    s = t.sum_dim(dim=0)

    assert s.shape == (3,)
    assert s._data[0].data.v == 3
    assert s._data[1].data.v == 5
    assert s._data[2].data.v == 7


def test_sum_invalid_dim() -> None:
    t = Tensor([Value.from_int(1)], (1,))

    with pytest.raises(ValueError):
        t.sum_dim(dim=3)


def test_mean_all() -> None:
    t = Tensor([Value.from_int(1), Value.from_int(3)], (2,))

    m = t.mean()

    assert abs(m.data.v - 2.0) < 1e-6


def test_mean_dim() -> None:
    data = [Value.from_int(1), Value.from_int(3), Value.from_int(5), Value.from_int(7)]
    t = Tensor(data, (2, 2))

    m = t.mean_dim(dim=0)

    assert m._data[0].data.v == 3
    assert m._data[1].data.v == 5


def test_grad_through_sum() -> None:
    t = Tensor([Value.from_int(1), Value.from_int(2)], (2,))

    s = t.sum()
    s.backward()

    for v in t._data:
        assert v.grad.v == 1


def test_grad_through_matmul() -> None:
    a = Tensor(
        [Value.from_int(1), Value.from_int(2), Value.from_int(3), Value.from_int(4)],
        (2, 2),
    )

    b = Tensor(
        [Value.from_int(5), Value.from_int(6), Value.from_int(7), Value.from_int(8)],
        (2, 2),
    )

    c = a @ b
    s = c.sum()

    s.backward()

    # Проверяем, что градиенты ненулевые
    for v in a._data:
        assert v.grad.v != 0

    for v in b._data:
        assert v.grad.v != 0
