from n4.core.numeric import PyFloat
from n4.tensor.tensor import Tensor
from n4.core import Value
import pytest

def test_init_correct_shape():
    data = [Value.from_int(i) for i in range(6)]
    t = Tensor(data, (2, 3))

    assert t.shape == (2, 3)
    assert t.ndim == 2
    assert t.size == 6


def test_init_wrong_shape():
    data = [Value.from_int(i) for i in range(5)]
    with pytest.raises(ValueError):
        Tensor(data, (2, 3))


def test_zeros():
    t = Tensor.zeros((2, 2), backend=PyFloat)

    for v in t._data:
        assert v.data.v == 0.0


def test_ones():
    t = Tensor.ones((2, 2), backend=PyFloat)

    for v in t._data:
        assert v.data.v == 1

def test_random_uniform_range():
    t = Tensor.random_uniform((10,), backend=PyFloat, low=-2, high=2)

    for v in t._data:
        assert -2 <= v.data.v <= 2

def test_to_list_2d():
    data = [Value.from_int(i) for i in range(6)]
    t = Tensor(data, (2, 3))

    lst = t.to_list()

    assert len(lst) == 2
    assert len(lst[0]) == 3


def test_get_single_element():
    data = [Value.from_int(i) for i in range(6)]
    t = Tensor(data, (2, 3))

    v: Value[PyFloat] = t[1, 2]  # ty:ignore[invalid-assignment]

    assert isinstance(v, Value)
    assert v.data.v == 5.0


def test_get_slice():
    data: list[Value[PyFloat]] = [Value.from_int(i) for i in range(6)]
    t = Tensor(data, (2, 3))

    row: Tensor[PyFloat] = t[1] # ty:ignore[invalid-assignment]
    
    assert isinstance(row, Tensor)
    assert row.shape == (3,)
    assert row._data[0].data.v == 1


def test_get_out_of_bounds():
    data = [Value.from_int(i) for i in range(6)]
    t = Tensor(data, (2, 3))

    with pytest.raises(IndexError):
        t[3]

def test_reshape_valid():
    data = [Value.from_int(i) for i in range(6)]
    t = Tensor(data, (2, 3))

    t2 = t.reshape((3, 2))

    assert t2.shape == (3, 2)
    assert t2.size == 6


def test_reshape_invalid():
    data = [Value.from_int(i) for i in range(6)]
    t = Tensor(data, (2, 3))

    with pytest.raises(ValueError):
        t.reshape((4, 4))



def test_tensor_add():
    a = Tensor([Value.from_int(1), Value.from_int(2)], (2,))
    b = Tensor([Value.from_int(3), Value.from_int(4)], (2,))

    c = a + b

    assert c._data[0].data.v == 4
    assert c._data[1].data.v == 6


def test_tensor_scalar_add():
    a = Tensor([Value.from_int(1), Value.from_int(2)], (2,))
    s = Value.from_int(5)

    c = a + s

    assert c._data[0].data.v == 6
    assert c._data[1].data.v == 7


def test_tensor_shape_mismatch():
    a = Tensor([Value.from_int(1), Value.from_int(2)], (2,))
    b = Tensor([Value.from_int(1), Value.from_int(2), Value.from_int(3)], (3,))

    with pytest.raises(NotImplementedError):
        a + b

def test_neg():
    t = Tensor([Value.from_int(2), Value.from_int(-3)], (2,))
    r = -t

    assert r._data[0].data.v == -2
    assert r._data[1].data.v == 3


def test_matmul_basic():
    a = Tensor([Value.from_int(1), Value.from_int(2),
                Value.from_int(3), Value.from_int(4)], (2, 2))

    b = Tensor([Value.from_int(5), Value.from_int(6),
                Value.from_int(7), Value.from_int(8)], (2, 2))

    c = a @ b

    assert c.shape == (2, 2)
    assert c._data[0].data.v == 19
    assert c._data[1].data.v == 22
    assert c._data[2].data.v == 43
    assert c._data[3].data.v == 50


def test_matmul_shape_error():
    a = Tensor([Value.from_int(1)], (1,1))
    b = Tensor([Value.from_int(1)], (1,))

    with pytest.raises(ValueError):
        a @ b
#
#
#def test_sum_all():
#    t = Tensor([Value.from_int(1), Value.from_int(2), Value.from_int(3)], (3,))
#
#    s : Value[PyFloat] = t.sum()
#
#    assert s.data.v == 6
#
#
#def test_sum_dim():
#    data = [Value(i) for i in range(6)]
#    t = Tensor(data, (2, 3))
#
#    s = t.sum(dim=0)
#
#    assert s.shape == (3,)
#    assert s._data[0].data == 3
#    assert s._data[1].data == 5
#    assert s._data[2].data == 7
#
#
#def test_sum_invalid_dim():
#    t = Tensor([Value(1)], (1,))
#
#    with pytest.raises(ValueError):
#        t.sum(dim=3)
#
#
## ============================================================
## MEAN
## ============================================================
#
#def test_mean_all():
#    t = Tensor([Value(1), Value(3)], (2,))
#
#    m = t.mean()
#
#    assert abs(m.data - 2.0) < 1e-6
#
#
#def test_mean_dim():
#    data = [Value(1), Value(3), Value(5), Value(7)]
#    t = Tensor(data, (2,2))
#
#    m = t.mean(dim=0)
#
#    assert m._data[0].data == 3
#    assert m._data[1].data == 5
#
#
## ============================================================
## GRADIENT TESTS
## ============================================================
#
#def test_grad_through_sum():
#    t = Tensor([Value(1), Value(2)], (2,))
#
#    s = t.sum()
#    s.backward()
#
#    for v in t._data:
#        assert v.get_grad() == 1
#
#
#def test_grad_through_matmul():
#    a = Tensor([Value(1), Value(2),
#                Value(3), Value(4)], (2, 2))
#
#    b = Tensor([Value(5), Value(6),
#                Value(7), Value(8)], (2, 2))
#
#    c = a @ b
#    s = c.sum()
#
#    s.backward()
#
#    # Проверяем, что градиенты ненулевые
#    for v in a._data:
#        assert v.get_grad() != 0
#
#    for v in b._data:
#        assert v.get_grad() != 0