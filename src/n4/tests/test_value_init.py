import pytest
from .helpers import new_value
from n4.numeric import PyFloat, NumpyFloat
from n4.core import Value
from n4.op import Add


def test_value_initialization() -> None:
    x = new_value(5)
    assert x.data.v == 5.0
    assert x.grad.v == 0.0


def test_value_from_int() -> None:
    x = Value.from_int(3, PyFloat)
    assert x.data.v == 3.0
    assert x.grad.v == 0.0


def test_value_from_float() -> None:
    x = Value.from_float(2.5, PyFloat)
    assert x.data.v == 2.5


def test_value_get_float() -> None:
    x = new_value(7)
    assert x.get_float() == 7.0


def test_value_zero_grad() -> None:
    x = new_value(3)
    x.grad = PyFloat.from_float(5.0)
    x.zero_grad()
    assert x.grad.v == 0.0


def test_value_repr() -> None:
    x = new_value(2)
    r = repr(x)
    assert "Value(" in r
    assert "data=" in r
    assert "grad=" in r


def test_value_str() -> None:
    x = new_value(4)
    s = str(x)
    assert s.startswith("Value(")


def test_value_repr_shows_op_name() -> None:
    a = new_value(1)
    b = new_value(2)
    c = a + b
    assert "Add" in repr(c)


def test_value_backend_mismatch_raises() -> None:
    """Проверяет что операция над разными бекендами выбрасывает ошибку"""
    a = new_value(1)
    b = Value.from_float(1.0, NumpyFloat)
    with pytest.raises(TypeError):
        Add([a, b])  # type: ignore[arg-type]


def test_value_collect_graph() -> None:
    a = new_value(2)
    b = new_value(3)
    c = a + b
    graph = c.collect_graph()
    assert graph is not None
