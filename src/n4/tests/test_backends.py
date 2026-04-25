import math
import pytest
from n4.numeric import DecimalNum, NumpyFloat, PyFloat
from n4.core import Value
from n4.tensor import Tensor


def test_decimal_from_float() -> None:
    x = DecimalNum.from_float(3.14)
    assert abs(x.get_float() - 3.14) < 1e-9


def test_decimal_random_uniform() -> None:
    x = DecimalNum.random_uniform(-1.0, 1.0)
    assert -1.0 <= x.get_float() <= 1.0


def test_decimal_arithmetic() -> None:
    a = DecimalNum.from_float(4.0)
    b = DecimalNum.from_float(2.0)
    assert abs((a + b).get_float() - 6.0) < 1e-9
    assert abs((a - b).get_float() - 2.0) < 1e-9
    assert abs((a * b).get_float() - 8.0) < 1e-9
    assert abs((a / b).get_float() - 2.0) < 1e-9
    assert abs((a ** b).get_float() - 16.0) < 1e-9


def test_decimal_neg() -> None:
    a = DecimalNum.from_float(5.0)
    assert (-a).get_float() == -5.0


def test_decimal_lt() -> None:
    a = DecimalNum.from_float(1.0)
    b = DecimalNum.from_float(2.0)
    assert a < b
    assert not b < a


def test_decimal_exp() -> None:
    a = DecimalNum.from_float(1.0)
    assert abs(a.exp().get_float() - math.e) < 1e-9


def test_decimal_tanh() -> None:
    a = DecimalNum.from_float(0.0)
    assert abs(a.tanh().get_float()) < 1e-9


def test_decimal_log() -> None:
    a = DecimalNum.from_float(1.0)
    assert abs(a.log().get_float()) < 1e-9


def test_decimal_sqrt() -> None:
    a = DecimalNum.from_float(4.0)
    assert abs(a.sqrt().get_float() - 2.0) < 1e-9


def test_decimal_repr() -> None:
    a = DecimalNum.from_float(1.5)
    assert "DecimalNum" in repr(a)


def test_decimal_autograd_add() -> None:
    a = Value.from_float(2.0, DecimalNum)
    b = Value.from_float(3.0, DecimalNum)
    c = a + b
    c.backward()
    assert abs(a.grad.get_float() - 1.0) < 1e-9
    assert abs(b.grad.get_float() - 1.0) < 1e-9


def test_numpy_from_float() -> None:
    x = NumpyFloat.from_float(2.71)
    assert abs(x.get_float() - 2.71) < 1e-9


def test_numpy_random_uniform() -> None:
    x = NumpyFloat.random_uniform(-1.0, 1.0)
    assert -1.0 <= x.get_float() <= 1.0


def test_numpy_arithmetic() -> None:
    a = NumpyFloat.from_float(6.0)
    b = NumpyFloat.from_float(3.0)
    assert abs((a + b).get_float() - 9.0) < 1e-9
    assert abs((a - b).get_float() - 3.0) < 1e-9
    assert abs((a * b).get_float() - 18.0) < 1e-9
    assert abs((a / b).get_float() - 2.0) < 1e-9
    assert abs((a ** b).get_float() - 216.0) < 1e-9


def test_numpy_neg() -> None:
    a = NumpyFloat.from_float(4.0)
    assert (-a).get_float() == -4.0


def test_numpy_lt() -> None:
    a = NumpyFloat.from_float(1.0)
    b = NumpyFloat.from_float(2.0)
    assert a < b
    assert not b < a


def test_numpy_exp() -> None:
    a = NumpyFloat.from_float(0.0)
    assert abs(a.exp().get_float() - 1.0) < 1e-9


def test_numpy_tanh() -> None:
    a = NumpyFloat.from_float(0.0)
    assert abs(a.tanh().get_float()) < 1e-9


def test_numpy_log() -> None:
    a = NumpyFloat.from_float(1.0)
    assert abs(a.log().get_float()) < 1e-9


def test_numpy_sqrt() -> None:
    a = NumpyFloat.from_float(9.0)
    assert abs(a.sqrt().get_float() - 3.0) < 1e-9


def test_numpy_repr() -> None:
    a = NumpyFloat.from_float(2.0)
    assert "NumpyFloat" in repr(a)


def test_numpy_autograd_mul() -> None:
    a = Value.from_float(3.0, NumpyFloat)
    b = Value.from_float(4.0, NumpyFloat)
    c = a * b
    c.backward()
    assert abs(a.grad.get_float() - 4.0) < 1e-9
    assert abs(b.grad.get_float() - 3.0) < 1e-9


def test_numpy_tensor_matmul() -> None:
    data_a = [Value.from_float(float(i), NumpyFloat) for i in [1, 2, 3, 4]]
    data_b = [Value.from_float(float(i), NumpyFloat) for i in [1, 0, 0, 1]]
    a = Tensor(data_a, (2, 2))
    b = Tensor(data_b, (2, 2))
    c = a @ b
    assert c.shape == (2, 2)
    assert abs(c._data[0].get_float() - 1.0) < 1e-9


def test_backends_cannot_mix() -> None:
    """Проверяет что смешение бекендов вызывает ошибку"""
    a = Value.from_float(1.0, PyFloat)
    b = Value.from_float(1.0, DecimalNum)
    with pytest.raises(TypeError):
        _ = a + b  # type: ignore[operator]
