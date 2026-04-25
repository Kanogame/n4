import math
from .helpers import new_value
from n4.op import Tanh, Log


def test_simple_add_forward() -> None:
    """Проверяет корректность прямого прохода сложения"""
    a = new_value(2)
    b = new_value(3)
    c = a + b
    assert c.data.v == 5


def test_simple_add_backward() -> None:
    """Проверяет корректность градиентов при сложении"""
    a = new_value(2)
    b = new_value(3)
    c = a + b
    c.backward()
    assert c.grad.v == 1.0
    assert c.parent_op is not None
    assert a.grad.v == 1.0
    assert b.grad.v == 1.0


def test_simple_mul_forward() -> None:
    """Проверяет прямой проход умножения"""
    a = new_value(4)
    b = new_value(5)
    c = a * b
    assert c.data.v == 20


def test_simple_mul_backward() -> None:
    """Проверяет корректность градиентов при умножении"""
    a = new_value(4)
    b = new_value(5)
    c = a * b
    c.backward()
    assert a.grad.v == 5
    assert b.grad.v == 4


def test_sub_forward() -> None:
    """Проверяет прямой проход вычитания"""
    a = new_value(7)
    b = new_value(3)
    c = a - b
    assert c.data.v == 4.0


def test_sub_backward() -> None:
    """Проверяет корректность градиентов при вычитании"""
    a = new_value(7)
    b = new_value(3)
    c = a - b
    c.backward()
    assert a.grad.v == 1.0
    assert b.grad.v == -1.0


def test_div_forward() -> None:
    """Проверяет прямой проход деления"""
    a = new_value(6)
    b = new_value(3)
    c = a / b
    assert c.data.v == 2.0


def test_div_backward() -> None:
    """Проверяет корректность градиентов при делении"""
    a = new_value(6)
    b = new_value(2)
    c = a / b
    c.backward()
    assert abs(a.grad.v - 0.5) < 1e-9
    assert abs(b.grad.v - (-1.5)) < 1e-9


def test_pow_forward() -> None:
    """Проверяет прямой проход возведения в степень"""
    a = new_value(3)
    b = new_value(2)
    c = a ** b
    assert c.data.v == 9.0


def test_pow_backward() -> None:
    """Проверяет корректность градиентов при возведении в степень"""
    a = new_value(3)
    b = new_value(2)
    c = a ** b
    c.backward()
    assert abs(a.grad.v - 6.0) < 1e-9
    assert abs(b.grad.v - (9.0 * math.log(3.0))) < 1e-9


def test_neg_forward() -> None:
    """Проверяет прямой проход унарного минуса"""
    a = new_value(5)
    b = -a
    assert b.data.v == -5.0


def test_neg_backward() -> None:
    """Проверяет корректность градиентов унарного минуса"""
    a = new_value(5)
    b = -a
    b.backward()
    assert a.grad.v == -1.0


def test_exp_forward() -> None:
    """Проверяет прямой проход экспоненты"""
    a = new_value(1.0)
    b = a.exp()
    assert abs(b.data.v - math.e) < 1e-9


def test_exp_backward() -> None:
    """Проверяет корректность градиентов экспоненты"""
    a = new_value(0.0)
    b = a.exp()
    b.backward()
    assert abs(a.grad.v - 1.0) < 1e-9


def test_relu_positive() -> None:
    """Проверяет relu для положительного входа"""
    a = new_value(3.0)
    b = a.relu()
    assert b.data.v == 3.0


def test_relu_negative() -> None:
    """Проверяет relu для отрицательного входа"""
    a = new_value(-2.0)
    b = a.relu()
    assert b.data.v == 0.0


def test_relu_backward_positive() -> None:
    """Проверяет градиент relu при положительном входе"""
    a = new_value(2.0)
    b = a.relu()
    b.backward()
    assert a.grad.v == 1.0


def test_relu_backward_negative() -> None:
    """Проверяет что relu блокирует градиент при отрицательном входе"""
    a = new_value(-2.0)
    b = a.relu()
    b.backward()
    assert a.grad.v == 0.0


def test_tanh_forward() -> None:
    """Проверяет прямой проход tanh"""
    a = new_value(0.0)
    b = a.apply_activation(Tanh)
    assert abs(b.data.v) < 1e-9


def test_tanh_backward() -> None:
    """Проверяет корректность градиентов tanh"""
    a = new_value(0.0)
    b = a.apply_activation(Tanh)
    b.backward()
    assert abs(a.grad.v - 1.0) < 1e-9


def test_log_forward() -> None:
    """Проверяет прямой проход логарифма"""
    a = new_value(1.0)
    b = a.apply_activation(Log)
    assert abs(b.data.v) < 1e-6


def test_log_backward() -> None:
    """Проверяет корректность градиентов логарифма"""
    a = new_value(1.0)
    b = a.apply_activation(Log)
    b.backward()
    assert abs(a.grad.v - 1.0) < 1e-6


def test_chain_rule_basic() -> None:
    """
    Проверяет корректность цепного правила:
    f = (a + b) * c
    """
    a = new_value(2)
    b = new_value(3)
    c = new_value(4)
    f = (a + b) * c
    f.backward()
    assert a.grad.v == 4
    assert b.grad.v == 4
    assert c.grad.v == 5


def test_shared_node() -> None:
    """
    Проверяет корректность графа при повторном использовании узла:
    f = a * a
    """
    a = new_value(3)
    f = a * a
    f.backward()
    assert a.grad.v == 6


def test_deep_chain() -> None:
    """
    Проверяет глубокую композицию:
    f = (((a + b) * c) + d) * e
    """
    a = new_value(1)
    b = new_value(2)
    c = new_value(3)
    d = new_value(4)
    e = new_value(5)

    f = (((a + b) * c) + d) * e
    f.backward()

    assert a.grad.v == 15
    assert b.grad.v == 15
    assert c.grad.v == (a.data.v + b.data.v) * e.data.v
    assert d.grad.v == e.data.v
    assert e.grad.v == 13


def test_used_twice() -> None:
    """
    Проверяет использование значения дважды:
    f = (x+y) + (x*z)
    """
    x = new_value(5)
    y = new_value(4)
    z = new_value(8)

    z1 = x + y
    z2 = x * z
    res = z1 + z2

    res.backward()

    assert x.grad.v == 9
    assert y.grad.v == 1
    assert z.grad.v == 5
