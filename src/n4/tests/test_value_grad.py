from n4.core import Value
from .helpers import new_value
# uv pip install -e .


# Addition
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


# Multiplication
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


## chain rule
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

    # f = (2+3)*4 = 20
    # df/da = c = 4
    # df/db = c = 4
    # df/dc = a + b = 5

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

    # f = a^2
    # df/da = 2a = 6

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

    # проверка вручную
    # g = (a+b)*c = 9
    # h = g + d = 13
    # f = h * e = 65

    # df/da = c * e = 3*5 = 15
    assert a.grad.v == 15
    assert b.grad.v == 15
    assert c.grad.v == (a.data.v + b.data.v) * e.data.v
    assert d.grad.v == e.data.v
    assert e.grad.v == 13


def test_used_twice() -> None:
    """
    Проверяет использовании значения дважды:
    f = (x+y) + (x*y)
    """

    x = new_value(5)
    y = new_value(4)
    z = new_value(8)

    z1 = x + y
    z2 = x * z
    res = z1 + z2

    res.backward()

    # f = (5 + 4) * (5 * 8) = 9 * 40 = 360
    # df / dres = 1
    # df / dz1 = 1
    # df / dz2 = 1
    # df / dx = df / dz1 * dz1 / dx + df / dz2 * dz2 / dx = 1 + 8 = 9
    # df / dy = df / dz1 * dz1 / dy = 1
    # df / dz = df / dz2 = 5

    assert x.grad.v == 9
    assert y.grad.v == 1
    assert z.grad.v == 5
