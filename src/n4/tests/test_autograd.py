from n4.core import Value

# Addition
def test_simple_add_forward():
    """Проверяет корректность прямого прохода сложения"""

    a = Value(2)
    b = Value(3)

    c = a + b

    assert c.data == 5


def test_simple_add_backward():
    """Проверяет корректность градиентов при сложении"""

    a = Value(2)
    b = Value(3)

    c = a + b
    c.backward()

    assert a.grad == 1
    assert b.grad == 1

# Multiplication
def test_simple_mul_forward():
    """Проверяет прямой проход умножения"""

    a = Value(4)
    b = Value(5)

    c = a * b

    assert c.data == 20


def test_simple_mul_backward():
    """Проверяет корректность градиентов при умножении"""

    a = Value(4)
    b = Value(5)

    c = a * b
    c.backward()

    assert a.grad == 5
    assert b.grad == 4

## chain rule
def test_chain_rule_basic():
    """
    Проверяет корректность цепного правила:
    f = (a + b) * c
    """

    a = Value(2)
    b = Value(3)
    c = Value(4)

    f = (a + b) * c
    f.backward()

    # f = (2+3)*4 = 20
    # df/da = c = 4
    # df/db = c = 4
    # df/dc = a + b = 5

    assert a.grad == 4
    assert b.grad == 4
    assert c.grad == 5


def test_shared_node():
    """
    Проверяет корректность графа при повторном использовании узла:
    f = a * a
    """

    a = Value(3)
    f = a * a

    f.backward()

    # f = a^2
    # df/da = 2a = 6

    assert a.grad == 6

def test_deep_chain():
    """
    Проверяет глубокую композицию:
    f = (((a + b) * c) + d) * e
    """

    a = Value(1)
    b = Value(2)
    c = Value(3)
    d = Value(4)
    e = Value(5)

    f = (((a + b) * c) + d) * e
    f.backward()

    # проверка вручную
    # g = (a+b)*c = 9
    # h = g + d = 13
    # f = h * e = 65

    # df/da = c * e = 3*5 = 15
    assert a.grad == 15
    assert b.grad == 15
    assert c.grad == (a.data + b.data) * e.data
    assert d.grad == e.data
    assert e.grad == 13

def test_used_twice():
    """
    Проверяет использовании значения дважды:
    f = (x+y) + (x*y)
    """

    x = Value(5)
    y = Value(4)
    z = Value(8)

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

    assert x.grad == 9
    assert y.grad == 1
    assert z.grad == 5