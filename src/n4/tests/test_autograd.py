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

