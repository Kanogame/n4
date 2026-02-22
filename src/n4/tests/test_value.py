from n4.core import Value

def test_value_initialization():
    """Проверяет корректную инициализацию Value"""

    x = Value(5)

    assert x.data == 5
    assert x.grad == 0
