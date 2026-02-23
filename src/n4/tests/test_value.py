from n4.core import Value

def test_value_initialization():
    """Проверяет корректную инициализацию Value"""

    x = Value.from_int(5)

    assert x.data.v == 5.0
    assert x.grad.v == 0.0
