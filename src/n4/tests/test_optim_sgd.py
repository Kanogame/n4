from .helpers import new_value
from n4.optim import SGD
from n4.numeric import PyFloat


def test_sgd_updates_single_value() -> None:
    """Проверяет, что SGD корректно обновляет одиночный параметр"""

    v = new_value(1.0)

    # Задаем искусственный градиент
    v.grad = PyFloat.from_float(0.5)

    opt = SGD([v], lr=0.1)
    opt.step()

    # Ожидаем: 1.0 - 0.1 * 0.5 = 0.95
    assert isinstance(v.data, PyFloat)
    assert v.data.v == 0.95


def test_sgd_zero_grad() -> None:
    """Проверяет, что zero_grad обнуляет градиент"""

    v = new_value(2.0)
    v.grad = PyFloat.from_float(2.0)

    opt = SGD([v], lr=0.5)
    opt.zero_grad()

    assert v.grad.v == 0.0


def test_sgd_updates_multiple_params() -> None:
    """Проверяет обновление нескольких параметров за один шаг"""

    a = new_value(3.0)
    b = new_value(-1.0)

    a.grad = PyFloat.from_float(1.0)
    b.grad = PyFloat.from_float(2.0)

    opt = SGD([a, b], lr=0.2)
    opt.step()

    # a: 3.0 - 0.2*1.0 = 2.8
    # b: -1.0 - 0.2*2.0 = -1.4
    assert a.data.v == 2.8
    assert b.data.v == -1.4
