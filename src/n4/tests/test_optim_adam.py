import pytest
from n4.numeric import PyFloat
from n4.core import Value
from n4.optim import Adam


def _param(v: float) -> Value[PyFloat]:
    return Value.from_float(v, PyFloat)


def test_adam_single_step_updates_param() -> None:
    """Проверяет что Adam обновляет параметр за один шаг"""
    p = _param(1.0)
    p.grad = PyFloat.from_float(1.0)

    opt = Adam([p], lr=0.1)
    opt.step()

    assert p.data.v < 1.0


def test_adam_zero_grad_via_value() -> None:
    """Проверяет обнуление градиентов через Value.zero_grad"""
    p = _param(1.0)
    p.grad = PyFloat.from_float(3.0)
    p.zero_grad()
    assert p.grad.v == 0.0


def test_adam_converges_simple() -> None:
    """Проверяет что Adam минимизирует простую квадратичную функцию"""
    p = _param(5.0)
    opt = Adam([p], lr=0.1)

    for _ in range(200):
        p.grad = p.data
        opt.step()

    assert abs(p.data.v) < 0.5


def test_adam_multiple_params() -> None:
    """Проверяет обновление нескольких параметров"""
    a = _param(2.0)
    b = _param(-3.0)
    a.grad = PyFloat.from_float(1.0)
    b.grad = PyFloat.from_float(-1.0)

    opt = Adam([a, b], lr=0.01)
    opt.step()

    assert a.data.v < 2.0
    assert b.data.v > -3.0


def test_adam_requires_params() -> None:
    """Проверяет что Adam без параметров выбрасывает ошибку"""
    with pytest.raises(ValueError):
        Adam([], lr=0.01)


def test_adam_backend_mismatch_raises() -> None:
    """Проверяет что смешивание бекендов вызывает ошибку"""
    from n4.numeric import NumpyFloat
    a = _param(1.0)
    b = Value.from_float(1.0, NumpyFloat)
    with pytest.raises(ValueError):
        Adam([a, b])  # type: ignore[arg-type]


def test_adam_custom_hyperparams() -> None:
    """Проверяет что кастомные гиперпараметры принимаются"""
    p = _param(1.0)
    p.grad = PyFloat.from_float(0.5)
    opt = Adam([p], lr=0.01, beta1=0.8, beta2=0.99, eps=1e-6)
    opt.step()
    assert p.data.v < 1.0
