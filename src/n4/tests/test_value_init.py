from .helpers import new_value


def test_value_initialization() -> None:

    x = new_value(5)

    assert x.data.v == 5.0
    assert x.grad.v == 0.0
