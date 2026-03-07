from n4.tensor.tensor import Tensor
from .helpers import new_value


def test_grad_through_sum() -> None:
    t = Tensor([new_value(1), new_value(2)], (2,))

    s = t.sum()
    s.backward()

    for v in t._data:
        assert v.grad.v == 1


def test_grad_through_matmul() -> None:
    a = Tensor(
        [new_value(1), new_value(2), new_value(3), new_value(4)],
        (2, 2),
    )

    b = Tensor(
        [new_value(5), new_value(6), new_value(7), new_value(8)],
        (2, 2),
    )

    c = a @ b
    s = c.sum()

    s.backward()

    # Проверяем, что градиенты ненулевые
    for v in a._data:
        assert v.grad.v != 0

    for v in b._data:
        assert v.grad.v != 0
