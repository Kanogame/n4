from n4.tensor.tensor import Tensor
from n4.core import Value


def test_grad_through_sum() -> None:
    t = Tensor([Value.from_int(1), Value.from_int(2)], (2,))

    s = t.sum()
    s.backward()

    for v in t._data:
        assert v.grad.v == 1


def test_grad_through_matmul() -> None:
    a = Tensor(
        [Value.from_int(1), Value.from_int(2), Value.from_int(3), Value.from_int(4)],
        (2, 2),
    )

    b = Tensor(
        [Value.from_int(5), Value.from_int(6), Value.from_int(7), Value.from_int(8)],
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
