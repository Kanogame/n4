import math
from typing import Sequence, Tuple

from n4.numeric import PyFloat
from n4.tensor import Tensor
from n4.core import Value
from n4.loss import MSELoss, CrossEntropyLoss


def make_tensor(vals: Sequence[float], shape: Tuple[int, ...]) -> Tensor[PyFloat]:
    data = [Value.from_float(v, PyFloat) for v in vals]
    return Tensor(data, shape)


def test_mse_loss() -> None:
    pred = make_tensor([1.0, 2.0], (2, 1))
    target = make_tensor([0.0, 0.0], (2, 1))

    loss: Value[PyFloat] = MSELoss[PyFloat]()(pred, target)
    assert abs(loss.data.v - 2.5) < 1e-6


def test_cross_entropy_loss() -> None:
    pred = make_tensor([0.7, 0.2, 0.1, 0.1, 0.8, 0.1], (2, 3))
    target = make_tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], (2, 3))

    loss: Value[PyFloat] = CrossEntropyLoss[PyFloat]()(pred, target)

    expected = (-math.log(0.7) + -math.log(0.8)) / 2.0
    assert abs(loss.data.v - expected) < 1e-6
