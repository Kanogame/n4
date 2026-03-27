from typing import Sequence, Tuple, cast
import math

from n4.numeric import PyFloat
from n4.tensor import Tensor
from n4.core import Value
from n4.nn.loss import MSELoss, CrossEntropyLoss


def make_tensor(vals: Sequence[float], shape: Tuple[int, ...]) -> Tensor[PyFloat]:
    data = [Value.from_float(v, PyFloat) for v in vals]
    return Tensor(data, shape)


def test_mse_loss() -> None:
    pred = make_tensor([1.0, 2.0], (2, 1))
    target = make_tensor([0.0, 0.0], (2, 1))

    loss = MSELoss()(pred, target)
    # mean of [1^2, 2^2] = (1 + 4)/2 = 2.5
    lv = cast(PyFloat, loss.data).v
    assert abs(lv - 2.5) < 1e-6


def test_cross_entropy_loss() -> None:
    # two samples, three classes
    # probabilities and one-hot targets
    pred = make_tensor([0.7, 0.2, 0.1, 0.1, 0.8, 0.1], (2, 3))
    # targets: first class 0, second class 1
    target = make_tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], (2, 3))

    loss = CrossEntropyLoss()(pred, target)

    # expected = mean(-log(0.7), -log(0.8))
    expected = (-math.log(0.7) + -math.log(0.8)) / 2.0
    lv = cast(PyFloat, loss.data).v
    assert abs(lv - expected) < 1e-6
