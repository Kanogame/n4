from n4.loss import Loss
from n4.tensor import Tensor
from n4.core import Value
from n4.numeric import NumericProtocol


class MSELoss[T: NumericProtocol](Loss[T]):
    """
    Среднее квадратное отклонение: mean((pred - target)^2)"""

    def __call__(self, pred: Tensor[T], target: Tensor[T]) -> Value[T]:
        if pred.shape != target.shape:
            raise ValueError("pred and target must have the same shape")

        diff = pred - target
        sq = diff * diff
        return sq.mean()
