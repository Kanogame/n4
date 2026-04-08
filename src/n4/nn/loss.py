from abc import ABC, abstractmethod
from typing import Self, TypeVar
from n4.tensor import Tensor
from n4.core import Value
from n4.op import Log
from n4.numeric import NumericProtocol


T = TypeVar("T", bound=NumericProtocol)


class Loss(ABC):
    """Abstract loss interface: call with (pred, target) and return scalar Value"""

    @abstractmethod
    def __call__(self: Self, pred: Tensor[T], target: Tensor[T]) -> Value[T]: ...


class MSELoss(Loss):
    """Mean squared error: mean((pred - target)^2)"""

    def __call__(self, pred: Tensor[T], target: Tensor[T]) -> Value[T]:
        if pred.shape != target.shape:
            raise ValueError("pred and target must have the same shape")

        diff = pred - target
        sq = diff * diff
        return sq.mean()


class CrossEntropyLoss(Loss):
    """
    Cross-entropy for probability distributions. Expects `pred` to contain probabilities
    (e.g. output of SoftmaxLayer) and `target` to be a one-hot Tensor of the same shape.
    Returns a scalar Value (mean negative log-likelihood).
    """

    def __call__(self, pred: Tensor[T], target: Tensor[T]) -> Value[T]:
        if pred.shape != target.shape:
            raise ValueError(
                f"pred and target must have the same shape, pred: {pred.shape}, target: {target.shape}"
            )

        log_vals = Tensor([v.apply_activation(Log) for v in pred._data], pred.shape)
        neg_ll = -(target * log_vals)

        # Sum over last dimension (classes), then mean over remaining dimensions
        last_dim_idx = neg_ll.ndim - 1
        summed = neg_ll.sum_dim(last_dim_idx)
        return summed.mean()  # Works for both 1D and batched inputs
