from n4.loss import Loss
from n4.tensor import Tensor
from n4.core import Value
from n4.numeric import NumericProtocol
from n4.op import Log


class CrossEntropyLoss[T: NumericProtocol](Loss[T]):
    """
    Cross-entropy для вероятносного распредения (Softmax).

    pred - результат Softmax
    target - one-hot тензор

    """

    def __call__(self, pred: Tensor[T], target: Tensor[T]) -> Value[T]:
        if pred.shape != target.shape:
            raise ValueError(
                f"pred and target must have the same shape, pred: {pred.shape}, target: {target.shape}"
            )

        log_vals = Tensor([v.apply_activation(Log) for v in pred._data], pred.shape)
        neg_ll = -(target * log_vals)

        # Sum over last dimension (classes), then mean over remaining dimensions
        # Индекс последнего измерения
        last_dim_idx = neg_ll.ndim - 1

        # Сумма по последнему измерению (только по элеметам в одного ввода)
        summed = neg_ll.sum_dim(last_dim_idx)
        return summed.mean()  # Среднее по всем изменениям, т.е. среднее по батчу
