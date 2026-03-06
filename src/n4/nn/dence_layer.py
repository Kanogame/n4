from n4.core.numeric import NumericProtocol
from n4.tensor import Tensor
from n4.core import Op, Value
from .layer import Layer
from typing import Optional


# TODO: requires refactoring
class DenseLayer[T: NumericProtocol](Layer[T]):
    """
    Полносвязный слой

    Аргументы:
        in_features: кол-во входных параметров
        out_features: кол-во выходных параметров
        activation: функция артивации
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        backend: type[T],
        activation: Optional[type[Op[T]]] = None
    ):
        super().__init__(backend)

        self.in_features = in_features
        self.out_features = out_features

        # Матрица весов
        self.weights = Tensor.random_uniform(
            (out_features, in_features), backend=self._backend, low=-1.0, high=1.0
        )

        # Вектор весов
        self.bias = Tensor.zeros((out_features,), backend=self._backend)
        self.activation = self.resolve_activation(activation)

    def forward_pass(self, x: Tensor[T]) -> Tensor[T]:
        """
        Применить слой

        Входные параметры:
            x: Тензор формы (..., in_features). Последняя размерность должна соответствовать in_features.

        Выходные параметры:
            Тензор формы  (..., out_features). Все оставльные размерности остаются нетронутыми
        """

        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"Expected last dimension to be {self.in_features}, got {x.shape[-1]}"
            )

        # For simplicity, we only handle 2D input (batch, in_features) here.
        # A full implementation would support arbitrary leading dimensions.
        if x.ndim != 2:
            raise NotImplementedError(
                "DenseLayer currently only supports 2D input (batch, in_features)"
            )

        # x shape: (batch, in_features)
        # weights shape: (out_features, in_features)
        # Want: x @ weights.T  -> (batch, out_features)
        # Since we don't have batched matmul, we do it manually:
        batch_size = x.shape[0]
        out_data: list[Value[T]] = []
        for i in range(batch_size):
            # Take row i of x (1D tensor of shape (in_features,))
            row = Tensor[T](
                [x._data[i * self.in_features + j] for j in range(self.in_features)],
                (self.in_features,),
            )
            # Multiply by weights.T: row @ weights.T gives shape (out_features,)
            # We can compute each output element as dot(row, weights[:, k])
            out_row = []
            for k in range(self.out_features):
                # weights[k, :] is a 1D tensor of shape (in_features,)
                w_row = Tensor[T](
                    self.weights._data[
                        k * self.in_features : (k + 1) * self.in_features
                    ],
                    (self.in_features,),
                )
                # Dot product
                prod = row * w_row
                dot = prod.sum()
                out_row.append(dot + self.bias._data[k])
            out_data.extend(out_row)

            # For each element in out_data, apply activation
        activated: list[Value[T]] = [
            v.apply_activation(self.activation) for v in out_data
        ]

        return Tensor[T](activated, (batch_size, self.out_features))

    def parameters(self) -> list[Value[T]]:
        """Return weights and bias as a flat list of Value[T]."""
        return self.weights._data + self.bias._data
