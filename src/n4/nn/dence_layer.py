from n4.numeric import NumericProtocol
from n4.tensor import Tensor
from n4.core import Op, Value
from .layer import Layer
from typing import Optional


class DenseLayer[T: NumericProtocol](Layer[T]):
    """Полносвязный слой нейронов, использует одну функцию активации"""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        backend: type[T],
        activation: Optional[type[Op[T]]] = None,
    ):
        """
        Для функции вида R^n -> R^m

        Аргументы:
            in_features: кол-во входных параметров (R^n) - кол-во входов каждого нейрона
            out_features: кол-во выходных параметров (R^m) - кол-во нейронов
            activation: функция артивации
        """

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
            x: Тензор формы (batch, in_features). Последняя размерность должна соответствовать in_features.

        Выходные параметры:
            Тензор формы  (batch, out_features). Batch не меняется
        """

        # Проверки размеров
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"Expected last dimension to be {self.in_features}, got {x.shape[-1]}"
            )

        if x.ndim != 2:
            raise NotImplementedError(
                "DenseLayer only supports 2D input (batch, in_features)"
            )

        batch_size = x.shape[0]
        out_data: list[Value[T]] = []

        for i in range(batch_size):
            # 1‑D view of the i‑th input row
            row = Tensor[T](
                [x._data[i * self.in_features + j] for j in range(self.in_features)],
                (
                    1,
                    self.in_features,
                ),
            )

            out_row = row @ self.weights.Transposed
            out_row = out_row + self.bias

            out_data.extend(out_row._data)

        return Tensor[T](out_data, (batch_size, self.out_features)).apply_activation(
            self.activation
        )

    def parameters(self) -> list[Value[T]]:
        return self.weights._data + self.bias._data
