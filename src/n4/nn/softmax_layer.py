from typing import Optional, Self
from n4.core import Value
from n4.tensor import Tensor
from n4.numeric import NumericProtocol
from .layer import Layer


class SoftmaxLayer[T: NumericProtocol](Layer[T]):
    """Softmax слой"""

    def forward_pass(self, x: Tensor[T]) -> Tensor[T]:
        """
        Применить слой для x

        Args:
            x: Тензор любой размерности. softmax подсчитан по последней размерности

        Returns:
            Тензор того же размера, со значениями [0,1] - выполняет св-во вероятностей
        """

        # x: (3, 5, 6)
        last_dim = x.shape[-1]  # 6
        num_rows = x.size // last_dim  # 15

        # x2d: (15, 6)
        # Производим упрощение размерности, оставляя нетронутым только последнее измерение
        x_2d = x.reshape((num_rows, last_dim))

        result_data = []
        for i in range(num_rows):
            # берем из тензора i строку - N
            row_data = x_2d[i]
            # row_data should be a Tensor view of the last dimension; use isinstance check
            if not isinstance(row_data, Tensor):
                raise ValueError("Size of tensor is incorrect")

            # Вычисляем экспоненты
            exps = [v.exp() for v in row_data._data]

            # Сумма экспонент
            sum_exp = Value.from_float(0, self._backend)
            for e in exps:
                sum_exp += e

            # Нормализуем, получая вероятности
            probs = [e / sum_exp for e in exps]
            result_data.extend(probs)

        # Восстанавливаем исходную форму
        return Tensor[T](result_data, x.shape)

    def parameters(self: Self) -> list[Value[T]]:
        return []

    def neuron_count(self: Self) -> Optional[int]:
        """
        Количество нейронов в слое
        """
        return None
