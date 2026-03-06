from typing import Self
from n4.tensor import Tensor
from abc import ABC, abstractmethod
from n4.core.numeric import NumericProtocol


class Model[T: NumericProtocol](ABC):
    """
    Базовый класс для всех моделей построенных на n4

    Применяется аналогично pytorch:

        ```
        import n4.nn as nn
        from n4.op import Relu, NonOp
        from n4.core.numeric import PyFloat
        from typing import Self


        class MyModel[T: PyFloat](nn.Model[T]):
            def __init__(self: Self) -> None:
                super().__init__()
                self.model = nn.Sequential(
                    nn.DenceLayer(5, 10, Relu),
                    nn.DenceLayer(10, 3, NonOp),
                    nn.SoftmaxLayer()
                )

            def forward(self, x):
                return self.model(x)
        ```
    """

    # Бекенд вычислений
    _backend: type[T]
    result: Tensor[T]

    def __init__(self: Self):
        self._backend = type(T)

    @abstractmethod
    def forward_pass(self: Self, x: Tensor[T]) -> Tensor[T]: ...
