from typing import Self
from abc import ABC, abstractmethod

from n4.tensor import Tensor
from n4.numeric import NumericProtocol
from n4.nn.nn_base import NnBase


class Model[T: NumericProtocol](NnBase, ABC):
    """
    Базовый класс для всех моделей построенных на n4

    Применяется аналогично pytorch:

        ```
        import n4.nn as nn
        from n4.op import Relu, NonOp
        from n4.numeric import PyFloat
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

    @abstractmethod
    def forward_pass(self: Self, x: Tensor[T]) -> Tensor[T]: ...
