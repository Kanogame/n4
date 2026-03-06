from n4.tensor import Tensor
from n4.core import Op, Value
from typing import Optional, Self
from abc import ABC, abstractmethod
from .nn_base import NnBase
from n4.core.numeric import NumericProtocol


class Layer[T: NumericProtocol](NnBase[T], ABC):
    """
    Базовый класс для любых слоев сети

    activation: Функция активации для всех нейронов в слое
    """

    def __init__(self: Self, backend: type[T]):
        super().__init__(backend)

    @abstractmethod
    def forward_pass(self: Self, x: Tensor[T]) -> Tensor[T]:
        """Прямой проход слоя, должен быть реализован подклассами"""
        pass

    def __call__(self: Self, x: Tensor[T]) -> Tensor[T]:
        return self.forward_pass(x)

    @abstractmethod
    def parameters(self: Self) -> list[Value[T]]:
        """Все параметры слоя, должен быть реализован подклассами"""
        pass
