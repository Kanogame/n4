# Todo:
# Abstrac layer with support of convolution
# What to do: 
# Some typed (Numeric) layer with inps and outputs
# Then, neuron layer, with connetion matrix
# Create convolution layer with conv matrix
# Create dense layer of neurons
# In modal everything should be layer
from n4.tensor import Tensor
from n4.core import Op, Value
from typing import Optional
from abc import ABC, abstractmethod
from .nn_base import NnBase
from n4.core.numeric import NumericProtocol

class Layer[T: NumericProtocol](NnBase, ABC):
    """
    Базовый класс для любых слоев сети

    activation: Функция активации для всех нейронов в слое
    """

    def __init__(self, activation: Optional[type[Op[T]]] = None):
        super().__init__(self)

    @abstractmethod
    def forward(self, x: Tensor[T]) -> Tensor[T]:
        """Прямой проход слоя, должен быть реализован подклассами"""
        pass

    def __call__(self, x: Tensor[T]) -> Tensor[T]:
        return self.forward(x)

    @abstractmethod
    def parameters(self) -> list[Value[T]]:
        """Все параметры слоя, должен быть реализован подклассами"""
        pass