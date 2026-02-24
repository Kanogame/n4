from abc import ABC, abstractmethod
from typing import Self
from n4.core import Value
from n4.core.numeric import NumericProtocol

class NnBase[T: NumericProtocol](ABC):
    """
    Базовый класс для всех элементов нейросети.
    Класс задан абстрактным, так как не имеет смысла сам по себе

    Позволяет обнулять все градиенты
    """

    def zero_grad(self: Self):
        for v in self.parameters():
            v.zero_grad()
        
    @abstractmethod
    def parameters(self: Self) -> Value[T]: ...