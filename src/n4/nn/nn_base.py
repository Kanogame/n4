from n4.op import NonOp
from abc import ABC, abstractmethod
from typing import Self, Optional
from n4.core import Value, Op
from n4.core.numeric import NumericProtocol


class NnBase[T: NumericProtocol](ABC):
    """
    Базовый класс для всех элементов нейросети.
    Класс задан абстрактным, так как не имеет смысла сам по себе

    Позволяет обнулять все градиенты
    """

    # Бекенд вычислений
    _backend: type[T]

    def __init__(self: Self, backend: type[T]):
        self._backend = backend

    def zero_grad(self: Self) -> None:
        for v in self.parameters():
            v.zero_grad()

    @abstractmethod
    def parameters(self: Self) -> list[Value[T]]: ...

    @staticmethod
    def resolve_activation(activation: Optional[type[Op[T]]]) -> type[Op[T]]:
        return NonOp if activation is None else activation
