from abc import ABC, abstractmethod
from n4.core import Value
from n4.numeric import NumericProtocol
from typing import Optional, Self


class Optimizer[T: NumericProtocol](ABC):
    """
    Абстрактный класс оптимизатора

    Все подклассы дожны имплементировать step, используя
    class-wide ссылку на параметры - self.params
    """

    params: list[Value[T]]

    def __init__(self: Self, params: Optional[list[Value[T]]] = None) -> None:
        self.params = [] if params is None else list(params)

    @abstractmethod
    def step(self: Self) -> None: ...
