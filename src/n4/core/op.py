from __future__ import annotations
from n4.core.numeric import NumericProtocol
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Self

if TYPE_CHECKING:
    from .value import Value

class Op[T: NumericProtocol](ABC):
    """
    Простой абстрактный класс для любой функции, которая поддерживает прямой и обратный проход

    Может отражать любую функцию, с любым кол-вом входов и выходов. 
    
    Никак не связан с operation Overload, это производит Value/NumericProtocol. Все вычисления не покидают backend
    """

    inputs: list[Value[T]]
    outputs: list[Value[T]]

    def __init__(self: Self, inps: list[Value[T]]):
        """
        Инициализация класса.
        
        inps: значения входных переменных
            Все входные переменные должны иметь одинаковый бекенд и тип (в том числе запрет неявного приведения)
        """
        if not inps:
            raise ValueError("Op requires at least one input.")

        self._ensure_same_backend(inps)

        self.inputs = inps
        self.outputs = []

    @staticmethod
    def _ensure_same_backend(values: list[Value[T]]) -> None:
        """
        Проверяет, что все Value имеют одинаковый backend.
        """

        first_backend = type(values[0].data)

        for v in values:
            if v.get_backend() is not first_backend:
                raise TypeError(
                    "Cannot perform operation on different backends."
                )

    @abstractmethod
    def forward_pass(self: Self) -> Value[T]:
        """прямой проход функции"""
        pass

    @abstractmethod
    def backward_pass(self: Self):
        """обратный проход функции"""
        pass


    def arg_count(self: Self, args: list[Value[T]], desired_size: int):
        """
        Проверяет количество переданных аргументов в функцию. Если кол-во не сответствует ожидаемому, значит функция вызвана неверно.

        Сразу записывает значения функции в inputs
        """
        if len(args) != desired_size:
            raise TypeError
        
        self.inputs = args