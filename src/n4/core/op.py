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
                raise TypeError("Cannot perform operation on different backends.")

    @abstractmethod
    def forward_pass(self: Self) -> list[Value[T]]:
        """прямой проход функции"""
        pass

    @abstractmethod
    def backward_pass(self: Self):
        """обратный проход функции"""
        pass

    @staticmethod
    def _count_args(args: list[Value[T]], desired_size: int):
        if len(args) != desired_size:
            raise TypeError("Received incorrent number of arguments")

        return args[:desired_size]

    def input_count(self: Self, desired_size: int) -> list[Value[T]]:
        """
        Проверяет количество переданных аргументов в функцию.

        Если кол-во не сответствует ожидаемому, значит функция вызвана неверно. В таком случае будет выброшена ошибка

        Сразу записывает значения функции в inputs
        """

        return self._count_args(self.inputs, desired_size)

    def output_count(self: Self, desired_size: int) -> list[Value[T]]:
        """
        Проверяет количество возрващенных значений функции.

        Если кол-во не сответствует ожидаемому, значит функция вызвана неверно. В таком случае будет выброшена ошибка

        Сразу записывает значения функции в inputs
        """

        return self._count_args(self.outputs, desired_size)
