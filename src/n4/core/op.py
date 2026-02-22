from __future__ import annotations
from typing import TYPE_CHECKING, Self, Protocol

if TYPE_CHECKING:
    from .value import Value

class Op[T](Protocol):
    """
    Простой интерфейс который для любой функции, которая поддерживает прямой и обратный проход
    """

    inputs: list[Value[T]]
    outputs: list[Value[T]]

    def __init__(self: Self, inps: list[Value[T]]):
        self.inputs = inps
        self.outputs = []

    def forward_pass(self: Self) -> Value[T]:
        """прямой проход функции"""
        pass

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