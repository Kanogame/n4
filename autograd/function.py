from value import Value
from typing import Protocol


class Function[T](Protocol):
    """
    Простой интерфейс который для любой функции, которая поддерживает прямой и обратный проход
    """

    inputs: list[Value[T]]
    outputs: list[Value[T]]

    def forward_pass(*args: list[Value[T]]) -> Value[T]:
        """прямой проход функции"""
        pass

    def backward_pass():
        """обратный проход функции"""
        pass


    def arg_count(self, args: list[Value[T]], desired_size: int):
        """
        Проверяет количество переданных аргументов в функцию. Если кол-во не сответствует ожидаемому, значит функция вызвана неверно.

        Сразу записывает значения функции в inputs
        """
        if len(args) != desired_size:
            raise TypeError
        
        self.inputs = args