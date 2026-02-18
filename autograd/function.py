from value import Value
from typing import Protocol


class Function:
    """
    Простой интерфейс который для любой функции, которая поддерживает прямой и обратный проход
    """

    def forward_pass(*args: list[Value]) -> Value:
        """прямой проход функции"""
        pass

    def backward_pass():
        """обратный проход функции"""
        pass

    @staticmethod
    def arg_count(args: list[Value], desired_size: int) -> list[Value]:
        """Проверяет количество переданных аргументов в функцию. Если кол-во не сответствует ожидаемому, значит функция вызвана неверно"""
        if len(args) != desired_size:
            raise TypeError
        
        return args