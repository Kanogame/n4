from typing import Protocol, runtime_checkable, Self


@runtime_checkable
class NumericProtocol(Protocol):
    """
    Интерфейс для числовых типов, используемых в Value.
    """

    # Из числа с плавующей точкой
    @classmethod
    def from_float(cls, f: float) -> Self: ...

    # Рандом подчиняющий равномерному распределению
    @classmethod
    def random_uniform(cls, start: float, end: float) -> Self: ...

    # Перегрузки операторов

    # <
    def __lt__(self: Self, other: Self) -> bool: ...

    # +
    def __add__(self: Self, other: Self) -> Self: ...

    # -
    def __sub__(self: Self, other: Self) -> Self: ...

    # *
    def __mul__(self: Self, other: Self) -> Self: ...

    # /
    def __truediv__(self: Self, other: Self) -> Self: ...

    # **
    def __pow__(self: Self, other: Self) -> Self: ...

    # -a
    def __neg__(self) -> Self: ...

    # e^a
    def exp(self) -> Self: ...
