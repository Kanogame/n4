from typing import Protocol, runtime_checkable, Self

@runtime_checkable
class NumericProtocol(Protocol):
    """
    Интерфейс для числовых типов, используемых в Value.
    """

    # Ноль
    @classmethod
    def zero(cls) -> Self: ...

    # Единица
    @classmethod
    def one(cls) -> Self: ...

    # Рандом подчиняющий равномерному распределению 
    # TODO: start, end typing
    @classmethod
    def random_uniform(cls, start, end) -> Self: ...

    # Перегрузки операторов

    # <
    def __lt__(self: Self, other: Self) -> bool: ...

    # +
    def __add__(self: Self, other: Self) -> Self: ...
    
    # *
    def __mul__(self: Self, other: Self) -> Self: ...
    
    # **
    def __pow__(self: Self, other: Self) -> Self: ...

    # -a
    def __neg__(self) -> Self: ...
    