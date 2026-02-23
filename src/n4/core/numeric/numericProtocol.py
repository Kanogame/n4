from typing import Protocol, runtime_checkable, Self

@runtime_checkable
class NumericProtocol(Protocol):
    """
    Интерфейс для числовых типов, используемых в Value.
    """

    @classmethod
    def zero(cls) -> Self: ...
    
    @classmethod
    def one(cls) -> Self: ...

    def __add__(self, other: Self) -> Self: ...
    def __mul__(self, other: Self) -> Self: ...