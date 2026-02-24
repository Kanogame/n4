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

    # Todo: add > 0 for relu?
    # Todo: add random gen? 

    def __add__(self, other: Self) -> Self: ...
    def __mul__(self, other: Self) -> Self: ...
    def __div__(self, other: Self) -> Self: ...
    def __pow__(self, other: Self) -> Self: ...
    