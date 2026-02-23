from typing import Self

class PyFloat:
    """
    Реализация NumericProtocol для python float.
    """

    def __init__(self, v: float | int):
        self.v: float = float(v)

    @classmethod
    def zero(cls) -> Self:
        return cls(0.0)

    @classmethod
    def one(cls) -> Self:
        return cls(1.0)

    def __add__(self, other: "PyFloat") -> "PyFloat":
        return PyFloat(self.v + other.v)

    def __mul__(self, other: "PyFloat") -> "PyFloat":
        return PyFloat(self.v * other.v)

    def __repr__(self) -> str:
        return f"PyFloat({self.v})"