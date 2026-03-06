from typing import Self
from random import uniform


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

    @classmethod
    def from_float(cls, f: float) -> Self:
        return cls(f)

    @classmethod
    def random_uniform(cls, start: float, end: float) -> Self:
        return cls(uniform(start, end))

    def __lt__(self, other: "PyFloat") -> bool:
        return self.v < other.v

    def __add__(self, other: "PyFloat") -> "PyFloat":
        return PyFloat(self.v + other.v)

    def __sub__(self, other: "PyFloat") -> "PyFloat":
        return PyFloat(self.v - other.v)

    def __mul__(self, other: "PyFloat") -> "PyFloat":
        return PyFloat(self.v * other.v)

    def __truediv__(self, other: "PyFloat") -> "PyFloat":
        return PyFloat(self.v / other.v)

    def __pow__(self, other: "PyFloat") -> "PyFloat":
        return PyFloat(self.v**other.v)

    def __neg__(self) -> "PyFloat":
        return PyFloat(-self.v)

    def __repr__(self) -> str:
        return f"PyFloat({self.v})"
