import math
from typing import Self
from random import uniform


class PyFloat:
    """
    Реализация NumericProtocol для python float.
    """

    def __init__(self, v: float | int):
        self.v: float = float(v)

    @classmethod
    def from_float(cls, f: float) -> Self:
        return cls(f)

    @classmethod
    def random_uniform(cls, start: float, end: float) -> Self:
        return cls(uniform(start, end))

    def get_float(self: Self) -> float:
        return self.v

    def __lt__(self, other: Self) -> bool:
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

    def exp(self) -> "PyFloat":
        return PyFloat(math.exp(self.v))

    def tanh(self) -> "PyFloat":
        return PyFloat(math.tanh(self.v))

    def log(self) -> "PyFloat":
        return PyFloat(math.log(self.v))

    def sqrt(self) -> "PyFloat":
        return PyFloat(math.sqrt(self.v))

    def __repr__(self) -> str:
        return f"PyFloat({self.v})"
