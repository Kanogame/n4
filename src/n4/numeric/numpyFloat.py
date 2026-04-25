import numpy as np
from random import uniform
from typing import Self


class NumpyFloat:
    """
    Реализация NumericProtocol на основе numpy.float64.
    Позволяет использовать аппаратное ускорение numpy для скалярных операций.
    """

    def __init__(self, v: float | int | np.float64) -> None:
        self.v: np.float64 = np.float64(v)

    @classmethod
    def from_float(cls, f: float) -> Self:
        return cls(f)

    @classmethod
    def random_uniform(cls, start: float, end: float) -> Self:
        return cls(np.float64(uniform(start, end)))

    def get_float(self) -> float:
        return float(self.v)

    def __lt__(self, other: "NumpyFloat") -> bool:
        return bool(self.v < other.v)

    def __add__(self, other: "NumpyFloat") -> "NumpyFloat":
        return NumpyFloat(self.v + other.v)

    def __sub__(self, other: "NumpyFloat") -> "NumpyFloat":
        return NumpyFloat(self.v - other.v)

    def __mul__(self, other: "NumpyFloat") -> "NumpyFloat":
        return NumpyFloat(self.v * other.v)

    def __truediv__(self, other: "NumpyFloat") -> "NumpyFloat":
        return NumpyFloat(self.v / other.v)

    def __pow__(self, other: "NumpyFloat") -> "NumpyFloat":
        return NumpyFloat(self.v ** other.v)

    def __neg__(self) -> "NumpyFloat":
        return NumpyFloat(-self.v)

    def exp(self) -> "NumpyFloat":
        return NumpyFloat(np.exp(self.v))

    def tanh(self) -> "NumpyFloat":
        return NumpyFloat(np.tanh(self.v))

    def log(self) -> "NumpyFloat":
        return NumpyFloat(np.log(self.v))

    def sqrt(self) -> "NumpyFloat":
        return NumpyFloat(np.sqrt(self.v))

    def __repr__(self) -> str:
        return f"NumpyFloat({self.v})"
