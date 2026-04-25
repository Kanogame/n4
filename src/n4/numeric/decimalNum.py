import math
from decimal import Decimal, InvalidOperation
from random import uniform
from typing import Self


class DecimalNum:
    """
    Реализация NumericProtocol на основе decimal.Decimal.
    Обеспечивает произвольную точность вычислений.
    """

    def __init__(self, v: float | int | str | Decimal) -> None:
        self.v: Decimal = Decimal(str(v)) if not isinstance(v, Decimal) else v

    @classmethod
    def from_float(cls, f: float) -> Self:
        return cls(f)

    @classmethod
    def random_uniform(cls, start: float, end: float) -> Self:
        return cls(uniform(start, end))

    def get_float(self) -> float:
        return float(self.v)

    def __lt__(self, other: "DecimalNum") -> bool:
        return self.v < other.v

    def __add__(self, other: "DecimalNum") -> "DecimalNum":
        return DecimalNum(self.v + other.v)

    def __sub__(self, other: "DecimalNum") -> "DecimalNum":
        return DecimalNum(self.v - other.v)

    def __mul__(self, other: "DecimalNum") -> "DecimalNum":
        return DecimalNum(self.v * other.v)

    def __truediv__(self, other: "DecimalNum") -> "DecimalNum":
        return DecimalNum(self.v / other.v)

    def __pow__(self, other: "DecimalNum") -> "DecimalNum":
        try:
            return DecimalNum(self.v ** other.v)
        except InvalidOperation:
            return DecimalNum(Decimal("NaN"))

    def __neg__(self) -> "DecimalNum":
        return DecimalNum(-self.v)

    def exp(self) -> "DecimalNum":
        return DecimalNum(Decimal(str(math.exp(float(self.v)))))

    def tanh(self) -> "DecimalNum":
        return DecimalNum(Decimal(str(math.tanh(float(self.v)))))

    def log(self) -> "DecimalNum":
        try:
            return DecimalNum(Decimal(str(math.log(float(self.v)))))
        except (ValueError, OverflowError):
            return DecimalNum(Decimal("NaN"))

    def sqrt(self) -> "DecimalNum":
        return DecimalNum(self.v.sqrt())

    def __repr__(self) -> str:
        return f"DecimalNum({self.v})"
