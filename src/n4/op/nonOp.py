from n4.numeric import NumericProtocol
from typing import Self
from n4.core import Op, Value


class NonOp[T: NumericProtocol](Op[T]):
    # Передаем входы как выходы
    def forward_pass(self: Self) -> list[Value[T]]:
        return self.inputs

    # Так как передали входы, градиенты пройдут напрямую
    def backward_pass(self: Self) -> None:
        return
