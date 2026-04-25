from typing import Self, cast
from n4.numeric import NumericProtocol
from n4.core import Op, Value


class Neg[T: NumericProtocol](Op[T]):
    def forward_pass(self: Self) -> list[Value[T]]:
        a, *_ = self.input_count(1)

        b = Value[T](cast(T, -a.data), parent_op=self)

        self.outputs = [b]

        return self.outputs

    def backward_pass(self: Self) -> None:
        out, *_ = self.output_count(1)
        a, *_ = self.input_count(1)

        a.grad -= out.grad
