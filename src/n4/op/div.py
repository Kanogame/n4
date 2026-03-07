from typing import Self
from n4.numeric import NumericProtocol
from n4.core import Op, Value


class Div[T: NumericProtocol](Op[T]):
    def forward_pass(self: Self) -> list[Value[T]]:
        a, b = self.input_count(2)

        c = Value[T](a.data / b.data, parent_op=self)

        self.outputs = [c]

        return self.outputs

    def backward_pass(self: Self) -> None:
        out, *_ = self.output_count(1)
        a, b = self.input_count(2)

        a.grad += out.grad / b.data
        b.grad -= (a.data / (b.data * b.data)) * out.grad
