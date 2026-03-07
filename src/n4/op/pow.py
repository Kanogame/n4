from n4.numeric import NumericProtocol
from typing import Self
from n4.core import Op, Value


class Pow[T: NumericProtocol](Op[T]):
    def forward_pass(self: Self) -> list[Value[T]]:
        a, b = self.input_count(2)

        c = Value(a.data**b.data, parent_op=self)

        self.outputs = [c]

        return self.outputs

    def backward_pass(self: Self) -> None:
        a, b = self.input_count(2)
        out, *_ = self.output_count(1)

        backendOne: T = a._backend.one()

        a.grad += (b.data * a.data ** (b.data - backendOne)) * out.grad
