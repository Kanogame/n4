from n4.numeric import NumericProtocol
from typing import Self
from n4.core import Op, Value


class Log[T: NumericProtocol](Op[T]):
    def forward_pass(self: Self) -> list[Value[T]]:
        (a,) = self.input_count(1)
        eps = a._backend.from_float(1e-7)
        self.eps = eps

        c = Value(a._backend.log(a.data + eps), parent_op=self)
        self.outputs = [c]
        return self.outputs

    def backward_pass(self: Self) -> None:
        (a,) = self.input_count(1)
        out, *_ = self.output_count(1)

        a.grad += out.grad / (a.data + self.eps)
