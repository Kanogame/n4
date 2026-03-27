from n4.numeric import NumericProtocol
from typing import Self
from n4.core import Op, Value


class Log[T: NumericProtocol](Op[T]):
    def forward_pass(self: Self) -> list[Value[T]]:
        (a,) = self.input_count(1)

        # natural log using backend
        c = Value(a._backend.log(a.data), parent_op=self)

        self.outputs = [c]
        return self.outputs

    def backward_pass(self: Self) -> None:
        (a,) = self.input_count(1)
        out, *_ = self.output_count(1)

        # derivative of ln(x) = 1/x
        a.grad += out.grad / a.data
