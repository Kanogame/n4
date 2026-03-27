from n4.numeric import NumericProtocol
from typing import Self
from n4.core import Op, Value


class Tanh[T: NumericProtocol](Op[T]):
    def forward_pass(self: Self) -> list[Value[T]]:
        (a,) = self.input_count(1)

        # Use backend tanh if available
        c = Value(a._backend.tanh(a.data), parent_op=self)

        self.outputs = [c]
        return self.outputs

    def backward_pass(self: Self) -> None:
        (a,) = self.input_count(1)
        out, *_ = self.output_count(1)

        # derivative: 1 - tanh(x)^2, out.data is tanh(a)
        one = a._backend.from_float(1)
        a.grad += out.grad * (one - (out.data * out.data))
