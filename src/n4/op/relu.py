from n4.core.numeric import NumericProtocol
from typing import Self
from n4.core import Op, Value

class Relu[T: NumericProtocol](Op[T]):
    def forward_pass(self: Self) -> list[Value[T]]:
        a, *_ = self.input_count(1)

        backendZero: T = a._backend.zero()

        b = Value(backendZero if a.data < 0 else a.data, parent_op=self)

        self.outputs = [b]

        return self.outputs

    def backward_pass(self: Self):
        out, *_ = self.output_count(1)
        a, *_ = self.input_count(1)

        a.grad += (out.data > 0) * out.grad
