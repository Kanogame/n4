from n4.core.numeric import NumericProtocol
from typing import Self
from n4.core import Op, Value


class Relu[T: NumericProtocol](Op[T]):
    def forward_pass(self: Self) -> list[Value[T]]:
        a, *_ = self.input_count(1)

        backendZero: T = a._backend.zero()

        b = Value(backendZero if a.data < backendZero else a.data, parent_op=self)

        self.outputs = [b]

        return self.outputs

    def backward_pass(self: Self) -> None:
        out, *_ = self.output_count(1)
        a, *_ = self.input_count(1)

        backendZero: T = a._backend.zero()

        if out.data > backendZero:
            a.grad += out.grad
