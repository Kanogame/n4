from n4.core.numeric import NumericProtocol
from typing import Self
from n4.core import Op, Value

class Add[T: NumericProtocol](Op[T]):
    def forward_pass(self: Self) -> Value[T]:
        a, b = self.input_count(2)

        c = Value[T](a.data + b.data, parent_op=self)

        self.outputs = [c]

        return c

    def backward_pass(self: Self):

        out = self.output_count(1)
        a, b = self.input_count(2)

        a.grad += out.grad
        b.grad += out.grad