from n4.core.numeric import NumericProtocol
from typing import Self
from n4.core import Op, Value

class Add[T: NumericProtocol](Op[T]):
    def forward_pass(self: Self) -> Value[T]:
        self.arg_count(self.inputs, 2)

        a, b = self.inputs

        c = Value[T](a.data + b.data, parent_op=self)

        self.outputs = [c]

        return c

    def backward_pass(self: Self):

        out = self.outputs[0]
        a, b = self.inputs

        a.grad += out.grad
        b.grad += out.grad