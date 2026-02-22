from typing import Self
from n4.core import Op, Value

class Add[T](Op[T]):
    def forward_pass(self: Self) -> Value[T]:
        self.arg_count(self.inputs, 2)

        a, b = self.inputs

        c = Value(a.data + b.data, last_op=self)

        self.outputs = [c]

        return c

    def backward_pass(self: Self):
        out = self.outputs[0]
        a, b = self.inputs

        a.grad += out.grad
        b.grad += out.grad