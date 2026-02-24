from n4.core.numeric import NumericProtocol
from n4.core import Op, Value
from typing import Self

class Div[T: NumericProtocol](Op[T]):
    def forward_pass(self: Self) -> Value:
        a, b = self.input_count(2)

        c = Value(a.data / b.data, parent_op=self)

        self.outputs = [c]

        return c
    
    def backward_pass(self):
        out = self.output_count(1)
        a, b = self.input_count(2)

        # todo
