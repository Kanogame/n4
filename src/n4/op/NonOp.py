from n4.core.numeric import NumericProtocol
from typing import Self
from n4.core import Op, Value

class NonOp[T: NumericProtocol](Op[T]):
    def forward_pass(self: Self) -> Value[T]:
        return self.inputs

    def backward_pass(self: Self):
        # Todo: pass all grads directly, to not cause problem 
        return