from typing import Callable
from function import Function
from value import Value

class Add(Function):
    def forward_pass(self, *args: list[Value]) -> Value:
        a, b = self.arg_count(args, 2)

        res = a.copy()

        res.data += b.data

        ## https://github.com/karpathy/micrograd/blob/master/micrograd/engine.py
        ## should I set backward to Value, or is there a cleaner way with backward_pass?

    def backward_pass(self, out: Value, *args: list[Value]):
        a, b = self.arg_count(args, 2)

        out.