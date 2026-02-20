from function import Function
from value import Value

class Add[T](Function[T]):
    def forward_pass(self, *args: list[Value[T]]) -> Value[T]:
        self.arg_count(args, 2)

        a, b = self.inputs

        c = Value(a + b, last_op=self)

        self.outputs = [c]

        return c

    def backward_pass(self):
        out = self.outputs[0]
        a, b = self.inputs

        a.grad += out.grad
        b.grad += out.grad


        ## https://github.com/karpathy/micrograd/blob/master/micrograd/engine.py
        ## should I set backward to Value, or is there a cleaner way with backward_pass?