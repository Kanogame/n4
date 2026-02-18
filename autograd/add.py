from function import Function
from value import Value

class Add(Function):
    adder_a: Value
    adder_b: Value
    result: Value

    def forward_pass(self, *args: list[Value]) -> Value:
        a, b = self.arg_count(args, 2)

        self.adder_a = a
        self.adder_b = b

        c = Value(a + b, last_op=self)

        self.result = c

        return c

    def backward_pass(self):
        
        a_new = self.adder_a.get_grad() + self.result.get_grad()

        self.adder_a.set_grad(a_new)

        b_new = self.adder_b.get_grad() + self.result.get_grad()

        self.adder_b.set_grad(b_new)


        ## https://github.com/karpathy/micrograd/blob/master/micrograd/engine.py
        ## should I set backward to Value, or is there a cleaner way with backward_pass?