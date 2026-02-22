from typing import Self
from .op import Op

class Value[T]():
    """Класс отражающий одно значение тензона, ожидает в качестве типа self некий класс, поддерживающий базовые операции"""

    def __init__(self, data: T, last_op: Op=None):
        self.data = data
        self.grad: int = 0

        self._backward = lambda: None
        self.parent_ops: list[Op] = [last_op ] if last_op else []


    def backward(self: Self):
        topo: list[Value[T]]  = []
        visited = set()

        def build(v: Value):
            if v not in visited:
                visited.add(v)
                for f in v.parent_ops:
                    for i in f.inputs:
                        # TODO: remove recursion, remove inplace function
                        build(i)
                topo.append(v)

        build(self)

        self.grad = 1

        for v in reversed(topo):
            for f in v.parent_ops:
                f.backward_pass()

    
    def __add__(self: Self, other: Self) -> Self:
        from n4.op import Add
        return Add([self, other]).forward_pass()

    def __mul__(self: Self, other: Self) -> Self:
        from n4.op import Mul
        return Mul([self, other]).forward_pass()