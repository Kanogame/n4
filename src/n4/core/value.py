from typing import Self
from .op import Op

class Value[T]():
    """
    Класс отражающий одно значение тензона, ожидает в качестве типа self некий класс, поддерживающий базовые операции согластно numericProtocol
    """

    def __init__(self, data: T, last_op: Op[T]=Op([])):
        """
        Инициализация класса, ожидает 
        
        data: numericProtocol, 
        last_op: Последняя операция произведенная над значением, заполняется только при вызове из операций
        """

        self.data = data
        self.grad: T = 0

        self._backward = lambda: None
        self.parent_op: Op = last_op

    def backward(self: Self):
        topo: list[Value[T]]  = []
        visited = set()

        def build(v: Value):
            if v not in visited:
                visited.add(v)
                for i in v.parent_op.inputs:
                    build(i)
                topo.append(v)

        build(self)

        self.grad = 1

        for v in reversed(topo):
            v.parent_op.backward_pass()

    
    def __add__(self: Self, other: Self) -> Self:
        from n4.op import Add
        return Add([self, other]).forward_pass()

    def __mul__(self: Self, other: Self) -> Self:
        from n4.op import Mul
        return Mul([self, other]).forward_pass()