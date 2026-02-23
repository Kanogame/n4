from n4.core.numeric import PyFloat, NumericProtocol
from collections import deque
from typing import Self, Optional, Type
from .op import Op

class Value[T: NumericProtocol]():
    """
    Класс отражающий одно значение тензона, ожидает в качестве типа self некий класс, поддерживающий базовые операции согластно numericProtocol
    """

    data: T
    grad: T
    parent_op: Optional[Op[T]]

    _backend: type[T]

    def __init__(self, data: T, parent_op: Optional[Op[T]]=None):
        """
        Инициализация класса, ожидает 
        
        data: числовой тип, выпоняющий интерфейс NumericProtocol
        
        last_op: Последняя операция
            Последняя операция произведенная над значением, заполняется только при вызове из операций
            Нужна чтобы проходиться по вычислительному графу
        """

        self.data: T = data
        self._backend = type(data)
        self.grad: T = self._backend.zero()
        self.parent_op = parent_op

    @classmethod
    def from_int(cls, value: int) -> "Value[PyFloat]":
        """
        Создание Value с backend PyFloat из int.
        """
        return Value(PyFloat(value))

    @classmethod
    def from_float(cls, value: float) -> "Value[PyFloat]":
        """
        Создание Value с backend PyFloat из float.
        """
        return Value(PyFloat(value))

    def get_backend(self: Self) -> Type:
        return self._backend

    def backward(self: Self):
        """
        Метод возврата по вычисительному графу с подсчетом градиента.

        1. Метод задает градиент текущего значений в 1
        2. Происходит обход графа через dfs
        3. На каждом из шагов обхода, градиент распространяется сразу
        """

        self.grad = self._backend.one()

        stack = deque[Value[T]]()
        stack.append(self)
        visited = set()

        while len(stack) != 0:
            v = stack.popleft()
            if v in visited: 
                continue
            visited.add(v)
            
            if v.parent_op is not None:
                v.parent_op.backward_pass()
            
                for i in v.parent_op.inputs:
                    if i not in visited:
                        stack.appendleft(i)
    
    def __add__(self: Self, other: "Value[T]") -> "Value[T]":
        """
        Перегрузка оператора суммирования с использованием класса Add
        """
        from n4.op import Add
        return Add([self, other]).forward_pass()

    def __mul__(self: Self, other: "Value[T]") -> "Value[T]":
        """
        Перегрузка оператора произведения с использованием класса Mul
        """
        from n4.op import Mul
        return Mul([self, other]).forward_pass()

    # TODO: all ops from micrograd
    
    def __repr__(self) -> str:
        return f"n4.core.Value(data: {self.data}, grad: {self.grad}, backend: {self._backend}, parent_op: {self.parent_op})"