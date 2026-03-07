from n4.numeric import PyFloat, NumericProtocol
from collections import deque
from typing import Self, Optional
from .op import Op


class Value[T: NumericProtocol]:
    """
    Класс отражающий одно значение тензона, ожидает в качестве типа self некий класс, поддерживающий базовые операции согластно numericProtocol
    """

    data: T
    grad: T
    parent_op: Optional[Op[T]]

    _backend: type[T]

    def __init__(self, data: T, parent_op: Optional[Op[T]] = None):
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

    def zero_grad(self: Self) -> None:
        """
        Метод позволят обнулить градиент
        """

        self.grad = self._backend.zero()

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

    def get_backend(self: Self) -> type:
        return self._backend

    def backward(self: Self) -> None:
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

    @staticmethod
    def _forward_pass_operation(op: type[Op[T]], *args: "Value[T]") -> "Value[T]":
        return op(list(args)).forward_pass()[0]

    def __add__(self: Self, other: "Value[T]") -> "Value[T]":
        """
        Перегрузка оператора суммирования с использованием класса Add
        """
        from n4.op import Add

        return self._forward_pass_operation(Add, self, other)

    def __sub__(self: Self, other: "Value[T]") -> "Value[T]":
        """
        Перегрузка оператора вычитания с использованием класса Sub
        """
        from n4.op import Sub

        return self._forward_pass_operation(Sub, self, other)

    def __mul__(self: Self, other: "Value[T]") -> "Value[T]":
        """
        Перегрузка оператора произведения с использованием класса Mul
        """
        from n4.op import Mul

        return self._forward_pass_operation(Mul, self, other)

    def __truediv__(self: Self, other: "Value[T]") -> "Value[T]":
        """
        Перегрузка оператора деления с использованием класса Div
        """
        from n4.op import Div

        return self._forward_pass_operation(Div, self, other)

    def __pow__(self: Self, other: "Value[T]") -> "Value[T]":
        """
        Перегрузка оператора возведения в степень с использованием класса Pow
        """

        from n4.op import Pow

        return self._forward_pass_operation(Pow, self, other)

    def __neg__(self: Self) -> "Value[T]":
        """
        Перегрузка "-N" с использованием класса Neg
        """
        from n4.op import Neg

        return self._forward_pass_operation(Neg, self)

    def relu(self: Self) -> "Value[T]":
        """
        Классический Relu с использованием класса Relu
        """

        from n4.op import Relu

        return self._forward_pass_operation(Relu, self)

    def apply_activation(self: Self, activation: type[Op[T]]) -> "Value[T]":
        return self._forward_pass_operation(activation, self)

    # TODO: all ops from micrograd

    def __repr__(self) -> str:
        return f"n4.core.Value(data: {self.data}, grad: {self.grad}, backend: {self._backend}, parent_op: {self.parent_op})"
