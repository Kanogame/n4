from n4.numeric import NumericProtocol
from typing import Self, Optional, Any, Tuple
from .op import Op


class Value[T: NumericProtocol]:
    """
    Класс отражающий одно значение тензона, ожидает в качестве типа self некий класс, поддерживающий базовые операции согластно NumericProtocol
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
        self.grad: T = self._backend.from_float(0)
        self.parent_op = parent_op

    def zero_grad(self: Self) -> None:
        """
        Метод позволят обнулить градиент
        """

        self.grad = self._backend.from_float(0)

    @classmethod
    def from_int[N: NumericProtocol](cls, value: int, backend: type[N]) -> "Value[N]":
        """
        Создание Value с бекендом backend из int.
        """

        return Value(backend.from_float(value))

    @classmethod
    def from_float[N: NumericProtocol](
        cls, value: float, backend: type[N]
    ) -> "Value[N]":
        """
        Создание Value с бекендом backend из int.
        """

        return Value(backend.from_float(value))

    def get_backend(self: Self) -> type:
        return self._backend

    def get_float(self: Self) -> float:
        return self.data.get_float()

    def collect_graph(self: Self) -> Any:
        from .comp_node import CompGraph

        return CompGraph.collect(self)

    def backward(self: Self) -> None:
        """
        Метод возврата по вычисительному графу с подсчетом градиента

        1. Происходит сбор топологии, с учетом детей
        2. Метод задает градиент текущего значений в 1
        3. Производим распространение
        """

        # P.S. Reinventing a wheel is always a bad idea
        # Когда я посмотрел на micrograd я подумал - я могу лучще
        # И вместо их правильной рекурсивной модели, использовал тупой BFS
        # И распространял градиент сразу - что было тупостью :)
        # Подробно в docs/backward.md

        # Первый проход: построение топологического порядка
        topo = []
        visited = set()
        stack: list[Tuple[Value[T], bool]] = [
            (self, False)
        ]  # (node, processed_children_flag)

        while stack:
            v, processed = stack.pop()
            if processed:
                topo.append(v)
                continue
            if v in visited:
                continue
            visited.add(v)
            stack.append((v, True))
            if v.parent_op is not None:
                for inp in v.parent_op.inputs:
                    if inp not in visited:
                        stack.append((inp, False))

        # Второй проход: установка градиента и обратный проход
        self.grad = self._backend.from_float(1)

        for v in reversed(topo):
            if v.parent_op is not None:
                v.parent_op.backward_pass()

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

    def exp(self: Self) -> "Value[T]":
        """
        Возведение экспоненты в степерь с использованием класса Exp
        """

        from n4.op import Exp

        return self._forward_pass_operation(Exp, self)

    def apply_activation(self: Self, activation: type[Op[T]]) -> "Value[T]":
        return self._forward_pass_operation(activation, self)

    def __repr__(self) -> str:
        return f"n4.core.Value(data: {self.data}, grad: {self.grad}, backend: {self._backend}, parent_op: {self.parent_op})"
