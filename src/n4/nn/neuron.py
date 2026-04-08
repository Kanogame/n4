from typing import Optional, Self
from n4.numeric import NumericProtocol
from n4.core import Value, Op
from n4.tensor import Tensor
from .nn_base import NnBase


# Todo: do we really need numberic protocol on high level?
# We need to AVOID AT ALL COSTS mixing backends
# to prevent roundtrips GPU -> CPU -> GPU, or C++ -> Python -> C++
# But on the other hand, explicit type give nothing to developer
# Other that something to "worry about".
#
# It is mostly a stylistic question, regradles of resuls, we need to control, either explcitly or implicitly
# Therefore it is better be done explicitly.
#
# Also, T in current impl in inferred implicitly (if ever), need improvements
#
# Convert above to docs
class Neuron[T: NumericProtocol](NnBase[T]):
    # Веса нейрона
    w: Tensor[T]

    # Bias нейрона
    b: Value[T]

    # Функция активации
    activation: type[Op[T]]

    def __init__(
        self: Self,
        w_len: int,
        backend: type[T],
        activation: Optional[type[Op[T]]] = None,
    ):
        """
        Инициализация нейрона

        w_len: Количество входов нейрона, и соответственно кол-во весов
            Устанавливается в заначение >=1. Иначе будет выброшена ошибка

        activation: Функция ативации
            Устанавливается в любую скаляную функцию скалярного агрумента.
            Если требуется сделать линейрный нейрон, передается None или NonOp
        """
        super().__init__(backend)

        self.w = Tensor.random_uniform(
            (w_len,), low=-1.0, high=1.0, backend=self._backend
        )
        self.b = Value.from_float(1, self._backend)

        self.activation = self.resolve_activation(activation)

    def __call__(self: Self, x: Tensor[T]) -> Value[T]:
        prod = self.w * x
        dot: Value[T] = prod.sum()
        pre_activation = dot + self.b
        return pre_activation.apply_activation(self.activation)

    def parameters(self: Self) -> list[Value[T]]:
        return self.w._data + [self.b]

    def __repr__(self: Self) -> str:
        return f"{self.activation} Neuron({len(self.w._data)})"
