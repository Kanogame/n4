from typing import Optional, Self
from n4.core.numeric import NumericProtocol, PyFloat
from n4.core import Value, Op
from n4.op import NonOp
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
class Neuron[T: NumericProtocol](NnBase):

    # Веса нейрона 
    w: list[Value[T]]

    # Bias нейрона
    b: Value[T]

    # Функция активации
    activation: Op[T]

    # Бекенд вычислений
    _backend: type[T]

    def __init__(self: Self, w_len: int, activation: Optional[Op[T]]=None, _backend: type[T]=PyFloat):
        """
        Инициализация нейрона

        w_len: Количество входов нейрона, и соответственно кол-во весов
            Устанавливается в заначение >=1. Иначе будет выброшена ошибка
        
        activation: Функция ативации
            Устанавливается в любую скаляную функцию скалярного агрумента.
            Если требуется сделать линейрный нейрон, передается None или NonOp 
        """

        # Todo: uniform rand?
        self.w = [self._backend.zero() for _ in w_len]
        self.b = self._backend.zero()

        self.activation = NonOp() if activation is None else activation

    def __call__(self, x):
        act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
        return act(self.activation)
    
    def parameters(self):
        return self.w + [self.b]
    
    def __repr__(self):
        return f"{self.activation} Neuron({len(self.w)})"