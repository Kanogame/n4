from n4.core import Value
from typing import Self, Optional
from n4.numeric import NumericProtocol
from n4.tensor import Tensor
from .layer import Layer
from n4.op import Tanh


class TanhLayer[T: NumericProtocol](Layer[T]):
    """Поэлементный TanH"""

    def forward_pass(self: Self, x: Tensor[T]) -> Tensor[T]:
        return x.apply_activation(Tanh)

    def parameters(self: Self) -> list[Value[T]]:
        return []

    def neuron_count(self: Self) -> Optional[int]:
        """
        Количество нейронов в слое
        """
        return None
