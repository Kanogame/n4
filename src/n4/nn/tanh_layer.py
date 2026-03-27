from n4.numeric import NumericProtocol
from n4.tensor import Tensor
from .layer import Layer
from n4.op import Tanh


class TanhLayer[T: NumericProtocol](Layer[T]):
    """Elementwise tanh activation layer"""

    def forward_pass(self, x: Tensor[T]) -> Tensor[T]:
        return x.apply_activation(Tanh)

    def parameters(self) -> list:
        return []
