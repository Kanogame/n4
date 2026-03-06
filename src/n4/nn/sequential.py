from n4.tensor import Tensor
from typing import Self
from n4.core.numeric import NumericProtocol

from .nn_base import NnBase
from .layer import Layer


# TODO: during layer config make sure that ins & outs match
class Sequential[T: NumericProtocol](NnBase[T]):
    layers: list[Layer[T]]

    def __init__(self: Self, *args: Layer[T]):
        super().__init__(self)

        self.layers = list(args)

        if len(args) == 0:
            raise ValueError("Sequential model must contain at least one layer")

        if self.layers_have_same_backend():
            raise ValueError("Sequential model must contain layers with same backend")

    def forward_pass(self: Self, x: Tensor[T]) -> Tensor[T]:
        nextv: Tensor[T] = x

        for i in self.layers:
            nextv = i(nextv)

        return nextv

    def layers_have_same_backend(self: Self) -> bool:
        first: type[T] = self.layers[0]._backend

        if self._backend != first:
            return False

        if len(self.layers) <= 1:
            return True

        for i in self.layers[1:]:
            if i._backend != first:
                return False

        return True
