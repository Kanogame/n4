# Todo:
# Abstrac layer with support of convolution
# What to do: 
# Some typed (Numeric) layer with inps and outputs
# Then, neuron layer, with connetion matrix
# Create convolution layer with conv matrix
# Create dense layer of neurons
# In modal everything should be layer
from n4.core import Op, Value
from typing import Optional
from abc import ABC, abstractmethod
from .nn_base import NnBase
from n4.core.numeric import NumericProtocol
from .neuron import Neuron

class Layer[T: NumericProtocol](NnBase, ABC):
    """
    Базовый класс для любых слоев сети

    activation: Функция активации для всех нейронов в слое
    """

    def __init__(self, activation: Optional[type[Op[T]]] = None):
        super().__init__(self)


        self.activation = self.resolve_activation(activation)
        self.neurons: list[Neuron[T]] = []

    @abstractmethod
    def forward(self, x: list[Value[T]]) -> list[Value[T]]:
        """Forward pass of the layer. To be implemented by subclasses."""
        pass

    def __call__(self, x: Any) -> Any:
        return self.forward(x)

    def parameters(self) -> List[Value[T]]:
        """Collect all trainable parameters from the layer's neurons."""
        params = []
        for neuron in self.neurons:
            params.extend(neuron.parameters())
        return params