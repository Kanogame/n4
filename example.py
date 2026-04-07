from n4.tensor import Tensor
from n4.op import Relu, NonOp
from typing import Self
from n4.numeric import PyFloat
from n4.nn import Model, DenseLayer, SoftmaxLayer, Sequential


class MyModel(Model[PyFloat]):
    def __init__(self: Self) -> None:
        self.backend = PyFloat
        self.model = Sequential(
            DenseLayer(5, 10, self.backend, Relu[PyFloat]),
            DenseLayer(10, 3, self.backend, NonOp),
            SoftmaxLayer(self.backend),
        )

    def forward_pass(self, x: Tensor[PyFloat]) -> Tensor[PyFloat]:
        return self.forward_pass(x)


model = MyModel()

res = model.forward_pass(Tensor.ones((1, 5), PyFloat))

print(res[0][0])
