from n4.tensor import Tensor
from n4.op import Relu, NonOp
from n4.core import Value
from typing import Self
from n4.numeric import PyFloat
from n4.nn import Model, DenseLayer, SoftmaxLayer, Sequential
from n4.nn.loss import CrossEntropyLoss
from n4.optim import SGD, Adam

import random


class MyModel(Model[PyFloat]):
    def __init__(self: Self) -> None:
        self.backend = PyFloat
        self.model = Sequential(
            DenseLayer(1, 2, self.backend, Relu),
            DenseLayer(2, 3, self.backend, NonOp),
            SoftmaxLayer(self.backend),
        )

    def forward_pass(self, x: Tensor[PyFloat]) -> Tensor[PyFloat]:
        return self.model.forward_pass(x)

    def parameters(self):
        return self.model.parameters()


random.seed(12)

model = MyModel()
loss_fn = CrossEntropyLoss()
opt = SGD(model.parameters(), lr=1e-1)

target = Tensor.from_list([1, 0, 0], (1, 3), PyFloat)

x = Tensor.ones((1, 1), PyFloat)

preds: Tensor

for i in range(200):
    model.zero_grad()
    preds = model.forward_pass(x)
    loss = loss_fn(preds, target)
    loss.backward()
    opt.step()
    print(f"loss: {loss.data}")

# graph = loss.collect_graph()
# graph.to_graphviz("out")


print(preds[0][0].data, preds[0][1].data, preds[0][2].data)
