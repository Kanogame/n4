from n4.core import Value
from typing import Self
import random

from n4.tensor import Tensor
from n4.op import Relu, NonOp, Tanh
from n4.numeric import PyFloat
from n4.nn import Model, DenseLayer, SoftmaxLayer, Sequential
from n4.loss import CrossEntropyLoss, MSELoss
from n4.optim import SGD

xor_data = [
    ([0.0, 0.0], 0.0),
    ([0.0, 1.0], 1.0),
    ([1.0, 0.0], 1.0),
    ([1.0, 1.0], 0.0),
]

# Alternative simpler way: create tensors directly
x = []
y = []

for inputs, output in xor_data:
    # Create tensor from list of values
    input_values = [Value.from_float(x, backend=PyFloat) for x in inputs]
    x.append(Tensor(input_values, shape=(1, 2)))
    
    output_values = [Value.from_float(output, backend=PyFloat)]
    y.append(Tensor(output_values, shape=(1, 1)))



class MyModel(Model[PyFloat]):
    def __init__(self: Self) -> None:
        self.backend = PyFloat
        self.model = Sequential(
            DenseLayer(2, 4, self.backend, Tanh),
            DenseLayer(4, 1, self.backend, Tanh),
        )

    def forward_pass(self, x: Tensor[PyFloat]) -> Tensor[PyFloat]:
        return self.model.forward_pass(x)

    def parameters(self: Self) -> list[Value[PyFloat]]:
        return self.model.parameters()


random.seed(328)

model = MyModel()
loss_fn = MSELoss[PyFloat]()
opt = SGD(model.parameters(), lr=1e-1)
l_loss: Value[PyFloat]

preds: Tensor[PyFloat]

for i in range(100):
    for i in range(len(x)):
        model.zero_grad()
        preds = model.forward_pass(x[i])
        loss = loss_fn(preds, y[i])
        loss.backward()
        l_loss = loss
        opt.step()
        print([i.data for i in x[i]._data])
    print(f"loss: {loss.data}")
graph = l_loss.collect_graph().export_graphviz()
graph.render('output_graph', format='svg', view=False)