# n4 Public API

This document describes the stable public API of the n4 library.

The goal of this API is:

- provide scalar reverse-mode autodiff
- allow neural network construction via OOP
- expose runtime computational graph
- allow step-wise backward execution (for IDE playback)
- remain backend/UI agnostic

IDE integrations MUST rely only on this interface.


--------------------------------------------------
Core Type: Value
--------------------------------------------------

Represents a scalar differentiable value.

All computational graphs are built implicitly via operator overloading.


### Constructor

Value(
    data: float,
    *,
    requires_grad: bool = False,
    name: Optional[str] = None
)


### Attributes

data: float
    numeric value

grad: float
    accumulated gradient after backward pass

requires_grad: bool
    whether gradient should be computed

name: Optional[str]
    optional identifier (useful for UI)


### Methods

backward() -> None

    Executes full reverse-mode autodiff pass.


backward_step() -> bool

    Executes exactly one reverse-mode step.

    Returns:
        True  - step executed
        False - backward finished

    Used by UI playback engine.


zero_grad() -> None

    Sets grad to 0.


trace() -> GraphTrace

    Returns runtime execution graph.

    Used by IDE to visualize graph.


--------------------------------------------------
Operator Overloads
--------------------------------------------------

The following are supported:

v1 + v2
v1 - v2
v1 * v2
v1 / v2
v1 ** float
-v1

All return Value.

All build computational graph implicitly.


Example:

x = Value(2.0, requires_grad=True)
y = Value(3.0, requires_grad=True)

z = (x * y + x) ** 2


--------------------------------------------------
Graph Introspection API (FOR IDE)
--------------------------------------------------

Returned by:

loss.trace()


### GraphNode

GraphNode:

id: int
    unique node id

op: str
    operation name (e.g. add, mul, pow)

inputs: list[int]
    parent node ids

output: int
    output node id

value: float
    forward value

grad: float
    gradient after backward


### GraphTrace

GraphTrace:

nodes: list[GraphNode]
edges: list[(int, int)]


backward_order() -> list[int]

    reverse topological order
    used for playback scheduling


Example:

trace = loss.trace()
order = trace.backward_order()


--------------------------------------------------
Parameter
--------------------------------------------------

Trainable scalar.

class Parameter(Value)

Constructor:

Parameter(data: float)

Always:

requires_grad = True


--------------------------------------------------
Module
--------------------------------------------------

Base class for neural network layers.

class Module:


parameters() -> Iterator[Parameter]

    returns all registered parameters


zero_grad() -> None

    clears gradients for all parameters


__call__(*args, **kwargs)

    calls forward()


forward(...)

    override in subclass



Example:

class Linear(Module):

    def __init__(self):
        self.w = Parameter(0.5)
        self.b = Parameter(0.0)

    def forward(self, x):
        return self.w * x + self.b


--------------------------------------------------
Activations
--------------------------------------------------

relu(x: Value) -> Value
tanh(x: Value) -> Value


--------------------------------------------------
Loss
--------------------------------------------------

mse(pred: list[Value], target: list[Value]) -> Value


Example:

loss = mse(pred, target)


--------------------------------------------------
Optimizer
--------------------------------------------------

class SGD


Constructor:

SGD(
    params: Iterable[Parameter],
    lr: float
)


Methods:

step() -> None
zero_grad() -> None


--------------------------------------------------
Example: Full Training Step
--------------------------------------------------

x = Value(1.0, requires_grad=False)
y_true = Value(2.0, requires_grad=False)

model = Linear()
opt = SGD(model.parameters(), lr=0.01)

y_pred = model(x)

loss = (y_pred - y_true) ** 2

loss.backward()
opt.step()
opt.zero_grad()


--------------------------------------------------
Example: Playback (IDE)
--------------------------------------------------

loss.backward()      # OR:

while loss.backward_step():
    update_ui()

trace = loss.trace()
order = trace.backward_order()


--------------------------------------------------
IDE Contract
--------------------------------------------------

UI integrations MUST:

- treat Value as black box
- not access OpNode / Tape internals
- use trace() for graph
- use backward_step() for playback
- use Module.parameters() for training