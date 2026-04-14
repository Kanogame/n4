# What is n4
n4 stands for NNNN, which stands for NaNo Neural Network.

This is simplistic OOP NN framework written in python.

In many ways inspired by PyTorch

# Rough project layout
- autograd
  - autodifferentiation system and computation graph constructor
- optim
  - SGD
- nn
  - layers
  - model strucure
  - loss functions
- dataset
  - default dataset interface

# Deps
- ty
- pytest
- ruff

TODO:
- SGD .parameters error
- bump version

# OPTIMIZATION OF COMP GRAPH
Your goal is: optimizing graph structure, for faster rendering
Graph is very detailed, has a lot of useless text, that slows generation dramatically, reduce the overhead. Display only necessary info


# Roadmap
## Stage 1 - n4 (standalone lib)
Soft deadline 02.28 -> Hard deadline 03.08

- Implement Value
  + data / grad / requires_grad
  +/- operator overloading
  + backward()
  - backward_step()

+ Implement internal OpNode
  + forward dependencies
  + saved values
  + backward function

- Implement execution tape
  - forward graph build
  - reverse traversal
  - gradient accumulation

- Graph Introspection API
  - GraphNode
  - GraphTrace
  - trace()
  - backward_order()

- Implement Parameter

- Implement Module
  - automatic parameter registration
  - parameters()
  - zero_grad()

- Implement nn
  - Linear
  - relu
  - tanh

- Implement loss
  - mse

- Implement optim
  - SGD
  - step()
  - zero_grad()

- Tests
  + numeric gradient check
  - scalar expression test
  - simple MLP convergence


## Stage 2 - runtime-loader (IDE project)
Soft deadline 03.15 -> Hard deadline 03.20

- Create project that depends on n4

- Dynamic loading of user-defined model
  - import model.py at runtime
  - find Module subclass
  - instantiate model

- Execute forward pass
  - create input Values
  - run model(x)
  - compute loss

- Extract runtime graph
  - loss.trace()
  - backward_order()


## Stage 3 - UI shell (Qt6)
Soft deadline 03.30 -> Hard deadline 04.05

- Create main window

- Layout
  - left pane: code editor
  - right pane: graph view
  - bottom pane: playback controls

- Editor
  - embed QScintilla widget
  - python syntax highlighting
  - file open / save

- Graph View
  - QGraphicsScene
  - QGraphicsView
  - pan / zoom enabled


## Stage 4 - graph render
Soft deadline 04.10 -> Hard deadline 04.15

- Use NetworkX for layout
  - convert GraphTrace to nx graph
  - run spring layout

- Render nodes
  - GraphNode as rect
  - show op name
  - show value / grad

- Render edges
  - forward dependencies


## Stage 5 - backward playback
Soft deadline 04.20 -> Hard deadline 04.25

- Controls
  - Step
  - Run
  - Reset

- On step
  - call backward_step()
  - highlight active node
  - update grad display

- Edge thickness based on grad magnitude


## Stage 6 - training loop
Soft deadline 04.20 -> Hard deadline 04.25

- Run N training steps
  - forward
  - backward
  - optimizer step

- Display loss curve
  - embed matplotlib canvas

- Reset model state