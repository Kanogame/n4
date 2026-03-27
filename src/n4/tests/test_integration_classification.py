from typing import Sequence, Tuple
from n4.numeric import PyFloat
from n4.tensor import Tensor
from n4.core import Value
from n4.nn.dence_layer import DenseLayer
from n4.nn.softmax_layer import SoftmaxLayer
from n4.nn.sequential import Sequential
from n4.nn.loss import CrossEntropyLoss
from n4.optim.sgd import SGD


def make_tensor(vals: Sequence[float], shape: Tuple[int, ...]) -> Tensor[PyFloat]:
    data = [Value.from_float(v, PyFloat) for v in vals]
    return Tensor(data, shape)


def test_mlp_classification_converges() -> None:
    # simple binary classification: inputs 0 and 1 map to class 0 and 1
    x = make_tensor([0.0, 1.0], (2, 1))
    # targets one-hot
    target = make_tensor([1.0, 0.0, 0.0, 1.0], (2, 2))

    model = Sequential(
        DenseLayer(1, 8, PyFloat, None),
        DenseLayer(8, 2, PyFloat, None),
        SoftmaxLayer(PyFloat),
    )

    loss_fn = CrossEntropyLoss()
    opt = SGD(model.parameters(), lr=1e-1)

    for epoch in range(200):
        model.zero_grad()
        preds = model.forward_pass(x)
        loss = loss_fn(preds, target)
        loss.backward()
        opt.step()

    final_preds = model.forward_pass(x)

    # compute accuracy
    rows, cols = final_preds.shape
    correct = 0
    for i in range(rows):
        # pick argmax
        row = [final_preds._data[i * cols + j].data.v for j in range(cols)]
        pred_idx = int(row.index(max(row)))
        # find target index
        trow = [target._data[i * cols + j].data.v for j in range(cols)]
        true_idx = int(trow.index(max(trow)))
        if pred_idx == true_idx:
            correct += 1

    assert correct / rows >= 0.5
