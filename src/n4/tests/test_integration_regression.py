from n4.numeric import PyFloat
from n4.tensor import Tensor
from n4.core import Value
from n4.nn.dence_layer import DenseLayer
from n4.nn.sequential import Sequential
from n4.loss import MSELoss
from n4.optim.sgd import SGD
from typing import Sequence, Tuple, cast


def make_tensor(vals: Sequence[float], shape: Tuple[int, ...]) -> Tensor[PyFloat]:
    data = [Value.from_float(v, PyFloat) for v in vals]
    return Tensor(data, shape)


def test_mlp_regression_converges() -> None:
    # simple identity mapping on 1D input
    x = make_tensor([-1.0, 0.0, 1.0], (3, 1))
    y = make_tensor([-1.0, 0.0, 1.0], (3, 1))

    model = Sequential(
        DenseLayer(1, 8, PyFloat, None),
        DenseLayer(8, 1, PyFloat, None),
    )

    loss_fn = MSELoss()
    opt = SGD(model.parameters(), lr=1e-2)

    for epoch in range(200):
        model.zero_grad()
        preds = model.forward_pass(x)
        loss = loss_fn(preds, y)
        loss.backward()
        opt.step()

    final_preds = model.forward_pass(x)
    # Mean squared error should be small after training
    final_loss = loss_fn(final_preds, y)
    from n4.numeric import PyFloat as PF

    lv = cast(PF, final_loss.data).v
    assert float(lv) < 0.2
