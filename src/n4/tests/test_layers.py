import math
import pytest
from typing import Sequence, Tuple
from n4.numeric import PyFloat, NumpyFloat
from n4.tensor import Tensor
from n4.nn.tanh_layer import TanhLayer
from n4.nn.softmax_layer import SoftmaxLayer
from n4.nn.dence_layer import DenseLayer
from n4.nn.sequential import Sequential
from n4.nn.neuron import Neuron
from n4.nn.conv_layer import ConvLayer
from n4.core import Value


def make_tensor_from_floats(
    vals: Sequence[float], shape: Tuple[int, ...]
) -> Tensor[PyFloat]:
    data = [Value.from_float(v, PyFloat) for v in vals]
    return Tensor(data, shape)


def test_tanh_layer_forward() -> None:
    vals = [-1.0, 0.0, 1.0]
    x = make_tensor_from_floats(vals, (1, 3))

    layer = TanhLayer(PyFloat)
    out = layer.forward_pass(x)

    for i, v in enumerate(out._data):
        assert math.isclose(v.data.v, math.tanh(vals[i]), rel_tol=1e-6)


def test_softmax_layer_probs_sum_to_one() -> None:
    vals = [1.0, 2.0, 3.0, 0.0, 0.0, 0.0]
    x = make_tensor_from_floats(vals, (2, 3))

    layer = SoftmaxLayer(PyFloat)
    out = layer.forward_pass(x)

    for r in range(2):
        row = [out._data[r * 3 + c] for c in range(3)]
        s = sum([v.data.v for v in row])
        assert abs(s - 1.0) < 1e-6


def test_dense_layer_shape_and_parameters() -> None:
    in_f = 4
    out_f = 2
    layer = DenseLayer(in_f, out_f, PyFloat)

    x = Tensor.ones((3, in_f), backend=PyFloat)
    out = layer.forward_pass(x)

    assert out.shape == (3, out_f)
    assert len(layer.parameters()) == in_f * out_f + out_f


def test_dense_layer_wrong_in_features_raises() -> None:
    """Проверяет ошибку при несовпадении входной размерности"""
    layer = DenseLayer(4, 2, PyFloat)
    x = Tensor.ones((3, 3), backend=PyFloat)
    with pytest.raises(ValueError):
        layer.forward_pass(x)


def test_dense_layer_non_2d_input_raises() -> None:
    """Проверяет ошибку при подаче не 2D тензора"""
    layer = DenseLayer(2, 3, PyFloat)
    x_3d = Tensor([Value.from_float(1.0, PyFloat)] * 8, (2, 2, 2))
    with pytest.raises(NotImplementedError):
        layer.forward_pass(x_3d)


def test_neuron_forward() -> None:
    """Проверяет прямой проход нейрона"""
    neuron = Neuron(3, PyFloat)
    x = Tensor.ones((3,), backend=PyFloat)
    out = neuron(x)
    assert isinstance(out, Value)


def test_neuron_parameters() -> None:
    """Проверяет что нейрон возвращает правильное количество параметров"""
    neuron = Neuron(4, PyFloat)
    params = neuron.parameters()
    assert len(params) == 5


def test_neuron_repr() -> None:
    """Проверяет __repr__ нейрона"""
    neuron = Neuron(2, PyFloat)
    r = repr(neuron)
    assert "Neuron" in r


def test_conv_layer_forward_shape() -> None:
    """Проверяет форму вывода ConvLayer"""
    layer = ConvLayer(
        in_channels=1,
        out_channels=2,
        kernel_size=3,
        backend=PyFloat,
        stride=1,
        padding=0,
    )
    data = [Value.from_float(float(i), PyFloat) for i in range(25)]
    x = Tensor(data, (1, 5, 5))
    out = layer.forward_pass(x)
    assert out.shape == (2, 3, 3)


def test_conv_layer_parameters() -> None:
    """Проверяет количество параметров ConvLayer"""
    layer = ConvLayer(in_channels=1, out_channels=1, kernel_size=3, backend=PyFloat)
    params = layer.parameters()
    assert len(params) == 1 * 1 * 3 * 3 + 1


def test_conv_layer_wrong_ndim_raises() -> None:
    """Проверяет ошибку при подаче тензора неверной размерности"""
    layer = ConvLayer(in_channels=1, out_channels=1, kernel_size=3, backend=PyFloat)
    x = Tensor.ones((3, 4), backend=PyFloat)
    with pytest.raises(ValueError):
        layer.forward_pass(x)


def test_conv_layer_wrong_channels_raises() -> None:
    """Проверяет ошибку при несовпадении числа каналов"""
    layer = ConvLayer(in_channels=2, out_channels=1, kernel_size=3, backend=PyFloat)
    data = [Value.from_float(1.0, PyFloat) for _ in range(25)]
    x = Tensor(data, (1, 5, 5))
    with pytest.raises(ValueError):
        layer.forward_pass(x)


def test_conv_layer_with_padding() -> None:
    """Проверяет ConvLayer с padding=1"""
    layer = ConvLayer(
        in_channels=1, out_channels=1, kernel_size=3, backend=PyFloat, padding=1
    )
    data = [Value.from_float(1.0, PyFloat) for _ in range(9)]
    x = Tensor(data, (1, 3, 3))
    out = layer.forward_pass(x)
    assert out.shape == (1, 3, 3)


def test_sequential_call_operator() -> None:
    """Проверяет что Sequential.__call__ делегирует forward_pass"""
    layer = DenseLayer(2, 3, PyFloat)
    seq = Sequential(layer)
    x = Tensor.ones((1, 2), backend=PyFloat)
    out = seq(x)
    assert out.shape == (1, 3)


def test_sequential_single_layer() -> None:
    """Проверяет Sequential с одним слоем"""
    layer = DenseLayer(2, 2, PyFloat)
    seq = Sequential(layer)
    assert seq.layers_have_same_backend()


def test_sequential_mixed_backend_raises() -> None:
    """Проверяет что Sequential с разными бекендами выбрасывает ошибку"""
    l1 = DenseLayer(2, 3, PyFloat)
    l2 = DenseLayer(3, 1, NumpyFloat)
    with pytest.raises(ValueError):
        Sequential(l1, l2)  # type: ignore


def test_tanh_layer_neuron_count_is_none() -> None:
    """Проверяет что TanhLayer возвращает None для neuron_count"""
    layer = TanhLayer(PyFloat)
    assert layer.neuron_count() is None


def test_softmax_layer_neuron_count_is_none() -> None:
    """Проверяет что SoftmaxLayer возвращает None для neuron_count"""
    layer = SoftmaxLayer(PyFloat)
    assert layer.neuron_count() is None
