import operator
from functools import reduce
from typing import List, Tuple, Any, Union, Optional, Self
from n4.core import Value
from n4.core.numeric import NumericProtocol
# TODO: sketch, full refactor required

class Tensor[T: NumericProtocol]:
    """
    Многомерный массим элементов Value, на бекенде T
    """

    _data: list[Value[T]]
    _shape: Tuple[int, ...]
    _backend: type[T]

    def __init__(self: Self, data: List[Value[T]], shape: Tuple[int, ...]):
        """
        Многомерный массим элементов Value, на бекенде T

        Тензор представляет собой одномерный массив, который может представлять содержимое ввиде многомерного массива формы shape

        Все операции намеренно запрещают смешивать бекенд

        data: массив Value
        shape: Кортеж чисел отражающий форму
            должен по размеру соответствовать длинне data
        """

        expected: int = self.get_total_size(shape)

        if len(data) != expected:
            raise ValueError(
                f"Data length {len(data)} does not match shape {shape} (expected {expected})"
            )

        self._data = data
        self._shape = shape
        self._backend = data[0].get_backend()

    @property
    def shape(self) -> Tuple[int, ...]:
        """Форма тензора."""

        return self._shape

    @property
    def ndim(self) -> int:
        """Размерность пространства тензора."""

        return len(self._shape)

    @property
    def size(self) -> int:
        """Общее кол-во элементов, эквивалентно перемножению всех размерностей shape."""

        return len(self._data)

    @property
    def backend(self) -> type[T]:
        """Бекенд хранения Value."""

        return self._backend

    @staticmethod
    def get_total_size(shape: Tuple[int, ...]) -> int:
        return reduce(operator.mul, shape, 1)

    @staticmethod
    def zeros[N: NumericProtocol](shape: Tuple[int, ...], backend: type[N]) -> "Tensor[N]":
        """Создать Тензор заплоненный нулями, размера shape"""

        total: int = Tensor.get_total_size(shape)
        data: list[Value[N]] = [Value(backend.zero()) for _ in range(total)]
        return Tensor(data, shape)

    @staticmethod
    def ones[N: NumericProtocol](shape: Tuple[int, ...], backend: type[N]) -> "Tensor[N]":
        """Создать Тензор заплоненный единицами, размера shape"""

        total: int = Tensor.get_total_size(shape)
        data: list[Value[N]] = [Value(backend.one()) for _ in range(total)]
        return Tensor(data, shape)

    @staticmethod
    def random_uniform[N: NumericProtocol](shape: Tuple[int, ...], backend: type[N], low: float = -1.0, high: float = 1.0) -> "Tensor[N]":
        """Создать тензор заполненный по равномерному распределению"""

        total: int = Tensor.get_total_size(shape)
        data: list[Value[N]] = [Value(backend.random_uniform(low, high)) for _ in range(total)]
        return Tensor(data, shape)

    def to_list(self) -> List[Any]:
        """Конвертировать в список согластно shape"""

        if self.ndim == 0:
            return self._data[0]  # type: ignore

        step = self.size // self._shape[0]
        return [
            Tensor[T](self._data[i * step : (i + 1) * step], self._shape[1:]).to_list()
            for i in range(self._shape[0])
        ]

    def __getitem__(self, idx: Union[int, Tuple[int, ...]]) -> Union[Value[T], "Tensor[T]"]:
        """
        Индексация тензора

        Подерживает индексацию по int или tuple. Если индекс указывает ВСЕ измерения, то вернет Value, 
        иначе, будет возращена копия тензора (слайс)
        """

        if isinstance(idx, int):
            idx = (idx,)
        if len(idx) > self.ndim:
            raise IndexError(f"Index {idx} has too many dimensions for shape {self._shape}")

        # Compute flat index of the start of the slice
        flat = 0
        stride = 1
        for i in range(len(idx) - 1, -1, -1):
            if idx[i] >= self._shape[i]:
                raise IndexError(f"Index {idx[i]} out of bounds for dimension {i}")
            flat += idx[i] * stride
            stride *= self._shape[i]

        if len(idx) == self.ndim:
            # Return a single element
            return self._data[flat]

        # Return a sub-tensor (copy)
        new_shape = self._shape[len(idx) :]
        step = self.get_total_size(new_shape)
        data_slice = self._data[flat : flat + step]
        return Tensor(data_slice, new_shape)


    def reshape(self, new_shape: Tuple[int, ...]) -> "Tensor[T]":
        """Создает новый тензор с указанной размерностью."""

        expected = self.get_total_size(new_shape)
        if expected != self.size:
            raise ValueError(
                f"Cannot reshape tensor of size {self.size} into shape {new_shape}"
            )
        return Tensor(self._data, new_shape)

    # TODO
    def _broadcast_shapes(self, other: "Tensor[T]") -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        """Подсчитать формы элементов для бродакастинга"""

        if self.ndim != other.ndim:
            raise NotImplementedError("Broadcasting with different numbers of dimensions")
        new_shape = []
        for i, (d1, d2) in enumerate(zip(self._shape, other._shape)):
            if d1 == d2 or d1 == 1 or d2 == 1:
                new_shape.append(max(d1, d2))
            else:
                raise ValueError(f"Incompatible shapes for broadcasting: {self._shape} and {other._shape}")
        return tuple(new_shape), tuple(new_shape)

    def _broadcast_to(self, target_shape: Tuple[int, ...]) -> "Tensor[T]":
        """Вер"""
        if self._shape == target_shape:
            return self
        # In a real implementation we would expand dimensions and repeat data.
        # For now we only handle the case where the shapes are already compatible
        # and the current shape can be expanded by repeating elements.
        # This is a placeholder – a full broadcasting implementation is complex.
        raise NotImplementedError("Full broadcasting not implemented in this example")

    def __add__(self, other: Union["Tensor[T]", Value[T]]) -> "Tensor[T]":

        if isinstance(other, Value):
            # Scalar addition: add the same value to every element
            new_data = [x + other for x in self._data]
            return Tensor(new_data, self._shape)
        # Tensor + Tensor
        # For simplicity, assume same shape (no broadcasting)

        if self._shape != other._shape:
            raise NotImplementedError("Broadcasting not implemented in this example")
        new_data = [x + y for x, y in zip(self._data, other._data)]
        return Tensor(new_data, self._shape)

    def __sub__(self, other: Union["Tensor[T]", Value[T]]) -> "Tensor[T]":

        if isinstance(other, Value):
            new_data = [x - other for x in self._data]
            return Tensor(new_data, self._shape)
        if self._shape != other._shape:
            raise NotImplementedError("Broadcasting not implemented")
        new_data = [x - y for x, y in zip(self._data, other._data)]
        return Tensor(new_data, self._shape)

    def __mul__(self, other: Union["Tensor[T]", Value[T]]) -> "Tensor[T]":

        if isinstance(other, Value):
            new_data = [x * other for x in self._data]
            return Tensor(new_data, self._shape)
        if self._shape != other._shape:
            raise NotImplementedError("Broadcasting not implemented")
        new_data = [x * y for x, y in zip(self._data, other._data)]
        return Tensor(new_data, self._shape)

    def __truediv__(self, other: Union["Tensor[T]", Value[T]]) -> "Tensor[T]":

        if isinstance(other, Value):
            new_data = [x / other for x in self._data]
            return Tensor(new_data, self._shape)
        if self._shape != other._shape:
            raise NotImplementedError("Broadcasting not implemented")
        new_data = [x / y for x, y in zip(self._data, other._data)]
        return Tensor(new_data, self._shape)

    def __neg__(self) -> "Tensor[T]":
        new_data = [-x for x in self._data]
        return Tensor(new_data, self._shape)

    def __matmul__(self, other: "Tensor[T]") -> "Tensor[T]":

        if self.ndim != 2 or other.ndim != 2:
            raise ValueError("__matmul__ currently only supports 2D tensors")
        m, n = self._shape
        n2, p = other._shape
        if n != n2:
            raise ValueError(f"Incompatible shapes for matmul: {self._shape} and {other._shape}")

        # Compute result matrix
        result_data = []
        for i in range(m):
            for j in range(p):
                # Dot product of row i of self and column j of other
                dot = self._backend.zero()
                for k in range(n):
                    dot += self._data[i * n + k] * other._data[k * p + j]
                result_data.append(dot)
        return Tensor(result_data, (m, p))

    def sum(self, dim: Optional[int] = None) -> Union[Value[T], "Tensor[T]"]:
        """Сумма всех элементов по измерению."""

        if dim is None:
            # Reduce to scalar
            total = self._backend.zero()
            for v in self._data:
                total += v
            return total

        if dim < 0 or dim >= self.ndim:
            raise ValueError(f"Dimension {dim} out of range for shape {self._shape}")

        # Sum along one dimension
        # Compute new shape
        new_shape = list(self._shape)
        new_shape.pop(dim)
        new_shape = tuple(new_shape)

        # Compute strides
        stride = 1
        for i in range(dim + 1, self.ndim):
            stride *= self._shape[i]

        outer = self.size // (self._shape[dim] * stride)

        new_data = []
        for i in range(outer):
            for j in range(stride):
                s = self._backend.zero()
                base = i * self._shape[dim] * stride + j
                for k in range(self._shape[dim]):
                    s += self._data[base + k * stride]
                new_data.append(s)

        return Tensor(new_data, new_shape)

    def mean(self, dim: Optional[int] = None) -> Union[Value[T], "Tensor[T]"]:
        """Среднее всех элементов по измерению."""

        if dim is None:
            total = self.sum()
            return total / self._backend.from_float(float(self.size))
        # Along a dimension
        summed = self.sum(dim)
        if isinstance(summed, Tensor):
            # Divide each element by the size of the reduced dimension
            factor = self._backend.from_float(float(self._shape[dim]))
            return summed / factor
        else:
            # Should not happen because sum(dim) returns a Tensor
            raise RuntimeError("Unexpected scalar from sum(dim)")

