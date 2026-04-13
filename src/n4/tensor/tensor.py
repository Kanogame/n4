import operator
from functools import reduce
from typing import List, Tuple, Any, Union, Self
from n4.core import Value, Op
from n4.numeric import NumericProtocol


class Tensor[T: NumericProtocol]:
    """
    Многомерный массим элементов Value, на бекенде T

    Тензор представляет собой одномерный массив, который может представлять содержимое ввиде многомерного массива формы shape

    Все операции намеренно запрещают смешивать бекенд

    Поддерживает broadcasting
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
        self._backend = self.ensure_same_backend(data)

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

    @property
    def Transposed(self) -> "Tensor[T]":
        """Транспозиция 2d матрицы"""
        if self.ndim != 2:
            raise ValueError("Transpose is only defined for 2‑D tensors")

        rows, cols = self.shape
        transposed_data: list[Value[T]] = [Value.from_float(0, self._backend)] * (
            rows * cols
        )

        for r in range(rows):
            row_offset = r * cols
            for c in range(cols):
                orig_idx = row_offset + c
                trans_idx = c * rows + r
                transposed_data[trans_idx] = self._data[orig_idx]

        return Tensor(transposed_data, (cols, rows))

    @staticmethod
    def get_total_size(shape: Tuple[int, ...]) -> int:
        return reduce(operator.mul, shape, 1)

    @staticmethod
    def ensure_same_backend(data: List[Value[T]]) -> type[T]:
        first = data[0].get_backend()

        for i in data[1:]:
            if first is not i.get_backend():
                raise ValueError("Data must have same backend")

        return first

    @staticmethod
    def zeros[N: NumericProtocol](
        shape: Tuple[int, ...], backend: type[N]
    ) -> "Tensor[N]":
        """Создать Тензор заплоненный нулями, размера shape"""

        total: int = Tensor.get_total_size(shape)
        data: list[Value[N]] = [Value.from_float(0, backend) for _ in range(total)]
        return Tensor(data, shape)

    @staticmethod
    def ones[N: NumericProtocol](
        shape: Tuple[int, ...], backend: type[N]
    ) -> "Tensor[N]":
        """Создать Тензор заплоненный единицами, размера shape"""

        total: int = Tensor.get_total_size(shape)
        data: list[Value[N]] = [Value.from_float(1, backend) for _ in range(total)]
        return Tensor(data, shape)

    @staticmethod
    def from_list[N: NumericProtocol](
        array: list[float | int], shape: Tuple[int, ...], backend: type[N]
    ) -> "Tensor[N]":
        """Создать тензор заполненный из массива чисел с нужной формой"""

        if Tensor.get_total_size(shape) != len(array):
            raise ValueError(
                "Cannot convert array to tensor with given shape: Sizes dont match"
            )

        data: list[Value[N]] = [Value.from_float(el, backend) for el in array]
        return Tensor(data, shape)

    @staticmethod
    def random_uniform[N: NumericProtocol](
        shape: Tuple[int, ...], backend: type[N], low: float = -1.0, high: float = 1.0
    ) -> "Tensor[N]":
        """Создать тензор заполненный по равномерному распределению"""

        total: int = Tensor.get_total_size(shape)
        data: list[Value[N]] = [
            Value(backend.random_uniform(low, high)) for _ in range(total)
        ]
        return Tensor(data, shape)

    def to_list(self: Self) -> List[Any]:
        """Конвертировать в список согластно shape"""

        if self.ndim == 0:
            return self._data[0]  # type: ignore

        step = self.size // self._shape[0]
        return [
            Tensor[T](self._data[i * step : (i + 1) * step], self._shape[1:]).to_list()
            for i in range(self._shape[0])
        ]

    def _normalize_index(self, idx: Union[int, Tuple[int, ...]]) -> Tuple[int, ...]:
        """Если индекс представляет собой int, создаем (idx), если он превышает размерность, выходим, иначе возвращаем idx"""

        if isinstance(idx, int):
            idx = (idx,)
        if len(idx) > self.ndim:
            raise IndexError(
                f"Index {idx} has too many dimensions for shape {self._shape}"
            )
        return idx

    def _flat_start(self, idx: Tuple[int, ...]) -> int:
        """Безопастно считает первых индекс слайса по индексу"""
        flat = 0
        stride = 1
        for i in range(len(idx) - 1, -1, -1):
            if idx[i] >= self._shape[i]:
                raise IndexError(f"Index {idx[i]} out of bounds for dimension {i}")
            flat += idx[i] * stride
            stride *= self._shape[i]
        return flat

    def __getitem__(
        self: Self, idx: Union[int, Tuple[int, ...]]
    ) -> Union[Value[T], "Tensor[T]"]:
        """
        Индексация тензора

        Подерживает индексацию по int или tuple. Если индекс указывает ВСЕ измерения, то вернет Value,
        иначе, будет возращена копия тензора (слайс)
        """

        # Всегда Tuple
        idx = self._normalize_index(idx)

        # Начало среза
        flat = self._flat_start(idx)

        # Если нужен 1 элемент, т.е. если индекс указал все измерения
        if len(idx) == self.ndim:
            return self._data[flat]

        # Под-тензор
        new_shape = self._shape[len(idx) :]
        step = self.get_total_size(new_shape)
        data_slice = self._data[flat : flat + step]
        return Tensor(data_slice, new_shape)

    def __setitem__(
        self: Self,
        idx: Union[int, Tuple[int, ...]],
        value: Union[Value[T], "Tensor[T]"],
    ) -> None:
        """
        Установка значения по индексанции тензора

        Подерживает индексацию по int или tuple. Если индекс указывает ВСЕ измерения, то задаст Value,
        иначе, задаст нужное кол-во элементов
        """

        # Всегда Tuple
        idx = self._normalize_index(idx)

        # Начало среза
        flat = self._flat_start(idx)

        # Если нужно обновить 1 элемент, т.е. если индекс указал все измерения
        if len(idx) == self.ndim:
            if isinstance(value, Tensor):
                raise TypeError("Scalar assignment requires a non‑Tensor value")
            self._data[flat] = value
            return

        # Обновить под-тензор
        new_shape = self._shape[len(idx) :]
        step = self.get_total_size(new_shape)

        if not isinstance(value, Tensor):
            raise TypeError("Slice assignment requires a Tensor value")
        if value._shape != new_shape:
            raise ValueError(
                f"Shape mismatch: target slice shape {new_shape} vs value shape {value._shape}"
            )

        self._data[flat : flat + step] = value._data

    def reshape(self, new_shape: Tuple[int, ...]) -> "Tensor[T]":
        """Создает новый тензор с указанной размерностью."""

        expected = self.get_total_size(new_shape)
        if expected != self.size:
            raise ValueError(
                f"Cannot reshape tensor of size {self.size} into shape {new_shape}"
            )
        return Tensor(self._data, new_shape)

    # broadcasing
    # Броадкастинг - это функционал автоизменения формы тензора для вычислений (как правило flatten), если его размер
    # (в данном случае len(_data)) совпадает
    #
    # Фактически это попытка упростить все математические операции, избавив разработчиков от
    # головной боли по поводу формы, при этом не игнорируя правила математики или логики полностью
    @staticmethod
    def _broadcast_both_tensors(
        a: "Tensor[T]", b: "Tensor[T]"
    ) -> Tuple["Tensor[T]", "Tensor[T]"]:

        new_a_shape, new_b_shape = a._broadcast_get_compatible_dims(a.shape, b.shape)
        return a._broadcast_to(new_a_shape), b._broadcast_to(new_b_shape)

    @staticmethod
    def _broadcast_both_tensors_zip(
        a: "Tensor[T]", b: "Tensor[T]"
    ) -> list[Tuple[Value[T], Value[T]]]:

        new_a, new_b = a._broadcast_both_tensors(a, b)
        return list(zip(new_a._data, new_b._data))

    @staticmethod
    def _broadcast_get_compatible_dims(
        a: Tuple[int, ...], b: Tuple[int, ...]
    ) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        """Вычисляет совместимые размеры для broadcasting.

        Применяет правила NumPy broadcasting:
        1. Выравнивает количество измерений, добавляя 1s в начало
        2. Результирующее измерение = max(a[i], b[i]) если хотя бы одно из них = 1

        Пример:
            a.shape = (3,)
            b.shape = (2, 3)

        После приведения:
            a -> (1, 3) -> (2, 3)
            b -> (2, 3) -> (2, 3)
        """

        max_n_dim = max(len(a), len(b))

        a_offset = max_n_dim - len(a)
        b_offset = max_n_dim - len(b)

        sized_a = [1] * a_offset + list(a)
        sized_b = [1] * b_offset + list(b)

        # Вычисляем итоговую форму для broadcasting
        result_a = []
        result_b = []

        for i in range(max_n_dim):
            dim_a = sized_a[i]
            dim_b = sized_b[i]

            if dim_a == dim_b:
                result_a.append(dim_a)
                result_b.append(dim_b)
            elif dim_a == 1:
                result_a.append(dim_b)
                result_b.append(dim_b)
            elif dim_b == 1:
                result_a.append(dim_a)
                result_b.append(dim_a)
            else:
                raise RuntimeError(
                    f"Tensors with shapes {a} and {b} cannot be broadcasted"
                )

        return (tuple(result_a), tuple(result_b))

    def _broadcast_to(self: Self, target_shape: Tuple[int, ...]) -> "Tensor[T]":
        """Преобразует Тензор в указанную форму путем повторения элементов.

        Поддерживает NumPy-подобный broadcasting: если текущая форма совпадает с target_shape,
        возвращает self. Иначе повторяет элементы вдоль размерностей размером 1.

        Также поддерживает добавление измерений размером 1 в начало (для выравнивания ndim).
        """

        if self.shape == target_shape:
            return self

        # Если нужно добавить измерения в начало
        if len(self.shape) < len(target_shape):
            # Добавляем измерения размером 1 в начало
            new_shape = (1,) * (len(target_shape) - len(self.shape)) + self.shape
            if new_shape != target_shape:
                # Теперь рекурсивно вызываем с расширенной формой
                padded_tensor = self.reshape(new_shape)
                return padded_tensor._broadcast_to(target_shape)
            else:
                # Уже правильная форма после padding
                return self.reshape(new_shape)

        if len(self.shape) > len(target_shape):
            raise ValueError(
                f"Cannot broadcast tensor of shape {self.shape} to shape {target_shape}: "
                f"cannot reduce dimensions"
            )

        # Теперь len(self.shape) == len(target_shape)
        # Проверяем, что каждое измерение либо совпадает, либо текущее измерение = 1
        for i, (cur_dim, tgt_dim) in enumerate(zip(self.shape, target_shape)):
            if cur_dim != tgt_dim and cur_dim != 1:
                raise ValueError(
                    f"Cannot broadcast tensor of shape {self.shape} to shape {target_shape}: "
                    f"dimension {i} mismatch ({cur_dim} vs {tgt_dim})"
                )

        # Если уже нужная форма, вернуть self
        if self.shape == target_shape:
            return self

        # Расширяем данные, повторяя элементы вдоль размерностей размером 1
        new_data: list[Value[T]] = []

        # Вычисляем шаги для итерации по текущему тензору
        strides: list[int] = []
        stride = 1
        for dim in reversed(self.shape):
            strides.insert(0, stride)
            stride *= dim

        target_strides: list[int] = []
        stride = 1
        for dim in reversed(target_shape):
            target_strides.insert(0, stride)
            stride *= dim

        # Для каждого индекса в target_shape, находим соответствующий индекс в self
        target_size = self.get_total_size(target_shape)
        for flat_idx in range(target_size):
            # Преобразуем flat_idx в многомерный индекс target_shape
            target_idx: list[int] = []
            remaining = flat_idx
            for stride in target_strides:
                target_idx.append(remaining // stride)
                remaining %= stride

            # Для каждого измерения с размером 1 в текущей форме, используем индекс 0
            self_idx: list[int] = []
            for i, (cur_dim, tgt_idx) in enumerate(zip(self.shape, target_idx)):
                if cur_dim == 1:
                    self_idx.append(0)
                else:
                    self_idx.append(tgt_idx)

            # Вычисляем flat индекс в текущем тензоре
            self_flat_idx = 0
            for stride, idx in zip(strides, self_idx):
                self_flat_idx += idx * stride

            new_data.append(self._data[self_flat_idx])

        return Tensor(new_data, target_shape)

    def __add__(self: Self, other: Union["Tensor[T]", Value[T]]) -> "Tensor[T]":
        """Поэлементное суммирование"""

        if isinstance(other, Value):
            new_data = [x + other for x in self._data]
            return Tensor(new_data, self._shape)

        new_a, new_b = self._broadcast_both_tensors(self, other)
        new_data = [x + y for x, y in zip(new_a._data, new_b._data)]
        return Tensor(new_data, new_a._shape)

    def __sub__(self: Self, other: Union["Tensor[T]", Value[T]]) -> "Tensor[T]":
        """Поэлементное вычитание"""

        if isinstance(other, Value):
            new_data = [x - other for x in self._data]
            return Tensor(new_data, self._shape)

        new_a, new_b = self._broadcast_both_tensors(self, other)
        new_data = [x - y for x, y in zip(new_a._data, new_b._data)]
        return Tensor(new_data, new_a._shape)

    def __mul__(self: Self, other: Union["Tensor[T]", Value[T]]) -> "Tensor[T]":
        """Поэлементное умножение"""

        if isinstance(other, Value):
            new_data = [x * other for x in self._data]
            return Tensor(new_data, self._shape)

        new_a, new_b = self._broadcast_both_tensors(self, other)
        new_data = [x * y for x, y in zip(new_a._data, new_b._data)]
        return Tensor(new_data, new_a._shape)

    def __truediv__(self: Self, other: Union["Tensor[T]", Value[T]]) -> "Tensor[T]":
        """Поэлементное деление"""

        if isinstance(other, Value):
            new_data = [x / other for x in self._data]
            return Tensor(new_data, self._shape)

        new_a, new_b = self._broadcast_both_tensors(self, other)
        new_data = [x / y for x, y in zip(new_a._data, new_b._data)]
        return Tensor(new_data, new_a._shape)

    def __neg__(self: Self) -> "Tensor[T]":
        new_data = [-x for x in self._data]
        return Tensor(new_data, self._shape)

    def __matmul__(self: Self, other: "Tensor[T]") -> "Tensor[T]":

        if self.ndim != 2 or other.ndim != 2:
            raise ValueError("__matmul__ currently only supports 2D tensors")
        m, n = self._shape
        n2, p = other._shape
        if n != n2:
            raise ValueError(
                f"Incompatible shapes for matmul: {self._shape} and {other._shape}"
            )

        # Compute result matrix
        result_data = []
        for i in range(m):
            for j in range(p):
                # Dot product of row i of self and column j of other
                dot: Value[T] = Value.from_float(0, self._backend)
                for k in range(n):
                    dot += self._data[i * n + k] * other._data[k * p + j]
                result_data.append(dot)
        return Tensor(result_data, (m, p))

    def sum(self: Self) -> Value[T]:
        """Сумма всех элементов тензора"""

        total: Value[T] = Value.from_float(0, self._backend)
        for v in self._data:
            total += v
        return total

    def sum_dim(self: Self, dim: int) -> "Tensor[T]":
        """Сумма всех элементов по измерению"""

        if dim < 0 or dim >= self.ndim:
            raise ValueError(f"Dimension {dim} out of range for shape {self._shape}")

        # Сумма по измерению
        new_shape = list(self._shape)
        new_shape.pop(dim)
        new_shape_tuple = tuple(new_shape)

        # Шаг
        stride = 1
        for i in range(dim + 1, self.ndim):
            stride *= self._shape[i]

        outer = self.size // (self._shape[dim] * stride)

        new_data = []
        for i in range(outer):
            for j in range(stride):
                s = Value.from_float(0, self._backend)
                base = i * self._shape[dim] * stride + j
                for k in range(self._shape[dim]):
                    s += self._data[base + k * stride]
                new_data.append(s)

        return Tensor(new_data, new_shape_tuple)

    def mean(self: Self) -> Value[T]:
        """Среднее всех элементов тензора"""

        total = self.sum()
        return total / Value(self._backend.from_float(float(self.size)))

    def mean_dim(self: Self, dim: int) -> "Tensor[T]":
        """Среднее всех элементов по измерению"""

        summed = self.sum_dim(dim)
        factor = Value(self._backend.from_float(float(self._shape[dim])))
        return summed / factor

    def apply_activation(self: Self, activation: type[Op[T]]) -> "Tensor[T]":
        # Create a new data list so we don't mutate the original tensor's values
        new_data: list[Value[T]] = [
            self._data[i].apply_activation(activation) for i in range(len(self._data))
        ]
        return Tensor(new_data, self.shape)
