from functools import reduce
from typing import List, Tuple
from n4.core import Value
from n4.core.numeric import NumericProtocol
# TODO: sketch, full refactor required

class Tensor[T: NumericProtocol]:
    """Multi-dimensional array of Value[T] elements.

    The tensor stores its data in a flat list and interprets it according to
    its shape. All operations preserve the backend and never mix numeric types.

    Args:
        data: Flat list of Value[T] elements.
        shape: Tuple of integers specifying the tensor dimensions.
        backend: The numeric backend class used to create the values.

    Raises:
        ValueError: If the product of the shape does not equal the length of data.
    """

    def __init__(self, data: List[Value[T]], shape: Tuple[int, ...], backend: type[T]):
        expected = reduce(operator.mul, shape, 1)
        if len(data) != expected:
            raise ValueError(
                f"Data length {len(data)} does not match shape {shape} (expected {expected})"
            )
        self._data = data
        self._shape = shape
        self._backend = backend

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of the tensor."""
        return self._shape

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return len(self._shape)

    @property
    def size(self) -> int:
        """Total number of elements."""
        return len(self._data)

    @property
    def backend(self) -> type[T]:
        """Numeric backend used by this tensor."""
        return self._backend

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------
    @classmethod
    def zeros(cls, shape: Tuple[int, ...], backend: type[T]) -> "Tensor[T]":
        """Create a tensor filled with zeros."""
        total = reduce(operator.mul, shape, 1)
        data = [backend.zero() for _ in range(total)]
        return cls(data, shape, backend)

    @classmethod
    def ones(cls, shape: Tuple[int, ...], backend: type[T]) -> "Tensor[T]":
        """Create a tensor filled with ones."""
        total = reduce(operator.mul, shape, 1)
        data = [backend.one() for _ in range(total)]
        return cls(data, shape, backend)

    @classmethod
    def random_uniform(
        cls, shape: Tuple[int, ...], low: float = -1.0, high: float = 1.0, backend: type[T] = None
    ) -> "Tensor[T]":
        """Create a tensor filled with random values from a uniform distribution."""
        if backend is None:
            raise ValueError("A backend must be provided.")
        total = reduce(operator.mul, shape, 1)
        data = [backend.random_uniform(low, high) for _ in range(total)]
        return cls(data, shape, backend)

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------
    def to_list(self) -> List[Any]:
        """Convert the tensor to a nested Python list (for debugging)."""
        if self.ndim == 0:
            return self._data[0]  # type: ignore
        # Recursively build nested lists
        step = self.size // self._shape[0]
        return [
            Tensor(self._data[i * step : (i + 1) * step], self._shape[1:], self._backend).to_list()
            for i in range(self._shape[0])
        ]

    # ------------------------------------------------------------------
    # Indexing (simplified, returns a copy)
    # ------------------------------------------------------------------
    def __getitem__(self, idx: Union[int, Tuple[int, ...]]) -> Union[Value[T], "Tensor[T]"]:
        """Index into the tensor.

        Supports integer and tuple indexing. If the index specifies all dimensions,
        a single Value[T] is returned. Otherwise, a new tensor (copy) with the
        remaining dimensions is returned.

        Args:
            idx: An integer or a tuple of integers.

        Returns:
            A Value[T] or a Tensor[T] view.
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
        step = reduce(operator.mul, new_shape, 1)
        data_slice = self._data[flat : flat + step]
        return Tensor(data_slice, new_shape, self._backend)

    # ------------------------------------------------------------------
    # Reshape
    # ------------------------------------------------------------------
    def reshape(self, new_shape: Tuple[int, ...]) -> "Tensor[T]":
        """Return a new tensor with the same data but a new shape."""
        expected = reduce(operator.mul, new_shape, 1)
        if expected != self.size:
            raise ValueError(
                f"Cannot reshape tensor of size {self.size} into shape {new_shape}"
            )
        return Tensor(self._data, new_shape, self._backend)

    # ------------------------------------------------------------------
    # Broadcasting helpers
    # ------------------------------------------------------------------
    def _broadcast_shapes(self, other: "Tensor[T]") -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        """Compute broadcasted shapes for two tensors (simplified, no alignment)."""
        # Real implementation would align shapes from the right; we keep it simple.
        # For our purposes (element-wise ops with same ndim) we assume they are already compatible.
        if self.ndim != other.ndim:
            raise NotImplementedError("Broadcasting with different numbers of dimensions")
        new_shape = []
        for i, (d1, d2) in enumerate(zip(self._shape, other._shape)):
            if d1 == d2 or d1 == 1 or d2 == 1:
                new_shape.append(max(d1, d2))
            else:
                raise ValueError(f"Incompatible shapes for broadcasting: {self._shape} and {other._shape}")
        return tuple(new_shape), tuple(new_shape)  # both become same shape after broadcast

    def _broadcast_to(self, target_shape: Tuple[int, ...]) -> "Tensor[T]":
        """Return a new tensor broadcasted to target_shape (simplified, assumes compatible)."""
        if self._shape == target_shape:
            return self
        # In a real implementation we would expand dimensions and repeat data.
        # For now we only handle the case where the shapes are already compatible
        # and the current shape can be expanded by repeating elements.
        # This is a placeholder – a full broadcasting implementation is complex.
        raise NotImplementedError("Full broadcasting not implemented in this example")

    # ------------------------------------------------------------------
    # Element-wise arithmetic
    # ------------------------------------------------------------------
    def __add__(self, other: Union["Tensor[T]", Value[T]]) -> "Tensor[T]":
        """Element-wise addition with broadcasting."""
        if isinstance(other, Value):
            # Scalar addition: add the same value to every element
            new_data = [x + other for x in self._data]
            return Tensor(new_data, self._shape, self._backend)
        # Tensor + Tensor
        # For simplicity, assume same shape (no broadcasting)
        if self._shape != other._shape:
            raise NotImplementedError("Broadcasting not implemented in this example")
        new_data = [x + y for x, y in zip(self._data, other._data)]
        return Tensor(new_data, self._shape, self._backend)

    def __sub__(self, other: Union["Tensor[T]", Value[T]]) -> "Tensor[T]":
        if isinstance(other, Value):
            new_data = [x - other for x in self._data]
            return Tensor(new_data, self._shape, self._backend)
        if self._shape != other._shape:
            raise NotImplementedError("Broadcasting not implemented")
        new_data = [x - y for x, y in zip(self._data, other._data)]
        return Tensor(new_data, self._shape, self._backend)

    def __mul__(self, other: Union["Tensor[T]", Value[T]]) -> "Tensor[T]":
        if isinstance(other, Value):
            new_data = [x * other for x in self._data]
            return Tensor(new_data, self._shape, self._backend)
        if self._shape != other._shape:
            raise NotImplementedError("Broadcasting not implemented")
        new_data = [x * y for x, y in zip(self._data, other._data)]
        return Tensor(new_data, self._shape, self._backend)

    def __truediv__(self, other: Union["Tensor[T]", Value[T]]) -> "Tensor[T]":
        if isinstance(other, Value):
            new_data = [x / other for x in self._data]
            return Tensor(new_data, self._shape, self._backend)
        if self._shape != other._shape:
            raise NotImplementedError("Broadcasting not implemented")
        new_data = [x / y for x, y in zip(self._data, other._data)]
        return Tensor(new_data, self._shape, self._backend)

    def __neg__(self) -> "Tensor[T]":
        new_data = [-x for x in self._data]
        return Tensor(new_data, self._shape, self._backend)

    # ------------------------------------------------------------------
    # Matrix multiplication (2D only, no batching)
    # ------------------------------------------------------------------
    def __matmul__(self, other: "Tensor[T]") -> "Tensor[T]":
        """Matrix multiplication for 2D tensors.

        Assumes self is shape (m, n) and other is shape (n, p). Returns (m, p).
        """
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
        return Tensor(result_data, (m, p), self._backend)

    # ------------------------------------------------------------------
    # Reductions
    # ------------------------------------------------------------------
    def sum(self, dim: Optional[int] = None) -> Union[Value[T], "Tensor[T]"]:
        """Sum over all elements or along a single dimension."""
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

        block = stride
        outer = self.size // (self._shape[dim] * stride)

        new_data = []
        for i in range(outer):
            for j in range(stride):
                s = self._backend.zero()
                base = i * self._shape[dim] * stride + j
                for k in range(self._shape[dim]):
                    s += self._data[base + k * stride]
                new_data.append(s)

        return Tensor(new_data, new_shape, self._backend)

    def mean(self, dim: Optional[int] = None) -> Union[Value[T], "Tensor[T]"]:
        """Mean over all elements or along a single dimension."""
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

