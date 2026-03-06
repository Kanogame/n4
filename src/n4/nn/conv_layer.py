from n4.tensor import Tensor
from n4.core import Op, Value
from typing import Optional
from n4.core.numeric import NumericProtocol
from .layer import Layer


# TODO: requres refactoring
class ConvLayer[T: NumericProtocol](Layer[T]):
    """2D convolutional layer using tensor operations.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels (filters).
        kernel_size: Side length of the square kernel.
        stride: Convolution stride.
        padding: Zero padding added to both sides.
        activation: Activation function.
        backend: Numeric backend class.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        activation: Optional[type[Op[T]]] = None,
    ) -> None:
        super().__init__(self)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Each filter is a flattened kernel: in_channels * kernel_size * kernel_size
        filter_len = in_channels * kernel_size * kernel_size
        self.weights = Tensor.random_uniform(
            (out_channels, filter_len), backend=self._backend, low=-1.0, high=1.0
        )
        self.bias = Tensor.zeros((out_channels,), backend=self._backend)

        self.activation = self.resolve_activation(activation)

    def _pad(self, x: Tensor[T]) -> Tensor[T]:
        """Apply zero padding to a 3D tensor (channels, height, width)."""
        if self.padding == 0:
            return x
        c, h, w = x.shape
        new_h = h + 2 * self.padding
        new_w = w + 2 * self.padding
        # Create a zero tensor of the new shape
        padded: Tensor[T] = Tensor.zeros((c, new_h, new_w), self._backend)
        # Copy original data into the center
        for ci in range(c):
            for i in range(h):
                for j in range(w):
                    padded[ci, i + self.padding, j + self.padding] = x[ci, i, j]
        return padded

    def _im2col(self, x: Tensor[T]) -> Tensor[T]:
        """Extract sliding patches as columns.

        Input: 3D tensor (channels, height, width)
        Output: 2D tensor (num_patches, filter_len)
        """
        c, h, w = x.shape
        k = self.kernel_size
        s = self.stride

        out_h = (h - k) // s + 1
        out_w = (w - k) // s + 1
        num_patches = out_h * out_w
        filter_len = c * k * k

        patches = []
        for i in range(0, h - k + 1, s):
            for j in range(0, w - k + 1, s):
                patch = []
                for ci in range(c):
                    for ki in range(k):
                        for kj in range(k):
                            patch.append(x[ci, i + ki, j + kj])
                patches.extend(patch)  # flat list of all patches concatenated
        # Reshape to (num_patches, filter_len)
        return Tensor[T](patches, (num_patches, filter_len))

    def forward(self, x: Tensor[T]) -> Tensor[T]:
        """Apply convolution to a 3D input tensor (channels, height, width).

        Returns a 3D tensor (out_channels, out_height, out_width).
        """
        if x.ndim != 3:
            raise ValueError(f"ConvLayer expects 3D input (C, H, W), got {x.ndim}D")
        if x.shape[0] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels, got {x.shape[0]}")

        # Pad input
        padded = self._pad(x)
        # Convert to columns: (num_patches, filter_len)
        cols = self._im2col(padded)

        # Apply all filters at once: cols @ weights.T  -> (num_patches, out_channels)
        # weights shape: (out_channels, filter_len)
        # We'll use manual batched matmul similar to DenseLayer
        num_patches, filter_len = cols.shape
        out_channels, _ = self.weights.shape

        out_data = []
        for p in range(num_patches):
            # Take patch p (1D tensor of length filter_len)
            patch = Tensor[T](
                cols._data[p * filter_len : (p + 1) * filter_len], (filter_len,)
            )
            out_row = []
            for oc in range(out_channels):
                # weights[oc, :] is a 1D tensor of length filter_len
                w_row = Tensor[T](
                    self.weights._data[oc * filter_len : (oc + 1) * filter_len],
                    (filter_len,),
                )
                # Dot product
                prod = patch * w_row
                dot = prod.sum()
                out_row.append(dot + self.bias._data[oc])
            out_data.extend(out_row)

            activated = [v.apply_activation(self.activation) for v in out_data]

        # Determine output spatial dimensions
        h_in = x.shape[1]
        w_in = x.shape[2]
        h_out = (h_in + 2 * self.padding - self.kernel_size) // self.stride + 1
        w_out = (w_in + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_shape = (self.out_channels, h_out, w_out)
        return Tensor[T](activated, out_shape)

    def parameters(self) -> list[Value[T]]:
        return self.weights._data + self.bias._data
