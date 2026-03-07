from n4.core import Value
from n4.tensor import Tensor
from n4.numeric import NumericProtocol
from .layer import Layer


# TODO: requires refactoring
# class SoftmaxLayer[T: NumericProtocol](Layer[T]):
#    """Softmax layer applied along the last dimension."""
#
#    def forward(self, x: Tensor[T]) -> Tensor[T]:
#        """Apply softmax to the last dimension.
#
#        Args:
#            x: Input tensor of any shape. The softmax is computed over the last axis.
#
#        Returns:
#            Tensor of the same shape with values in [0,1] summing to 1 along the last axis.
#        """
#        # Flatten all dimensions except the last
#        original_shape = x.shape
#        last_dim = original_shape[-1]
#        # Reshape to 2D: (num_rows, last_dim)
#        num_rows = x.size // last_dim
#        x_2d = x.reshape((num_rows, last_dim))
#
#        result_data = []
#        for i in range(num_rows):
#            # Extract row i (1D tensor of length last_dim)
#            row_data = x_2d._data[i * last_dim : (i + 1) * last_dim]
#            # Numerical stability: subtract max
#            max_val = max(
#                v.data for v in row_data
#            )  # Access underlying data for comparison
#            # Compute exponentials
#            exps = [
#                (v - Value(self._backend.from_float(max_val))).exp() for v in row_data
#            ]
#            sum_exp = self._backend.zero()
#            for e in exps:
#                sum_exp += e
#            # Normalize
#            probs = [e / sum_exp for e in exps]
#            result_data.extend(probs)
#
#        # Restore original shape
#        return Tensor[T](result_data, original_shape)
#
#    def parameters(self) -> list[Value[T]]:
#        return []
#
