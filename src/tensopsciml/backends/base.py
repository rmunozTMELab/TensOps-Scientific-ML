"""Backend protocol defining the minimal tensor operations TensOps needs."""

from typing import Any, Protocol, Sequence


class Backend(Protocol):
    """Protocol describing the minimal backend surface."""

    # Data type attributes.
    float32: Any
    float64: Any

    def tensor(
        self,
        data: Any,
        *,
        dtype: Any | None = None,
        device: Any | None = None,
        requires_grad: bool = False,
    ) -> Any:
        """Create a backend-specific tensor from arbitrary data.

        Args:
            data (Any): Structure convertible to a backend tensor.
            dtype (Any | None): Optional target dtype for the resulting tensor.
            device (Any | None): Optional target device for the resulting
                tensor.
            requires_grad (bool): Whether the resulting tensor tracks gradients.

        Returns:
            Any: Backend-specific tensor instance.
        """

    def clone(self, tensor: Any) -> Any:
        """Create a backend-specific copy of a tensor.

        Args:
            tensor (Any): Tensor to duplicate.

        Returns:
            Any: Cloned tensor.
        """

    def einsum(self, subscripts: str, *operands: Any) -> Any:
        """Evaluate Einstein summation for the given operands.

        Args:
            subscripts (str): Einsum notation describing the contraction.
            *operands (Any): Input tensors for the operation.

        Returns:
            Any: Result of the einsum expression.
        """

    def equal(self, a: Any, b: Any) -> bool:
        """Check whether two tensors are exactly equal.

        Args:
            a (Any): First tensor to compare.
            b (Any): Second tensor to compare.

        Returns:
            bool: True if both tensors contain identical values.
        """

    def det(self, matrix: Any) -> Any:
        """Compute the determinant of a matrix tensor.

        Args:
            matrix (Any): Tensor representing a square matrix.

        Returns:
            Any: Determinant of the matrix.
        """

    def inv(self, matrix: Any) -> Any:
        """Compute the inverse of a matrix tensor.

        Args:
            matrix (Any): Tensor representing a square matrix.

        Returns:
            Any: Inverted matrix tensor.
        """

    def transpose(self, matrix: Any, dims: tuple[int, ...]) -> Any:
        """Transpose tensor dimensions according to the provided order.

        Args:
            matrix (Any): Tensor to transpose.
            dims (tuple[int, ...]): Target dimension order.

        Returns:
            Any: Transposed tensor.
        """

    def autograd(
        self,
        y: Any,
        x: Any,
        *,
        grad_outputs: Any | None = None,
        create_graph: bool = False,
        retain_graph: bool = False,
        allow_unused: bool = False,
    ) -> Any:
        """Compute the vector-Jacobian product of y with respect to x.

        Args:
            y (Any): Output tensor to differentiate.
            x (Any): Input tensor for which gradients are computed.
            grad_outputs (Any | None): Weighting tensor matching y's shape
                (the "vector" in vector-Jacobian product). Defaults to a
                tensor of ones, giving d(sum(y))/dx.
            create_graph (bool): Whether to construct higher-order graphs.
            retain_graph (bool): Whether to retain the graph after gradient
                computation.
            allow_unused (bool): Whether to allow tensors that do not
                contribute to y.

        Returns:
            Any: Vector-Jacobian product tensor, shaped like x.
        """

    def permute(self, tensor: Any, dims: Sequence[int]) -> Any:
        """Reorder tensor dimensions according to the provided sequence.

        Args:
            tensor (Any): Tensor to permute.
            dims (Sequence[int]): Target dimension order.

        Returns:
            Any: Permuted tensor.
        """

    def unsqueeze(self, tensor: Any, dim: int) -> Any:
        """Insert a singleton dimension at a specific index.

        Args:
            tensor (Any): Tensor to expand.
            dim (int): Position for the new dimension.

        Returns:
            Any: Tensor with an added dimension.
        """

    def stack(self, tensors: Sequence[Any], dim: int) -> Any:
        """Stack tensors along a new dimension.

        Args:
            tensors (Sequence[Any]): Iterable of tensors to stack.
            dim (int): Index of the new dimension.

        Returns:
            Any: Stacked tensor.
        """

    def squeeze(self, tensor: Any, dim: int | None = None) -> Any:
        """Remove singleton dimensions from a tensor.

        Args:
            tensor (Any): Tensor to squeeze.
            dim (int | None): Optional specific dimension to squeeze.

        Returns:
            Any: Tensor with singleton dimensions removed.
        """

    def to(
        self,
        tensor: Any,
        *,
        dtype: Any | None = None,
        device: Any | None = None,
    ) -> Any:
        """Move a tensor to a different dtype or device.

        Args:
            tensor (Any): Tensor to convert.
            dtype (Any | None): Optional target dtype.
            device (Any | None): Optional target device.

        Returns:
            Any: Converted tensor.
        """

    def zeros_like(
        self,
        tensor: Any,
        *,
        dtype: Any | None = None,
        device: Any | None = None,
        requires_grad: bool = False,
    ) -> Any:
        """Create a zeros tensor with the same shape as an existing tensor.

        Args:
            tensor (Any): Reference tensor that provides the target shape.
            dtype (Any | None): Optional target dtype.
            device (Any | None): Optional target device.
            requires_grad (bool): Whether the tensor tracks gradients.

        Returns:
            Any: Tensor filled with zeros.
        """

    def zeros(
        self,
        shape: Sequence[int],
        *,
        dtype: Any | None = None,
        device: Any | None = None,
        requires_grad: bool = False,
    ) -> Any:
        """Create a zeros tensor with a specific shape.

        Args:
            shape (Sequence[int]): Shape specification for the tensor.
            dtype (Any | None): Optional target dtype.
            device (Any | None): Optional target device.
            requires_grad (bool): Whether the tensor tracks gradients.

        Returns:
            Any: Tensor filled with zeros.
        """

    def ones(
        self,
        shape: Sequence[int],
        *,
        dtype: Any | None = None,
        device: Any | None = None,
        requires_grad: bool = False,
    ) -> Any:
        """Create a ones tensor with a specific shape.

        Args:
            shape (Sequence[int]): Shape specification for the tensor.
            dtype (Any | None): Optional target dtype.
            device (Any | None): Optional target device.
            requires_grad (bool): Whether the tensor tracks gradients.

        Returns:
            Any: Tensor filled with ones.
        """

    def eye(
        self,
        n: int,
        m: int | None = None,
        *,
        dtype: Any | None = None,
        device: Any | None = None,
        requires_grad: bool = False,
    ) -> Any:
        """Create a 2D tensor with ones on the diagonal and zeros elsewhere.

        Args:
            n (int): Number of rows.
            m (int | None): Optional number of columns; defaults to n.
            dtype (Any | None): Optional target dtype.
            device (Any | None): Optional target device.
            requires_grad (bool): Whether the tensor tracks gradients.

        Returns:
            Any: Tensor with ones on the main diagonal.
        """

    def conv1d(
        self,
        input: Any,
        weight: Any,
        *,
        bias: Any | None = None,
        stride: int | tuple[int, ...] = 1,
        padding: str | int | tuple[int, ...] = "valid",
        dilation: int | tuple[int, ...] = 1,
        groups: int = 1,
    ) -> Any:
        """Perform a one-dimensional convolution.

        Args:
            input (Any): Input tensor for the convolution.
            weight (Any): Convolution kernel tensor.
            bias (Any | None): Optional bias tensor.
            stride (int | tuple[int, ...]): Convolution stride or strides.
            padding (str | int | tuple[int, ...]): Padding specification.
            dilation (int | tuple[int, ...]): Dilation factor or factors.
            groups (int): Number of grouped convolutions.

        Returns:
            Any: Result of the convolution.
        """

    def conv2d(
        self,
        input: Any,
        weight: Any,
        *,
        bias: Any | None = None,
        stride: int | tuple[int, int] = 1,
        padding: str | int | tuple[int, int] = "valid",
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
    ) -> Any:
        """Perform a two-dimensional convolution.

        Args:
            input (Any): Input tensor for the convolution.
            weight (Any): Convolution kernel tensor.
            bias (Any | None): Optional bias tensor.
            stride (int | tuple[int, int]): Convolution stride or strides.
            padding (str | int | tuple[int, int]): Padding specification.
            dilation (int | tuple[int, int]): Dilation factor or factors.
            groups (int): Number of grouped convolutions.

        Returns:
            Any: Result of the convolution.
        """

    def conv3d(
        self,
        input: Any,
        weight: Any,
        *,
        bias: Any | None = None,
        stride: int | tuple[int, int, int] = 1,
        padding: str | int | tuple[int, int, int] = "valid",
        dilation: int | tuple[int, int, int] = 1,
        groups: int = 1,
    ) -> Any:
        """Perform a three-dimensional convolution.

        Args:
            input (Any): Input tensor for the convolution.
            weight (Any): Convolution kernel tensor.
            bias (Any | None): Optional bias tensor.
            stride (int | tuple[int, int, int]): Convolution stride or strides.
            padding (str | int | tuple[int, int, int]): Padding specification.
            dilation (int | tuple[int, int, int]): Dilation factor or factors.
            groups (int): Number of grouped convolutions.

        Returns:
            Any: Result of the convolution.
        """

    def requires_grad_(self, tensor: Any, requires_grad: bool = True) -> Any:
        """Enable or disable gradient tracking for a tensor in place.

        Args:
            tensor (Any): Tensor on which to toggle gradient tracking.
            requires_grad (bool): Whether to enable gradient tracking.

        Returns:
            Any: Tensor with updated gradient tracking state.
        """

    def allclose(
        self,
        a: Any,
        b: Any,
        rtol: float = 1e-5,
        atol: float = 1e-8,
    ) -> bool:
        """Check whether all elements of two tensors are close within tolerance.

        Args:
            a (Any): First tensor to compare.
            b (Any): Second tensor to compare.
            rtol (float): Relative tolerance.
            atol (float): Absolute tolerance.

        Returns:
            bool: True if all elements are close within the tolerances.
        """
