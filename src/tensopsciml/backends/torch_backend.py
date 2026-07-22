"""PyTorch implementation of the :class:`~tensopsciml.backends.base.Backend` protocol."""

from typing import Any, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from .base import Backend


class TorchBackend(Backend):
    """Backend implementation powered by PyTorch tensors."""

    # Data type attributes.
    float32 = torch.float32
    float64 = torch.float64

    @staticmethod
    def _to_tensor(value: Any) -> Tensor:
        """Convert a value to a torch.Tensor when necessary.

        Args:
            value: Object that may already be a tensor or convertible to one.

        Returns:
            Tensor: Tensor representation of the value.
        """

        return value if isinstance(value, torch.Tensor) else torch.as_tensor(value)

    @staticmethod
    def _normalize_padding(
        padding: str | int | tuple[int, ...],
    ) -> str | int | tuple[int, ...]:
        """Normalize textual padding values to PyTorch-compatible options.

        Args:
            padding: Padding specification provided to convolution routines.

        Returns:
            str | int | tuple[int, ...]: Padding value accepted by PyTorch.

        Raises:
            ValueError: If padding is a string other than "valid" or "same".
        """

        if isinstance(padding, str):
            padding_lower = padding.lower()
            if padding_lower == "valid":
                return 0
            if padding_lower == "same":
                return "same"
            raise ValueError(f"Unsupported padding mode '{padding}'.")
        return padding

    def tensor(
        self,
        data: Any,
        *,
        dtype: Any | None = None,
        device: Any | None = None,
        requires_grad: bool = False,
    ) -> Tensor:
        """Create a tensor from arbitrary data.

        Args:
            data: Input compatible with torch.as_tensor.
            dtype: Target data type. If None, preserves the input dtype.
            device: Target device. If None, preserves the current device.
            requires_grad: Whether the tensor tracks gradients.

        Returns:
            Tensor: PyTorch tensor with the requested type and device.
        """

        tensor = self._to_tensor(data)
        if dtype is not None:
            tensor = tensor.to(dtype=dtype)
        if device is not None:
            tensor = tensor.to(device=device)
        if requires_grad:
            tensor.requires_grad_(True)
        return tensor

    def clone(self, tensor: Any) -> Tensor:
        """Create a detached copy of a tensor.

        Args:
            tensor: Tensor to duplicate.

        Returns:
            Tensor: Cloned tensor detached from the original computation graph.
        """

        tensor_obj = self._to_tensor(tensor)
        return tensor_obj.clone()

    def einsum(self, subscripts: str, *operands: Any) -> Tensor:
        """Evaluate Einstein summation using PyTorch.

        Args:
            subscripts: Einsum notation describing the operation.
            *operands: Input operands for the einsum expression.

        Returns:
            Tensor: Result of the einsum expression.
        """

        tensors = [self._to_tensor(operand) for operand in operands]
        return torch.einsum(subscripts, *tensors)

    def equal(self, a: Any, b: Any) -> bool:
        """Compare tensors for exact equality.

        Args:
            a: First tensor to compare.
            b: Second tensor to compare.

        Returns:
            bool: True if the tensors share identical values.
        """

        return torch.equal(self._to_tensor(a), self._to_tensor(b))

    def det(self, matrix: Any) -> Tensor:
        """Compute the determinant of a tensor.

        Args:
            matrix: Tensor representing a square matrix.

        Returns:
            Tensor: Determinant of the matrix tensor.
        """

        tensor = self._to_tensor(matrix)
        return torch.linalg.det(tensor)

    def inv(self, matrix: Any) -> Tensor:
        """Compute the inverse of a tensor.

        Args:
            matrix: Tensor representing a square matrix.

        Returns:
            Tensor: Inverted matrix tensor.
        """

        tensor = self._to_tensor(matrix)
        return torch.linalg.inv(tensor)

    def transpose(self, matrix: Any, dims: tuple[int, ...]) -> Tensor:
        """Transpose tensor dimensions.

        Args:
            matrix: Tensor to transpose.
            dims: Pair specifying the dimensions to swap.

        Returns:
            Tensor: Transposed tensor.
        """

        tensor = self._to_tensor(matrix)
        return torch.transpose(tensor, *dims)

    def permute(self, tensor: Any, dims: Sequence[int]) -> Tensor:
        """Permute tensor dimensions.

        Args:
            tensor: Tensor to permute.
            dims: Target ordering of the dimensions.

        Returns:
            Tensor: Permuted tensor.
        """

        tensor_obj = self._to_tensor(tensor)
        return tensor_obj.permute(*dims)

    def unsqueeze(self, tensor: Any, dim: int) -> Tensor:
        """Insert a singleton dimension.

        Args:
            tensor: Tensor to expand.
            dim: Position for the new dimension.

        Returns:
            Tensor: Tensor with the inserted dimension.
        """

        tensor_obj = self._to_tensor(tensor)
        return tensor_obj.unsqueeze(dim)

    def squeeze(self, tensor: Any, dim: int | None = None) -> Tensor:
        """Remove singleton dimensions, optionally along a specific axis.

        Args:
            tensor: Tensor to squeeze.
            dim: Optional axis to squeeze.

        Returns:
            Tensor: Tensor with singleton dimensions removed.
        """

        tensor_obj = self._to_tensor(tensor)
        return tensor_obj.squeeze(dim) if dim is not None else tensor_obj.squeeze()

    def stack(self, tensors: Sequence[Any], dim: int) -> Tensor:
        """Stack tensors along a new dimension.

        Args:
            tensors: Iterable of tensors to stack.
            dim: Dimension index for the new axis.

        Returns:
            Tensor: Stacked tensor.
        """

        converted = [self._to_tensor(item) for item in tensors]
        return torch.stack(converted, dim=dim)

    def to(
        self,
        tensor: Any,
        *,
        dtype: Any | None = None,
        device: Any | None = None,
    ) -> Tensor:
        """Move a tensor to the requested dtype or device.

        Args:
            tensor: Tensor to convert.
            dtype: Optional target dtype.
            device: Optional target device.

        Returns:
            Tensor: Converted tensor.
        """

        tensor_obj = self._to_tensor(tensor)
        return tensor_obj.to(dtype=dtype, device=device)

    def zeros_like(
        self,
        tensor: Any,
        *,
        dtype: Any | None = None,
        device: Any | None = None,
        requires_grad: bool = False,
    ) -> Tensor:
        """Create a zero tensor shaped like an existing tensor.

        Args:
            tensor: Reference tensor that provides the target shape.
            dtype: Optional target dtype.
            device: Optional target device.
            requires_grad: Whether the tensor tracks gradients.

        Returns:
            Tensor: Tensor filled with zeros.
        """

        tensor_obj = self._to_tensor(tensor)
        return torch.zeros_like(
            tensor_obj,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )

    def zeros(
        self,
        shape: Sequence[int],
        *,
        dtype: Any | None = None,
        device: Any | None = None,
        requires_grad: bool = False,
    ) -> Tensor:
        """Create a zero tensor with a specific shape.

        Args:
            shape: Shape specification for the tensor.
            dtype: Optional target dtype.
            device: Optional target device.
            requires_grad: Whether the tensor tracks gradients.

        Returns:
            Tensor: Tensor filled with zeros.
        """

        return torch.zeros(
            tuple(shape),
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )

    def ones(
        self,
        shape: Sequence[int],
        *,
        dtype: Any | None = None,
        device: Any | None = None,
        requires_grad: bool = False,
    ) -> Tensor:
        """Create a one tensor with a specific shape.

        Args:
            shape: Shape specification for the tensor.
            dtype: Optional target dtype.
            device: Optional target device.
            requires_grad: Whether the tensor tracks gradients.

        Returns:
            Tensor: Tensor filled with ones.
        """

        return torch.ones(
            tuple(shape),
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )

    def eye(
        self,
        n: int,
        m: int | None = None,
        *,
        dtype: Any | None = None,
        device: Any | None = None,
        requires_grad: bool = False,
    ) -> Tensor:
        """Create a 2D tensor with ones on the diagonal and zeros elsewhere.

        Args:
            n: Number of rows.
            m: Optional number of columns; defaults to n.
            dtype: Target data type.
            device: Target device.
            requires_grad: Whether the tensor tracks gradients.

        Returns:
            Tensor: Identity matrix or tensor with diagonal ones.
        """
        if m is None:
            return torch.eye(
                n,
                dtype=dtype,
                device=device,
                requires_grad=requires_grad,
            )
        return torch.eye(
            n,
            m,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )

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
    ) -> Tensor:
        """Apply a one-dimensional convolution.

        Args:
            input: Input tensor supporting conversion to torch.Tensor.
            weight: Convolution kernel tensor.
            bias: Optional bias tensor added to the output channels.
            stride: Stride of the convolution.
            padding: Padding specification; accepts valid, same, integers,
                or tuples.
            dilation: Spacing between kernel elements.
            groups: Number of blocked connections from input channels
                to output channels.

        Returns:
            Tensor: Result of torch.nn.functional.conv1d.

        Raises:
            ValueError: If padding is a string other than valid or same.
        """
        input_tensor = self._to_tensor(input)
        weight_tensor = self._to_tensor(weight)
        bias_tensor = None if bias is None else self._to_tensor(bias)
        padding_value = self._normalize_padding(padding)

        return F.conv1d(
            input_tensor,
            weight_tensor,
            bias=bias_tensor,
            stride=stride,
            padding=padding_value,
            dilation=dilation,
            groups=groups,
        )

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
    ) -> Tensor:
        """Apply a two-dimensional convolution.

        Args:
            input: Input tensor supporting conversion to torch.Tensor.
            weight: Convolution kernel tensor.
            bias: Optional bias tensor added to the output channels.
            stride: Stride of the convolution along each spatial dimension.
            padding: Padding specification; accepts valid, same, integers,
                or tuples of two integers.
            dilation: Spacing between kernel elements along each spatial dimension.
            groups: Number of blocked connections from input channels
                to output channels.

        Returns:
            Tensor: Result of torch.nn.functional.conv2d.

        Raises:
            ValueError: If padding is a string other than valid or same.
        """
        input_tensor = self._to_tensor(input)
        weight_tensor = self._to_tensor(weight)
        bias_tensor = None if bias is None else self._to_tensor(bias)
        padding_value = self._normalize_padding(padding)

        return F.conv2d(
            input_tensor,
            weight_tensor,
            bias=bias_tensor,
            stride=stride,
            padding=padding_value,
            dilation=dilation,
            groups=groups,
        )

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
    ) -> Tensor:
        """Apply a three-dimensional convolution.

        Args:
            input: Input tensor supporting conversion to torch.Tensor.
            weight: Convolution kernel tensor.
            bias: Optional bias tensor added to the output channels.
            stride: Stride of the convolution along each spatial dimension.
            padding: Padding specification; accepts valid, same, integers,
                or tuples of three integers.
            dilation: Spacing between kernel elements along each spatial dimension.
            groups: Number of blocked connections from input channels
                to output channels.

        Returns:
            Tensor: Result of torch.nn.functional.conv3d.

        Raises:
            ValueError: If padding is a string other than valid or same.
        """
        input_tensor = self._to_tensor(input)
        weight_tensor = self._to_tensor(weight)
        bias_tensor = None if bias is None else self._to_tensor(bias)
        padding_value = self._normalize_padding(padding)

        return F.conv3d(
            input_tensor,
            weight_tensor,
            bias=bias_tensor,
            stride=stride,
            padding=padding_value,
            dilation=dilation,
            groups=groups,
        )

    def autograd(
        self,
        y: Any,
        x: Any,
        *,
        grad_outputs: Any | None = None,
        create_graph: bool = False,
        retain_graph: bool = False,
        allow_unused: bool = False,
    ) -> Tensor:
        """Evaluate a vector-Jacobian product via PyTorch autograd.

        Args:
            y: Output tensor to differentiate.
            x: Input tensor with respect to which the gradient is computed.
            grad_outputs: Weighting tensor matching y's shape. Defaults to a
                tensor of ones, giving d(sum(y))/dx.
            create_graph: Whether to construct higher-order graphs.
            retain_graph: Whether to retain the graph after gradient computation.
            allow_unused: Whether to allow tensors that do not contribute to y.

        Returns:
            Tensor: Vector-Jacobian product of y with respect to x, shaped
                like x.
        """

        y_tensor = self._to_tensor(y)
        x_tensor = self._to_tensor(x)
        weights = (
            self._to_tensor(grad_outputs)
            if grad_outputs is not None
            else torch.ones_like(y_tensor)
        )
        gradient = torch.autograd.grad(
            outputs=y_tensor,
            inputs=x_tensor,
            grad_outputs=weights,
            create_graph=create_graph,
            retain_graph=retain_graph,
            allow_unused=allow_unused,
        )
        return gradient[0]

    def requires_grad_(self, tensor: Any, requires_grad: bool = True) -> Tensor:
        """Toggle gradient tracking on a tensor in place.

        Non-leaf tensors (e.g. the result of ``clone`` on a tracked tensor)
        reject ``Tensor.requires_grad_``: if the requested state already
        holds the tensor is returned untouched, and disabling tracking on a
        non-leaf returns a detached view instead of raising.

        Args:
            tensor: Tensor whose gradient tracking state is toggled.
            requires_grad: Whether to enable gradient tracking.

        Returns:
            Tensor: Tensor with updated gradient tracking state.
        """

        tensor_obj = self._to_tensor(tensor)
        if tensor_obj.requires_grad == requires_grad:
            return tensor_obj
        if not tensor_obj.is_leaf:
            # Only reachable when disabling tracking (non-leaf tensors always
            # have requires_grad=True); detach shares storage with the input.
            return tensor_obj.detach()
        tensor_obj.requires_grad_(requires_grad)
        return tensor_obj

    def allclose(
        self,
        a: Any,
        b: Any,
        rtol: float = 1e-5,
        atol: float = 1e-8,
    ) -> bool:
        """Check whether all elements of two tensors are close.

        Args:
            a: First tensor to compare.
            b: Second tensor to compare.
            rtol: Relative tolerance parameter.
            atol: Absolute tolerance parameter.

        Returns:
            bool: True if all elements are close within the tolerances.
        """
        a_tensor = self._to_tensor(a)
        b_tensor = self._to_tensor(b)
        return torch.allclose(
            a_tensor,
            b_tensor,
            rtol=rtol,
            atol=atol,
        )
