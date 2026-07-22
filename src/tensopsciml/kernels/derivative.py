"""Finite-difference convolution kernels for numerical derivatives."""

from typing import Any, Sequence

from ..backends import Backend, get_backend


class Derivative:
    """Generate finite difference kernels for numerical derivatives."""

    def __init__(
        self,
        d1: float,
        d2: float | None = None,
        d3: float | None = None,
        *,
        order: str = "1",
        indexing: str = "ij",
        device: Any = "cpu",
        backend: Backend | None = None,
        dtype: Any | None = None,
    ) -> None:
        """Initialize the Derivative kernel generator.

        Args:
            d1 (float): Grid spacing in the first direction.
            d2 (float | None): Grid spacing in the second direction. Required
                for 2D/3D kernels.
            d3 (float | None): Grid spacing in the third direction. Required
                for 3D kernels.
            order (str): Derivative order. Only first order is currently
                supported.
            indexing (str): Indexing convention (``"ij"`` or ``"xy"``).
            device (Any): Backend-specific device descriptor for the kernels.
            backend (Backend | None): Optional backend override. Defaults to
                the active backend.
            dtype (Any | None): Optional dtype override for the created
                kernels.

        Raises:
            ValueError: If an unsupported order or indexing convention is requested.
        """

        self.d1 = d1
        self.d2 = d2
        self.d3 = d3

        if order != "1":
            raise ValueError(
                "Only first-order derivative filters are currently supported.")
        if indexing not in {"ij", "xy"}:
            raise ValueError("indexing must be 'ij' or 'xy'")

        self.backend = backend or get_backend()
        self.device = device
        self.dtype = dtype
        self.order = order
        self.indexing = indexing

    def __create_kernel(self, coeffs: Sequence[Any], spacing: float) -> Any:
        """Create a backend tensor kernel for finite differences.

        Args:
            coeffs (Sequence[Any]): Nested list describing the stencil
                coefficients.
            spacing (float): Grid spacing used to normalize the stencil.

        Returns:
            Any: Backend tensor representing the finite-difference kernel.
        """

        backend_tensor_kwargs = {"device": self.device}
        if self.dtype is not None:
            backend_tensor_kwargs["dtype"] = self.dtype

        kernel = self.backend.tensor(coeffs, **backend_tensor_kwargs) / spacing

        kernel = self.backend.unsqueeze(kernel, 0)
        return self.backend.unsqueeze(kernel, 0)

    def one_dimension(self) -> Any:
        """Generate a 1D finite difference kernel.

        Returns:
            Any: A first-order finite difference kernel for 1D data.
        """
        D = self.__create_kernel([[-1, +1]], 1 * self.d1)

        return D

    def two_dimensions(self) -> Any:
        """Generate 2D finite difference kernels.

        This method returns a stack of two kernels for computing derivatives
        along the first and second dimensions (``D1``, ``D2``).

        Returns:
            Any: Kernels for the first and second directions in backend format.

        Raises:
            ValueError: If ``d2`` is not specified.

        Notes:
            The order of the returned kernels depends on the indexing
            convention, mirroring ``numpy.meshgrid``:
            - ``"ij"`` (default): ``D[0]`` differentiates along axis 0 using
              spacing ``d1``; ``D[1]`` differentiates along axis 1 using
              spacing ``d2`` (no axis swap).
            - ``"xy"``: ``D[0]`` differentiates along axis 1 using spacing
              ``d1``; ``D[1]`` differentiates along axis 0 using spacing
              ``d2`` (first two axes swapped, as in
              ``numpy.meshgrid(..., indexing="xy")``).
        """
        if self.d2 is None:
            raise ValueError(
                "The 'd2' value must be specified for 2D derivatives.")

        if self.indexing == "xy":
            D0 = self.__create_kernel([[-1, +1], [-1, +1]], 2 * self.d1)
            D1 = self.__create_kernel([[-1, -1], [+1, +1]], 2 * self.d2)
        elif self.indexing == "ij":
            D0 = self.__create_kernel([[-1, -1], [+1, +1]], 2 * self.d1)
            D1 = self.__create_kernel([[-1, +1], [-1, +1]], 2 * self.d2)
        else:
            raise ValueError("indexing must be 'ij' or 'xy'")

        return self.backend.stack([D0, D1], dim=0)

    def three_dimensions(self) -> Any:
        """Generate 3D finite difference kernels.

        This method returns a stack of three kernels for computing derivatives
        along the first, second, and third dimensions (``D1``, ``D2``, ``D3``).

        Returns:
            Any: Kernels for the first, second, and third directions in backend format.

        Raises:
            ValueError: If ``d3`` is not specified.

        Notes:
            The order of the returned kernels depends on the indexing
            convention, mirroring ``numpy.meshgrid``:
            - ``"ij"`` (default): ``D[0]``/``D[1]``/``D[2]`` differentiate
              along axes 0/1/2 using spacings ``d1``/``d2``/``d3``
              respectively (no axis swap).
            - ``"xy"``: ``D[0]`` differentiates along axis 1 (spacing
              ``d1``); ``D[1]`` differentiates along axis 0 (spacing
              ``d2``); ``D[2]`` is unaffected and differentiates along
              axis 2 (spacing ``d3``), as in
              ``numpy.meshgrid(..., indexing="xy")``.
        """
        if self.d3 is None:
            raise ValueError(
                "The 'd3' value must be specified for 3D derivatives.")

        if self.indexing == "xy":
            D0 = self.__create_kernel(
                [[[-1, -1], [+1, +1]], [[-1, -1], [+1, +1]]], 4 * self.d1)
            D1 = self.__create_kernel(
                [[[-1, -1], [-1, -1]], [[+1, +1], [+1, +1]]], 4 * self.d2)
        elif self.indexing == "ij":
            D0 = self.__create_kernel(
                [[[-1, -1], [-1, -1]], [[+1, +1], [+1, +1]]], 4 * self.d1)
            D1 = self.__create_kernel(
                [[[-1, -1], [+1, +1]], [[-1, -1], [+1, +1]]], 4 * self.d2)
        else:
            raise ValueError("indexing must be 'ij' or 'xy'")

        D2 = self.__create_kernel(
            [[[-1, +1], [-1, +1]], [[-1, +1], [-1, +1]]], 4 * self.d3)

        return self.backend.stack([D0, D1, D2], dim=0)
