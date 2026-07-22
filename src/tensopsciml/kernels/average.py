"""Averaging (smoothing) convolution kernels for 1D, 2D, and 3D grids."""

from typing import Any, Sequence

from ..backends import Backend, get_backend


class Average:
    """Generate averaging (smoothing) kernels for 1D, 2D, and 3D data."""

    def __init__(
        self,
        order: str = "1",
        indexing: str = "ij",
        device: Any = "cpu",
        *,
        backend: Backend | None = None,
        dtype: Any | None = None,
    ) -> None:
        """Initialize the Average kernel generator.

        Args:
            order (str): Order of the averaging filter. Only first order is
                supported.
            indexing (str): Indexing convention (``"ij"`` or ``"xy"``).
            device (Any): Backend-specific device descriptor where kernels
                will live.
            backend (Backend | None): Optional backend instance. Defaults to
                the active backend.
            dtype (Any | None): Optional backend-specific dtype. Defaults to
                the backend default.

        Raises:
            ValueError: If ``order`` is not ``"1"`` or ``indexing`` is invalid.
        """

        if order != "1":
            raise ValueError(
                "Only first-order averaging filters are currently supported.")

        if indexing not in {"ij", "xy"}:
            raise ValueError("indexing must be 'ij' or 'xy'")

        self.backend = backend or get_backend()
        self.device = device
        self.dtype = dtype
        self.order = order
        self.indexing = indexing

    def __create_kernel(self, coeffs: Sequence[Any], scale_factor: float) -> Any:
        """Create a backend tensor kernel from coefficient lists.

        Args:
            coeffs (Sequence[Any]): Nested list describing the kernel stencil.
            scale_factor (float): Normalization factor applied to the
                coefficients.

        Returns:
            Any: Backend tensor ready to be used as a convolution kernel.
        """

        backend_tensor_kwargs = {"device": self.device}
        if self.dtype is not None:
            backend_tensor_kwargs["dtype"] = self.dtype

        kernel = self.backend.tensor(
            coeffs, **backend_tensor_kwargs) / scale_factor
        return kernel.unsqueeze(0).unsqueeze(0)

    def one_dimension(self) -> Any:
        """Generate a 1D averaging kernel.

        This kernel averages adjacent elements in a 1D array.

        Returns:
            Any: Backend tensor shaped ``(1, 1, 1, 2)``.
        """
        M = self.__create_kernel([[+1, +1]], 2)

        return M

    def two_dimensions(self) -> list[Any]:
        """Generate 2D averaging kernels for each direction.

        Returns:
            list[Any]: Kernels for averaging along the first and second dimensions.

        Notes:
            The order of the returned kernels depends on the indexing
            convention:
            - ``"ij"`` (default): ``(M1, M2)`` corresponds to ``(rows, columns)``
            - ``"xy"``: ``(M1, M2)`` corresponds to ``(x, y)``
        """
        M1 = self.__create_kernel([[+1, +1]], 2)
        M2 = self.__create_kernel([[+1], [+1]], 2)

        return [M1, M2] if self.indexing == "xy" else [M2, M1]

    def three_dimensions(self) -> list[Any]:
        """Generate 3D averaging kernels for each direction.

        Returns:
            list[Any]: Kernels for averaging along the first, second,
                and third dimensions, respectively.

        Notes:
            The order of the returned kernels depends on the indexing
            convention:
            - ``"ij"`` (default): ``(M1, M2, M3)`` corresponds to ``(dim0, dim1, dim2)``
            - ``"xy"``: ``(M1, M2, M3)`` corresponds to ``(x, y, z)`` with the
                first two axes swapped
        """
        M1 = self.__create_kernel([[[+1], [+1]]], 2)
        M2 = self.__create_kernel([[[+1]], [[+1]]], 2)
        M3 = self.__create_kernel([[[+1, +1]]], 2)

        return [M1, M2, M3] if self.indexing == "xy" else [M2, M1, M3]
