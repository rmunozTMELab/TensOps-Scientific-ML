"""Metric tensor specialization of :class:`TensOps`."""

from __future__ import annotations

from typing import Any

from ..backends import get_backend
from .tensor import TensOps


def _matrix_view(values: Any) -> Any:
    """Move a rank-2 tensor field's two codomain axes to the end.

    Turns a tensor field shaped ``[n_samples, d, d, discretization...]`` into
    ``[n_samples, discretization..., d, d]``, so batched matrix routines
    (determinant, inverse, ...) act on the trailing ``d x d`` block at every
    domain point.

    Args:
        values (Any): Backend tensor with two codomain axes at positions 1
            and 2.

    Returns:
        Any: Permuted backend tensor.
    """
    backend = get_backend()
    axes = range(len(values.shape))
    permute_order = [axis for axis in axes if axis not in (1, 2)] + [1, 2]
    return backend.permute(values, permute_order)


class MetrOps(TensOps):
    """Metric tensor ``g_{ij}`` with a fixed rank-2 covariant structure.

    A metric is a rank-2, fully covariant tensor field, so its tensorial order
    is fixed to ``[0, 2]`` (two covariant indices, base ``V* x V*``).
    ``MetrOps`` specializes :class:`TensOps` by hard-coding that order and only
    exposing the data, domain and codomain, whose mutual consistency is checked
    by the inherited :class:`TensOps` validation.

    The two covariant indices live on value axes 1 and 2, so the data follows
    the shape convention ``[n_samples, d, d, discretization...]``, where ``d``
    is the dimension of the vector space the metric acts on.
    """

    def __init__(self, data: Any, domain_dim: int, codomain_dim: list[int]):
        """Initialize a metric tensor from backend-compatible data.

        Args:
            data (Any): Tensor data with shape
                ``[n_samples, d, d, discretization...]``. The two ``d`` axes
                carry the covariant indices.
            domain_dim (int): Number of domain dimensions (spatial
                discretization).
            codomain_dim (list[int]): Dimensions of the codomain, ``[d, d]``.

        Raises:
            ValueError: If ``data``, ``domain_dim`` and ``codomain_dim`` are
                mutually inconsistent (see :class:`TensOps`), if the codomain is
                not square, if the metric is not symmetric, or if it is not
                positive definite and non-degenerate.
        """
        super().__init__(
            data,
            domain_dim=domain_dim,
            codomain_dim=codomain_dim,
            order=[0, 2],
            contravariant_indices=None,
            covariant_indices=[0, 1],
        )

        self.__validate_square()
        self.__validate_symmetry()
        self.__validate_positive_definite()

    @classmethod
    def _from_inverse(
        cls,
        data: Any,
        domain_dim: int,
        codomain_dim: list[int],
        order: list[int],
        contravariant_indices: list[int] | None,
        covariant_indices: list[int] | None,
    ) -> "MetrOps":
        """Build the opposite-variance counterpart of a metric tensor.

        A ``MetrOps`` declared from raw data is always covariant, ``g_{ij}``
        (order ``[0, 2]``); this is the only way to reach the contravariant
        form, ``g^{ij}`` (order ``[2, 0]``), or to invert back. It exists
        solely so ``inverse()`` (in ``algebra.matrix_algebra``) can return a
        ``MetrOps`` instead of a plain ``TensOps`` -- it bypasses the public
        constructor's covariant-only order, but still runs the same
        square / symmetry / positive-definiteness checks, since the inverse
        of a symmetric positive-definite matrix is itself symmetric positive
        definite.

        Args:
            data (Any): Backend tensor data for the inverted metric.
            domain_dim (int): Domain dimension, inherited from the original
                metric.
            codomain_dim (list[int]): Codomain, inherited from the original
                metric.
            order (list[int]): Either ``[0, 2]`` or ``[2, 0]``.
            contravariant_indices (list[int] | None): Contravariant indices
                matching ``order``.
            covariant_indices (list[int] | None): Covariant indices matching
                ``order``.

        Returns:
            MetrOps: New metric instance with the requested variance.
        """
        instance = cls.__new__(cls)
        TensOps.__init__(
            instance,
            data,
            domain_dim=domain_dim,
            codomain_dim=codomain_dim,
            order=order,
            contravariant_indices=contravariant_indices,
            covariant_indices=covariant_indices,
        )

        instance.__validate_square()
        instance.__validate_symmetry()
        instance.__validate_positive_definite()

        return instance

    def __validate_square(self) -> None:
        """Ensure the metric acts on a single vector space, ``g_{ij}: V x V``.

        A metric is a bilinear form on one vector space, so both covariant
        indices must range over the same dimension and the codomain must be
        square, ``[d, d]``.

        Raises:
            ValueError: If the two codomain dimensions differ.
        """
        if self.codomain[0] != self.codomain[1]:
            raise ValueError(
                "A metric tensor must have a square codomain [d, d]; "
                f"received {self.codomain}."
            )

    def __validate_symmetry(self) -> None:
        """Ensure the metric is symmetric, ``g_{ij} = g_{ji}``, everywhere.

        Symmetry is checked up to floating-point tolerance (``allclose`` rather
        than exact equality), since a metric is usually assembled numerically
        (network outputs, matrix products, reductions) and is only symmetric to
        machine precision.

        Raises:
            ValueError: If swapping the two covariant axes changes the tensor.
        """
        backend = get_backend()
        transposed = backend.transpose(self.values, (1, 2))
        if not backend.allclose(self.values, transposed):
            raise ValueError(
                "A metric tensor must be symmetric: g_{ij} = g_{ji}."
            )

    def __validate_positive_definite(self) -> None:
        """Ensure the metric is positive definite and non-degenerate.

        Applies Sylvester's criterion point-wise: since the metric is already
        known to be symmetric, it is positive definite if and only if the
        determinant of each of its leading principal minors is strictly
        positive. The full determinant (the last minor) being positive also
        certifies that the metric is non-degenerate.

        Raises:
            ValueError: If any leading principal minor has a non-positive
                determinant at any point of the domain.
        """
        backend = get_backend()
        matrices = _matrix_view(self.values)
        dimension = self.codomain[0]

        for size in range(1, dimension + 1):
            leading_minor = matrices[..., :size, :size]
            minor_determinant = backend.det(leading_minor)
            if not self.__all_positive(minor_determinant):
                raise ValueError(
                    "A metric tensor must be positive definite and "
                    "non-degenerate; every leading principal minor must have a "
                    "positive determinant at every point of the domain."
                )

    @staticmethod
    def __all_positive(tensor: Any) -> bool:
        """Return whether every element of a tensor is strictly positive.

        Args:
            tensor (Any): Backend tensor of real values.

        Returns:
            bool: True if all elements are greater than zero.
        """
        return bool((tensor > 0).all())

    def __reject_arithmetic(self, operator_symbol: str) -> None:
        """Raise for any arithmetic operator inherited from ``TensOps``.

        Not every arithmetic operation preserves the metric invariant
        (symmetric, positive definite, non-degenerate): subtraction, negation,
        and multiplication/division by a non-positive scalar can all break it.
        Even addition or multiplication by a positive scalar, which do
        preserve it mathematically, already return a plain ``TensOps`` from
        the inherited operators (they are not re-validated as a metric), so
        allowing only "the safe half" would be inconsistent. Arithmetic on
        ``MetrOps`` is therefore disabled entirely for now.

        Args:
            operator_symbol (str): Symbol of the attempted operator, used in
                the error message.

        Raises:
            TypeError: Always.
        """
        raise TypeError(
            f"Arithmetic operator '{operator_symbol}' is not supported on "
            "MetrOps. Convert to a plain TensOps first "
            "(TensOps(g.values, g.domain, g.codomain, g.order, "
            "g.contravariant, g.covariant)) if you need to combine metrics."
        )

    def __add__(self, other: TensOps) -> TensOps:
        self.__reject_arithmetic("+")

    def __sub__(self, other: TensOps) -> TensOps:
        self.__reject_arithmetic("-")

    def __mul__(self, scalar: int | float | TensOps) -> TensOps:
        self.__reject_arithmetic("*")

    def __rmul__(self, scalar: int | float) -> TensOps:
        self.__reject_arithmetic("*")

    def __truediv__(self, scalar: int | float | TensOps) -> TensOps:
        self.__reject_arithmetic("/")

    def __neg__(self) -> TensOps:
        self.__reject_arithmetic("-")

    def __pos__(self) -> TensOps:
        self.__reject_arithmetic("+")


class VolFormOps(TensOps):
    """Volume form ``sqrt(det(g))`` associated to a metric tensor.

    In Riemannian geometry, the volume element of a metric ``g_{ij}`` is
    ``sqrt(det(g)) dx^1 ^ ... ^ dx^n``. ``VolFormOps`` stores the scalar density
    factor ``sqrt(det(g))`` as a scalar field (order ``[0, 0]``, codomain
    ``[1]``), so it can be used directly as the weight in volume integrals,
    e.g. ``integrand * volume_form``.

    Its only argument is a :class:`MetrOps` instance: since a valid metric is
    guaranteed to be positive definite, ``det(g) > 0`` everywhere, so the
    square root is always real and well-defined -- no extra validation is
    needed here.
    """

    def __init__(self, metric: MetrOps):
        """Compute the volume form of a metric tensor.

        Args:
            metric (MetrOps): Metric tensor ``g_{ij}`` to derive the volume
                form from.

        Raises:
            ValueError: If ``metric`` is not a ``MetrOps`` instance.
        """
        if not isinstance(metric, MetrOps):
            raise ValueError(
                f"Expected a MetrOps instance, received {type(metric).__name__}."
            )

        backend = get_backend()
        matrix = _matrix_view(metric.values)
        metric_determinant = backend.det(matrix)
        metric_determinant = backend.unsqueeze(metric_determinant, 1)
        volume_values = metric_determinant ** 0.5

        super().__init__(
            volume_values,
            domain_dim=metric.domain,
            codomain_dim=[1],
            order=[0, 0],
            contravariant_indices=None,
            covariant_indices=None,
        )

    @classmethod
    def _from_values(cls, data: Any, domain_dim: int) -> "VolFormOps":
        """Wrap already-computed scalar values as a ``VolFormOps`` directly.

        Bypasses the ``sqrt(det(g))`` computation in the public constructor,
        which needs a :class:`MetrOps` to derive from. This exists solely so
        :func:`~tensopsciml.utils.averaging.m` can restrict an existing,
        already-valid volume form onto a coarser grid by directional
        averaging -- the same restriction the library already performs
        internally to move a volume form from one operator's grid to a
        coarser one (see :func:`~tensopsciml.calculus.operators.laplacian`)
        -- without needing the metric it originally came from.

        Args:
            data (Any): Scalar field values, shape
                ``[n_samples, 1, discretization...]``.
            domain_dim (int): Number of domain dimensions.

        Returns:
            VolFormOps: New instance wrapping ``data`` directly.
        """
        instance = cls.__new__(cls)
        TensOps.__init__(
            instance,
            data,
            domain_dim=domain_dim,
            codomain_dim=[1],
            order=[0, 0],
            contravariant_indices=None,
            covariant_indices=None,
        )
        return instance


