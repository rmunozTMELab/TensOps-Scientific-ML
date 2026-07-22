"""Affine connections on the tangent bundle, derived from tensor data.

A connection is deliberately **not** modelled as a :class:`TensOps`: its
coefficients ``Gamma^k_{ij}`` do not transform as a tensor. Under a general
change of coordinates they pick up an inhomogeneous term involving the second
derivative of the coordinate change,

    Gamma'^k_{ij} = (dx'^k/dx^a)(dx^b/dx'^i)(dx^c/dx'^j) Gamma^a_{bc}
                    + (dx'^k/dx^a)(d^2 x^a / dx'^i dx'^j),

whose second (affine, non-multilinear) piece is exactly what a genuine tensor
lacks. To keep this distinction explicit at the type level, ``ConnectOps`` is a
standalone class: it exposes neither a tensor ``base`` nor the
contravariant/covariant vocabulary of :class:`TensOps`, and instead describes
its index layout with the neutral terms *upper* and *lower* indices.
"""

from __future__ import annotations

from typing import Any

from ..algebra import contraction, inverse
from ..backends import get_backend
from ..tensors.metrics import MetrOps
from ..utils.averaging import m
from .operators import partial_derivative


def _christoffel_values(metric: MetrOps, D: Any) -> Any:
    """Compute the raw Christoffel symbol values ``Gamma^m_{pq}``.

    Applies ``Gamma^m_{pq} = (1/2) g^{ml} (d_p g_{ql} + d_q g_{pl} - d_l g_{pq})``.

    ``dg = partial_derivative(metric, D)`` is a rank-3 covariant tensor with value
    axes ``(a, b, c)`` such that ``dg[..., a, b, c, ...] = d_c g_{ab}``
    (``a, b`` are the metric's own indices, ``c`` is the appended derivative
    index). The metric is first restricted onto ``dg``'s (one node smaller)
    grid via ``m()``, and only then inverted: matrix inversion does not
    commute with cell-averaging, so ``inverse(m(metric))`` is the exact
    inverse of the restricted metric, whereas ``m(inverse(metric))`` is not
    (averaging the inverses computed at the original grid points does not
    generally equal the inverse of the averaged metric) -- using the latter
    would silently break metric compatibility (``nabla g = 0``). Each term is
    obtained by contracting this inverse metric with exactly one of ``dg``'s
    three covariant slots:

    - Contracting slot 0 (``a``) leaves free axes ``(b, c)``; swapping them
      gives ``g^{ml} d_p g_{ql}`` (the first term).
    - Contracting slot 1 (``b``) leaves free axes ``(a, c)`` already in the
      right order: ``g^{ml} d_q g_{pl}`` (the second term).
    - Contracting slot 2 (``c``, the derivative index itself) leaves free axes
      ``(a, b)``: ``g^{ml} d_l g_{pq}`` (the third term).

    Args:
        metric (MetrOps): Metric tensor to differentiate.
        D (Any): Bank of derivative kernels matching the metric's
            domain dimension.

    Returns:
        Any: Backend tensor with value axes ``(m, p, q)`` holding
            ``Gamma^m_{pq}``.
    """
    backend = get_backend()

    dg = partial_derivative(metric, D)
    g_inv = inverse(m(metric))

    term_from_slot0 = contraction(g_inv, dg, [[1], [None]], [[None], [0]])
    term_from_slot1 = contraction(g_inv, dg, [[1], [None]], [[None], [1]])
    term_from_slot2 = contraction(g_inv, dg, [[1], [None]], [[None], [2]])

    term1 = backend.transpose(term_from_slot0.values, (2, 3))   # g^ml d_p g_ql
    term2 = term_from_slot1.values                              # g^ml d_q g_pl
    term3 = term_from_slot2.values                              # g^ml d_l g_pq

    return 0.5 * (term1 + term2 - term3)


class ConnectOps:
    """Affine connection ``Gamma^k_{ij}`` on the tangent bundle.

    In coordinates a connection is defined by its coefficients ``Gamma^k_{ij}``
    (one **upper** index ``k``, two **lower** indices ``i, j``), which in
    general are arbitrary: they need not be symmetric (torsion-free) nor
    compatible with any metric. This class is therefore a permissive container
    that only validates the index/shape structure; specific connections are
    built through factory methods that enforce their own defining properties.
    The first such factory is :meth:`levi_civita` -- the unique torsion-free,
    metric-compatible connection of a metric -- and further ones (e.g. metric
    connections with torsion, or Weyl connections) can be added later.

    A connection is **not a tensor** (see the module docstring), so this class
    is intentionally independent from :class:`TensOps`. Its coefficients are
    stored with data shape ``[n_samples, d, d, d, discretization...]``, where
    axis 1 is the single upper index ``k`` and axes 2 and 3 are the two lower
    indices ``i, j``. ``d`` is the dimension of the space.

    Attributes:
        values (Any): Backend tensor holding the connection coefficients.
        domain (int): Number of domain dimensions (spatial discretization).
        codomain (list[int]): Index dimensions ``[d, d, d]``.
        dimension (int): Dimension ``d`` of the space the connection lives on.
        upper_indices (list[int]): Positions of the upper indices (``[0]``).
        lower_indices (list[int]): Positions of the lower indices (``[1, 2]``).
    """

    def __init__(self, data: Any, domain_dim: int, codomain_dim: list[int]):
        """Initialize a connection from its raw coefficients.

        Args:
            data (Any): Coefficient data with shape
                ``[n_samples, d, d, d, discretization...]``. Axis 1 is the
                upper index, axes 2 and 3 the two lower indices.
            domain_dim (int): Number of domain dimensions (spatial
                discretization).
            codomain_dim (list[int]): Index dimensions ``[d, d, d]``; the three
                entries must be equal.

        Raises:
            ValueError: If the metadata is malformed (unsupported domain, a
                codomain that is not three equal dimensions) or the data shape
                is inconsistent with it.
        """
        backend = get_backend()

        self.__validate_metadata(domain_dim, codomain_dim)

        self.__values = backend.clone(data)
        self.__domain = domain_dim
        self.__codomain = codomain_dim
        self.__upper_indices = [0]
        self.__lower_indices = [1, 2]

        self.__validate_shape()

    @classmethod
    def levi_civita(cls, metric: MetrOps, D: Any) -> "ConnectOps":
        """Build the Levi-Civita connection of a metric tensor.

        The Levi-Civita connection is the unique torsion-free connection
        compatible with a metric ``g_{ij}``, given by the Christoffel
        symbols::

            Gamma^m_{pq} = (1/2) g^{ml} (d_p g_{ql} + d_q g_{pl} - d_l g_{pq})

        Args:
            metric (MetrOps): Metric tensor ``g_{ij}`` to derive the
                connection from.
            D (Any): Bank of derivative kernels (see
                :func:`tensopsciml.calculus.partial_derivative`) matching the metric's
                domain dimension.

        Returns:
            ConnectOps: The Levi-Civita connection of ``metric``.

        Raises:
            ValueError: If ``metric`` is not a ``MetrOps`` instance, or if its
                codomain dimension does not match its domain dimension (the
                connection's indices must range over the same tangent space the
                domain is discretized over).
        """
        if not isinstance(metric, MetrOps):
            raise ValueError(
                f"Expected a MetrOps instance, received {type(metric).__name__}."
            )

        if metric.domain != metric.codomain[0]:
            raise ValueError(
                "A connection requires the metric's codomain dimension to "
                "match its domain dimension (the tangent space rank must "
                f"equal the number of spatial dimensions); received codomain "
                f"{metric.codomain} over a {metric.domain}D domain."
            )

        christoffel_values = _christoffel_values(metric, D)
        dimension = metric.codomain[0]

        return cls(
            christoffel_values,
            domain_dim=metric.domain,
            codomain_dim=[dimension, dimension, dimension],
        )

    def __validate_metadata(
        self, domain_dim: int, codomain_dim: list[int]
    ) -> None:
        """Validate the domain and codomain metadata of the connection.

        Args:
            domain_dim (int): Number of domain dimensions.
            codomain_dim (list[int]): Declared index dimensions.

        Raises:
            ValueError: If the domain is unsupported, or the codomain is not a
                list of three equal integer dimensions.
        """
        if domain_dim not in {1, 2, 3}:
            raise ValueError(
                f"Domain dimension {domain_dim} must be one of [1, 2, 3]."
            )

        if (
            not isinstance(codomain_dim, list)
            or len(codomain_dim) != 3
            or not all(isinstance(dim, int) for dim in codomain_dim)
        ):
            raise ValueError(
                "A connection codomain must be a list of three integer "
                f"dimensions [d, d, d]; received {codomain_dim}."
            )

        if not codomain_dim[0] == codomain_dim[1] == codomain_dim[2]:
            raise ValueError(
                "A connection's three indices must share the same dimension; "
                f"received codomain {codomain_dim}."
            )

    def __validate_shape(self) -> None:
        """Ensure the stored values match the declared index structure.

        Raises:
            ValueError: If the coefficient tensor shape is incompatible with
                the domain or codomain metadata.
        """
        actual_shape = list(self.__values.shape)
        expected_shape = (
            [actual_shape[0]] + self.__codomain + actual_shape[-self.__domain:]
        )
        if actual_shape != expected_shape:
            raise ValueError(
                "Inconsistent connection shape: "
                f"expected {expected_shape}, received {actual_shape}."
            )

    @property
    def values(self) -> Any:
        """Return the underlying connection coefficient data.

        Returns:
            Any: Backend tensor holding the numerical coefficients.
        """
        return self.__values

    @property
    def domain(self) -> int:
        """Return the number of domain dimensions (spatial discretization).

        Returns:
            int: Number of dimensions in the connection's domain.
        """
        return self.__domain

    @property
    def codomain(self) -> list[int]:
        """Return the index dimensions ``[d, d, d]``.

        Returns:
            list[int]: Dimensions of the three connection indices.
        """
        return self.__codomain

    @property
    def dimension(self) -> int:
        """Return the dimension ``d`` of the space the connection lives on.

        Returns:
            int: Common size of every connection index.
        """
        return self.__codomain[0]

    @property
    def upper_indices(self) -> list[int]:
        """Return the positions of the upper indices.

        Returns:
            list[int]: Position of the single upper index (``[0]``).
        """
        return self.__upper_indices

    @property
    def lower_indices(self) -> list[int]:
        """Return the positions of the lower indices.

        Returns:
            list[int]: Positions of the two lower indices (``[1, 2]``).
        """
        return self.__lower_indices

    def __str__(self) -> str:
        """Return a user-friendly string representation of the connection.

        Returns:
            str: A formatted string describing the connection's structure.
        """
        return (
            f"<class '{self.__class__.__name__}'>\n"
            f" Shape: {list(self.values.shape)}\n"
            "\n"
            f" Domain dim: {self.domain}\n"
            f" Codomain dim: {self.codomain}\n"
            "\n"
            f" Upper indices: {self.upper_indices}\n"
            f" Lower indices: {self.lower_indices}"
        )

    def __repr__(self) -> str:
        """Return a detailed string representation for debugging.

        Returns:
            str: A formatted string describing the connection's structure.
        """
        return self.__str__()
