"""Matrix algebra on second-order tensor fields (trace, transpose, inverse...)."""

from ..backends import get_backend
from ..tensors import MetrOps, TensOps
from ..utils.validation import validate_tensops


def trace(A: TensOps) -> TensOps:
    """Compute the trace of a second-order tensor.

    Args:
        A (TensOps): Second-order tensor (square matrix).

    Returns:
        TensOps: Scalar field representing the trace of A.

    Raises:
        ValueError: If A is not a square matrix or its domain dimension
            is unsupported.

    Examples:
        >>> import torch
        >>> from tensopsciml import TensOps
        >>> from tensopsciml.algebra import trace
        >>> F_torch = torch.rand((100, 2, 2, 10, 10, 10))
        >>> F = TensOps(
        ...     F_torch,
        ...     domain_dim=3,
        ...     codomain_dim=[2, 2],
        ...     order=[1, 1],
        ...     contravariant_indices=[1],
        ...     covariant_indices=[0],
        ... )
        >>> tr_F = trace(F)
        >>> tr_F.order
        [0, 0]
    """
    validate_tensops(A)
    backend = get_backend()

    if sum(A.order) != 2 or A.codomain[0] != A.codomain[1]:
        raise ValueError(
            "trace() requires a square second-order tensor; received order "
            f"{A.order}, codomain {A.codomain}.")

    if A.order != [1, 1]:
        raise ValueError(
            "trace() requires a mixed tensor (order [1, 1]) for its indices to "
            f"contract; received order {A.order}.")

    if A.domain not in {1, 2, 3}:
        raise ValueError(
            f"trace() requires a domain dimension of 1, 2, or 3; received {A.domain}.")

    # Build the einsum expression based on the domain dimension.
    discretization = {1: 'x', 2: 'xy', 3: 'xyz'}[A.domain]
    indices = "nii" + discretization
    output_shape = "n" + discretization

    trace_values = backend.einsum(f'{indices}->{output_shape}', A.values)
    trace_values = backend.unsqueeze(trace_values, 1)

    return TensOps(
        trace_values,
        domain_dim=A.domain,
        codomain_dim=[1],
        order=[0, 0],
        contravariant_indices=None,
        covariant_indices=None,
    )


def transpose(A: TensOps) -> TensOps:
    """Compute the transpose of a same-variance second-order tensor.

    Swaps the two value axes of a same-variance tensor — fully covariant
    ``A_{ij}`` (order ``[0, 2]``) or fully contravariant ``A^{ij}`` (order
    ``[2, 0]``) — yielding ``A_{ji}`` / ``A^{ji}``. Because both indices share
    the same variance, this is a pure axis swap: no metric is applied and the
    variance metadata is left unchanged.

    Mixed tensors (order ``[1, 1]``) are rejected: transposing ``A^i{}_j`` into
    ``A^j{}_i`` would exchange the variance of the two slots, which is only
    well-defined once an index is raised or lowered with the metric. Use
    :func:`raise_index` / :func:`lower_index` for that case.

    Args:
        A (TensOps): Same-variance second-order tensor, order ``[0, 2]`` or
            ``[2, 0]``.

    Returns:
        TensOps: Transposed tensor (its two value axes swapped), with the same
            variance metadata as ``A``.

    Raises:
        ValueError: If ``A`` is not a same-variance second-order tensor (order
            ``[0, 2]`` or ``[2, 0]``), or its domain dimension is unsupported.

    Examples:
        >>> import torch
        >>> from tensopsciml import TensOps
        >>> from tensopsciml.algebra import transpose
        >>> A_torch = torch.rand((100, 2, 2, 10, 10, 10))
        >>> A = TensOps(
        ...     A_torch,
        ...     domain_dim=3,
        ...     codomain_dim=[2, 2],
        ...     order=[0, 2],
        ...     contravariant_indices=None,
        ...     covariant_indices=[0, 1],
        ... )
        >>> A_t = transpose(A)
        >>> (A_t.order, A_t.contravariant, A_t.covariant)
        ([0, 2], None, [0, 1])
    """
    validate_tensops(A)
    backend = get_backend()

    if A.order not in ([2, 0], [0, 2]):
        raise ValueError(
            "transpose() requires a same-variance second-order tensor (order "
            f"[2, 0] or [0, 2]); received order {A.order}. Raise or lower one "
            "index (with an explicit metric) first for a mixed tensor.")

    if A.domain not in {1, 2, 3}:
        raise ValueError(
            f"transpose() requires a domain dimension of 1, 2, or 3; received {A.domain}.")

    transposed_values = backend.transpose(A.values, (1, 2))

    return TensOps(
        transposed_values,
        domain_dim=A.domain,
        codomain_dim=A.codomain[::-1],
        order=A.order,
        contravariant_indices=A.contravariant,
        covariant_indices=A.covariant,
    )


def inverse(A: TensOps) -> TensOps:
    """Compute the inverse of a square second-order tensor.

    Args:
        A (TensOps): Square second-order tensor.

    Returns:
        TensOps: Matrix inverse with the same variance structure as A. If
            ``A`` is a :class:`MetrOps` (order ``[0, 2]`` or ``[2, 0]``),
            the result is also a ``MetrOps``, flipped to the other
            variance.

    Raises:
        ValueError: If A is not a matrix, not square, or its domain dimension
            is unsupported.

    Examples:
        >>> import torch
        >>> from tensopsciml import TensOps
        >>> from tensopsciml.algebra import inverse
        >>> F_torch = torch.rand((100, 2, 2, 10, 10, 10))
        >>> F = TensOps(
        ...     F_torch,
        ...     domain_dim=3,
        ...     codomain_dim=[2, 2],
        ...     order=[1, 1],
        ...     contravariant_indices=[1],
        ...     covariant_indices=[0],
        ... )
        >>> inverse(F).order
        [1, 1]
    """
    validate_tensops(A)
    backend = get_backend()

    if sum(A.order) != 2:
        raise ValueError(
            "inverse() requires a second-order tensor (a matrix); received "
            f"order {A.order}.")

    if A.domain not in {1, 2, 3}:
        raise ValueError(
            f"inverse() requires a domain dimension of 1, 2, or 3; received {A.domain}.")

    if A.codomain[0] != A.codomain[1]:
        raise ValueError(
            f"inverse() requires a square matrix; received codomain {A.codomain}.")

    # Derive permutations for stable inverse computation.
    permute_order = (
        [i for i in range(len(A.values.shape)) if i not in [1, 2]] + [1, 2]
    )
    permute_back_order = [
        0,
        len(A.values.shape) - 2,
        len(A.values.shape) - 1,
    ] + [
        i
        for i in range(len(A.values.shape))
        if i not in [0, len(A.values.shape) - 2, len(A.values.shape) - 1]
    ]

    permuted = backend.permute(A.values, permute_order)
    inverted = backend.inv(permuted)
    A_inverse = backend.permute(inverted, permute_back_order)

    # Inverting a same-variance tensor flips its variance (fully covariant
    # <-> fully contravariant); a mixed tensor keeps its variance unchanged.
    if A.order in ([0, 2], [2, 0]):
        new_order = A.order[::-1]
        new_contravariant = A.covariant
        new_covariant = A.contravariant
    else:
        new_order = A.order
        new_contravariant = A.contravariant
        new_covariant = A.covariant

    if isinstance(A, MetrOps):
        return MetrOps._from_inverse(
            A_inverse,
            domain_dim=A.domain,
            codomain_dim=A.codomain,
            order=new_order,
            contravariant_indices=new_contravariant,
            covariant_indices=new_covariant,
        )

    return TensOps(
        A_inverse,
        domain_dim=A.domain,
        codomain_dim=A.codomain,
        order=new_order,
        contravariant_indices=new_contravariant,
        covariant_indices=new_covariant,
    )


def determinant(A: TensOps) -> TensOps:
    """Compute the determinant of a square second-order tensor.

    Args:
        A (TensOps): Square second-order tensor (2x2 or 3x3).

    Returns:
        TensOps: Scalar field representing the determinant of A.

    Raises:
        ValueError: If ``A`` is not a square matrix (2x2 or 3x3), or its domain
            dimension is unsupported.

    Examples:
        >>> import torch
        >>> from tensopsciml import TensOps
        >>> from tensopsciml.algebra import determinant
        >>> F_torch = torch.rand((100, 2, 2, 10, 10, 10))
        >>> F = TensOps(
        ...     F_torch,
        ...     domain_dim=3,
        ...     codomain_dim=[2, 2],
        ...     order=[1, 1],
        ...     contravariant_indices=[0],
        ...     covariant_indices=[1],
        ... )
        >>> determinant(F).order
        [0, 0]
    """
    validate_tensops(A)
    backend = get_backend()

    if sum(A.order) != 2 or A.codomain[0] != A.codomain[1]:
        raise ValueError(
            "determinant() requires a square second-order tensor; received "
            f"order {A.order}, codomain {A.codomain}.")

    if A.codomain[0] not in {2, 3}:
        raise ValueError(
            "determinant() only supports 2x2 or 3x3 matrices; received "
            f"codomain {A.codomain}.")

    permute_order = (
        [i for i in range(len(A.values.shape)) if i not in [1, 2]] + [1, 2]
    )
    permuted = backend.permute(A.values, permute_order)
    result = backend.det(permuted)
    result = backend.unsqueeze(result, 1)

    return TensOps(
        result,
        domain_dim=A.domain,
        codomain_dim=[1],
        order=[0, 0],
        contravariant_indices=None,
        covariant_indices=None,
    )


def spherical(A: TensOps, I: TensOps) -> TensOps:
    """Compute the spherical (volumetric) part of a mixed second-order tensor.

    Returns ``(1/n) * tr(A) * I``. The trace is only defined for the mixed
    form ``A^i{}_j`` (order ``[1,1]``), so ``A`` must already be mixed; if it
    is fully covariant or contravariant, raise/lower one index first (with an
    explicit metric) before calling this. ``I`` is the mixed identity
    ``delta^i{}_j`` (order ``[1,1]``) supplied by the caller, matching ``A``'s
    grid, dtype and device -- e.g. ``kronecker_delta(metric)``.

    Args:
        A (TensOps): Square mixed second-order tensor, order ``[1,1]``.
        I (TensOps): Mixed identity ``delta^i{}_j`` (order ``[1,1]``) on the
            same grid as ``A``; scales the trace to form the spherical tensor.

    Returns:
        TensOps: Spherical tensor ``(1/n) * tr(A) * I``, order ``[1,1]``.

    Raises:
        ValueError: If ``A`` is not a square mixed second-order tensor
            (order ``[1,1]`` with a square codomain).
    """
    validate_tensops(A)

    if A.order != [1, 1] or A.codomain[0] != A.codomain[1]:
        raise ValueError(
            "spherical() requires a square mixed second-order tensor (order "
            f"[1, 1]); received order {A.order}, codomain {A.codomain}. "
            "Raise or lower one index (with an explicit metric) first if A is "
            "fully covariant or contravariant.")

    n = A.codomain[0]

    return (1.0 / n) * (trace(A) * I)


def deviatoric(A: TensOps, I: TensOps) -> TensOps:
    """Compute the deviatoric part of a mixed second-order tensor.

    Args:
        A (TensOps): Square mixed second-order tensor, order ``[1,1]``.
        I (TensOps): Mixed identity ``delta^i{}_j`` (order ``[1,1]``) on the
            same grid as ``A`` (see :func:`spherical`).

    Returns:
        TensOps: Deviatoric tensor ``A - spherical(A, I)``, which has zero
            trace.

    Raises:
        ValueError: If ``A`` is not a square mixed second-order tensor
            (order ``[1,1]``).
    """
    return A - spherical(A, I)