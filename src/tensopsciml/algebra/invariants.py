"""Principal invariants of a square second-order tensor."""

from ..tensors import TensOps
from ..utils.validation import validate_tensops
from .matrix_algebra import determinant, trace
from .tensor_algebra import contraction


def first_invariant(A: TensOps) -> TensOps:
    """Compute the first invariant (trace) of a square tensor.

    Args:
        A (TensOps): Square mixed second-order tensor (order ``[1, 1]``).

    Returns:
        TensOps: Scalar field representing ``tr(A)``.

    Raises:
        ValueError: If ``A`` is not a square mixed second-order tensor, or its
            domain dimension is unsupported.

    Examples:
        >>> import torch
        >>> from tensopsciml import TensOps
        >>> from tensopsciml.algebra import first_invariant
        >>> F_torch = torch.rand((100, 2, 2, 10, 10, 10))
        >>> F = TensOps(
        ...     F_torch,
        ...     domain_dim=3,
        ...     codomain_dim=[2, 2],
        ...     order=[1, 1],
        ...     contravariant_indices=[0],
        ...     covariant_indices=[1],
        ... )
        >>> first_invariant(F).order
        [0, 0]
    """
    validate_tensops(A)
    return trace(A)


def second_invariant(A: TensOps) -> TensOps:
    """Compute the second invariant of a square tensor.

    The second invariant is defined as ``I2 = 0.5 * (tr(A)^2 - tr(A^2))``.

    Args:
        A (TensOps): Square mixed second-order tensor (order ``[1, 1]``).

    Returns:
        TensOps: Scalar field representing ``I2``.

    Raises:
        ValueError: If ``A`` is not a square mixed second-order tensor, or its
            domain dimension is unsupported.

    Examples:
        >>> import torch
        >>> from tensopsciml import TensOps
        >>> from tensopsciml.algebra import second_invariant
        >>> F_torch = torch.rand((100, 2, 2, 10, 10, 10))
        >>> F = TensOps(
        ...     F_torch,
        ...     domain_dim=3,
        ...     codomain_dim=[2, 2],
        ...     order=[1, 1],
        ...     contravariant_indices=[0],
        ...     covariant_indices=[1],
        ... )
        >>> second_invariant(F).order
        [0, 0]
    """
    validate_tensops(A)

    if A.order != [1, 1]:
        raise ValueError(
            "second_invariant() requires a mixed second-order tensor (order "
            f"[1, 1]) so its indices can contract; received order {A.order}. "
            "Raise or lower one index first for a same-variance tensor.")

    # Build A^2 via contraction over a covariant-contravariant pair.
    A_sqr = contraction(A, A, [[None], [0]], [[0], [None]])
    return 0.5 * (trace(A) * trace(A) - trace(A_sqr))


def third_invariant(A: TensOps) -> TensOps:
    """Compute the third invariant (determinant) of a square tensor.

    Args:
        A (TensOps): Square second-order tensor (2x2 or 3x3).

    Returns:
        TensOps: Scalar field representing ``det(A)``.

    Raises:
        ValueError: If ``A`` is not a square matrix, or its domain dimension is
            unsupported.

    Examples:
        >>> import torch
        >>> from tensopsciml import TensOps
        >>> from tensopsciml.algebra import third_invariant
        >>> F_torch = torch.rand((100, 2, 2, 10, 10, 10))
        >>> F = TensOps(
        ...     F_torch,
        ...     domain_dim=3,
        ...     codomain_dim=[2, 2],
        ...     order=[1, 1],
        ...     contravariant_indices=[0],
        ...     covariant_indices=[1],
        ... )
        >>> third_invariant(F).order
        [0, 0]
    """
    validate_tensops(A)
    return determinant(A)
