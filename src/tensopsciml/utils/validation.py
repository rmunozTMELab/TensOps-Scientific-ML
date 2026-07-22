"""Validation helpers ensuring tensors and kernels meet TensOps requirements."""

from typing import Any

from ..tensors import MetrOps, TensOps, VolFormOps


def validate_tensops(*args: TensOps) -> None:
    """Ensure every provided argument is an instance of ``TensOps``.

    Args:
        *args (TensOps): Objects to validate.

    Returns:
        None: The function raises an error if validation fails.

    Raises:
        ValueError: If any object is not a ``TensOps`` instance.
    """

    for arg in args:
        if not isinstance(arg, TensOps):
            raise ValueError(
                f"Expected TensOps instance, received {type(arg).__name__}.")


def validate_metrops(*args: MetrOps) -> None:
    """Ensure every provided argument is an instance of ``MetrOps``.

    Args:
        *args (MetrOps): Objects to validate.

    Returns:
        None: The function raises an error if validation fails.

    Raises:
        ValueError: If any object is not a ``MetrOps`` instance.
    """

    for arg in args:
        if not isinstance(arg, MetrOps):
            raise ValueError(
                f"Expected MetrOps instance, received {type(arg).__name__}.")


def validate_volformops(*args: VolFormOps) -> None:
    """Ensure every provided argument is an instance of ``VolFormOps``.

    Args:
        *args (VolFormOps): Objects to validate.

    Returns:
        None: The function raises an error if validation fails.

    Raises:
        ValueError: If any object is not a ``VolFormOps`` instance.
    """

    for arg in args:
        if not isinstance(arg, VolFormOps):
            raise ValueError(
                f"Expected VolFormOps instance, received {type(arg).__name__}.")


def validate_connectops(*args: Any) -> None:
    """Ensure every provided argument is an instance of ``ConnectOps``.

    ``ConnectOps`` lives in ``calculus.connections``, which (via
    ``calculus.connections`` -> ``utils.averaging`` -> ``utils.validation``)
    already depends on this module; importing it at module scope here would
    close that cycle. The import is deferred into the function body instead,
    where it's safe because by call time every module involved has already
    finished loading.

    Args:
        *args (Any): Objects to validate.

    Returns:
        None: The function raises an error if validation fails.

    Raises:
        ValueError: If any object is not a ``ConnectOps`` instance.
    """
    from ..calculus.connections import ConnectOps

    for arg in args:
        if not isinstance(arg, ConnectOps):
            raise ValueError(
                f"Expected ConnectOps instance, received {type(arg).__name__}.")


def validate_codomain_matches_connection(field: TensOps, connection: Any) -> None:
    """Validate that a field's codomain dimensions match a connection's.

    Every covariant operator that applies Christoffel corrections
    (:func:`cov_derivative`, :func:`lie_derivative`, ...) requires each of the
    field's codomain (index) dimensions to equal the connection's ``dimension``
    -- otherwise the ``Gamma``-index contractions are ill-defined. Scalar
    fields (no free indices) are exempt.

    Args:
        field (TensOps): Field whose indices are corrected by the connection.
        connection (ConnectOps): Affine connection to match against.

    Returns:
        None: An exception is raised when the dimensions are incompatible.

    Raises:
        ValueError: If any codomain dimension differs from the connection's.
    """

    if sum(field.order) > 0 and any(
        dim != connection.dimension for dim in field.codomain
    ):
        raise ValueError(
            "The field's codomain dimensions must match the connection's "
            f"({connection.dimension}); received {field.codomain}."
        )


def validate_kernels(field: TensOps, D: Any) -> None:
    """Validate that a derivative kernel bank matches a field's spatial domain.

    This is the universal precondition shared by every convolution-based
    derivative operator (``d1``/``d2``/``d3``, :func:`partial_derivative`, ...),
    independent of the field's variance: the number of kernels must equal the
    number of spatial dimensions, which must be one of ``{1, 2, 3}``. Variance
    restrictions (e.g. the scalar-only requirement of ``d1``/``d2``/``d3``) are
    the caller's responsibility.

    Args:
        field (TensOps): Field whose derivatives are requested.
        D (Any): Bank of derivative kernels.

    Returns:
        None: An exception is raised when the field and kernels are
        incompatible.

    Raises:
        ValueError: If ``field`` is not a ``TensOps``, its domain is outside
            ``{1, 2, 3}``, or the kernel count mismatches the domain.
    """

    validate_tensops(field)

    if field.domain not in {1, 2, 3}:
        raise ValueError(
            "Unsupported domain dimension. Supported values are {1, 2, 3}.")
    if field.domain != D.shape[0]:
        raise ValueError(
            "The number of derivative kernels must match the field's domain "
            "dimension.")


__all__ = [
    "validate_tensops",
    "validate_metrops",
    "validate_volformops",
    "validate_connectops",
    "validate_codomain_matches_connection",
    "validate_kernels",
]
