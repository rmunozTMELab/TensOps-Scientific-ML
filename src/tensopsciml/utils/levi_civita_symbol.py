"""Levi-Civita symbol generation utilities."""

import itertools
from typing import Any, Sequence

from ..backends import get_backend
from .indices import levi_civita_indices_value


def levi_civita_symbol(
    N_data: int,
    order: int,
    discretization: list[int],
    *,
    device: Any | None = None,
    dtype: Any | None = None,
) -> Any:
    """Generate the raw Levi-Civita symbol values.

    The Levi-Civita symbol is a tensor density, not a tensor: it doesn't
    transform correctly under a change of metric on its own. This function
    only builds the raw, backend-native array of ``+1``/``-1``/``0``
    values; wrap it with a metric (see :func:`levi_civita_tensor`) to turn
    it into an actual ``TensOps`` tensor.

    Args:
        N_data (int): Number of samples in the batch dimension.
        order (int): Order of the Levi-Civita symbol. Must be greater than
            ``1``.
        discretization (list[int]): Spatial discretization for each domain
            axis.
        device (Any | None): Optional target device for the underlying
            tensor.
        dtype (Any | None): Optional target dtype for the underlying
            tensor.

    Returns:
        Any: Backend-native array of shape
            ``[N_data] + order * [order] + discretization`` containing the
            Levi-Civita symbol values.

    Raises:
        ValueError: If ``discretization`` is invalid or ``order`` is less
            than or equal to ``1``.

    Examples:
        >>> N_data = 10
        >>> order = 3
        >>> discretization = [20, 20, 20]
        >>> e = levi_civita_symbol(N_data, order, discretization)
        >>> e.shape
        torch.Size([10, 3, 3, 3, 20, 20, 20])
    """
    # Previous validations to check if the tensor fulfil the conditions
    if not isinstance(discretization, list):
        if isinstance(discretization, Sequence) and not isinstance(
            discretization, (str, bytes)
        ):
            discretization = list(discretization)
        else:
            raise ValueError("Discretization must be a list.")

    if not isinstance(order, int):
        raise ValueError("Order of LC symbol must be an integer.")

    if order <= 1:
        raise ValueError("LC symbol order has to be greater than 1.")

    backend = get_backend()

    # Initialize a tensor to hold Levi-Civita values with the specified shape
    e_values = backend.zeros(
        [N_data] + order * [order] + discretization,
        device=device,
        dtype=dtype,
    )

    # Iterate over all possible index combinations for Levi-Civita calculation
    for indices in itertools.product(*map(range, order * [order])):
        e_values[:, *indices, :] = levi_civita_indices_value(list(indices))

    return e_values
