"""Directional averaging helpers built on backend convolutions."""

import itertools
from typing import Any

from ..backends import get_backend
from ..kernels import Average
from ..tensors.metrics import MetrOps, VolFormOps
from ..tensors.tensor import TensOps
from ..utils.validation import validate_tensops


def m1(f: TensOps) -> TensOps:
    """Average values along the first spatial axis.

    Args:
        f (TensOps): Scalar field defined on a one-, two-, or
            three-dimensional grid.

    Returns:
        TensOps: Field whose first spatial dimension is reduced by one node.

    Raises:
        ValueError: If ``f`` is not scalar or the domain dimension is
            unsupported.

    Examples:
        >>> import torch
        >>> from tensopsciml.tensors import TensOps
        >>> from tensopsciml.utils.averaging import m1
        >>>
        >>> values = torch.rand((100, 1, 10, 10, 10))
        >>> f = TensOps(values, domain_dim=3, codomain_dim=[1], order=[0, 0])
        >>> m1(f).values.shape
        torch.Size([100, 1, 9, 10, 10])
    """
    validate_tensops(f)

    if sum(f.order) != 0:
        raise ValueError(
            "Directional averaging requires a scalar (order-0) field.")

    values = f.values
    average = Average()
    backend = get_backend()

    if f.domain == 1:
        kernel = average.one_dimension()[0]
        kernel = kernel.to(device=values.device, dtype=values.dtype)
        averaged_values = backend.conv1d(values, kernel, stride=1, padding="valid")
    elif f.domain == 2:
        kernel = average.two_dimensions()[0]
        kernel = kernel.to(device=values.device, dtype=values.dtype)
        averaged_values = backend.conv2d(values, kernel, stride=1, padding="valid")
    elif f.domain == 3:
        kernel = average.three_dimensions()[0]
        kernel = kernel.to(device=values.device, dtype=values.dtype)
        averaged_values = backend.conv3d(values, kernel, stride=1, padding="valid")
    else:
        raise ValueError(f"Unsupported space dimension: {f.domain}.")

    return TensOps(
        averaged_values,
        domain_dim=f.domain,
        codomain_dim=f.codomain,
        order=f.order,
        contravariant_indices=f.contravariant,
        covariant_indices=f.covariant,
    )


def m2(f: TensOps) -> TensOps:
    """Average values along the second spatial axis.

    Args:
        f (TensOps): Scalar field defined on a two- or three-dimensional
            grid.

    Returns:
        TensOps: Field whose second spatial dimension is reduced by one node.

    Raises:
        ValueError: If the domain dimension is one or ``f`` is not scalar.

    Examples:
        >>> import torch
        >>> from tensopsciml.tensors import TensOps
        >>> from tensopsciml.utils.averaging import m2
        >>>
        >>> values = torch.rand((100, 1, 10, 10, 10))
        >>> f = TensOps(values, domain_dim=3, codomain_dim=[1], order=[0, 0])
        >>> m2(f).values.shape
        torch.Size([100, 1, 10, 9, 10])
    """
    validate_tensops(f)

    if sum(f.order) != 0:
        raise ValueError(
            "Directional averaging requires a scalar (order-0) field.")

    values = f.values
    average = Average()
    backend = get_backend()

    if f.domain == 1:
        raise ValueError(
            "Cannot average along the second dimension for a 1D domain.")
    elif f.domain == 2:
        kernel = average.two_dimensions()[1]
        kernel = kernel.to(device=values.device, dtype=values.dtype)
        averaged_values = backend.conv2d(values, kernel, stride=1, padding="valid")
    elif f.domain == 3:
        kernel = average.three_dimensions()[1]
        kernel = kernel.to(device=values.device, dtype=values.dtype)
        averaged_values = backend.conv3d(values, kernel, stride=1, padding="valid")
    else:
        raise ValueError(f"Unsupported space dimension: {f.domain}.")

    return TensOps(
        averaged_values,
        domain_dim=f.domain,
        codomain_dim=f.codomain,
        order=f.order,
        contravariant_indices=f.contravariant,
        covariant_indices=f.covariant,
    )


def m3(f: TensOps) -> TensOps:
    """Average values along the third spatial axis.

    Args:
        f (TensOps): Scalar field defined on a three-dimensional grid.

    Returns:
        TensOps: Field whose third spatial dimension is reduced by one node.

    Raises:
        ValueError: If the domain dimension is not three or ``f`` is not
            scalar.

    Examples:
        >>> import torch
        >>> from tensopsciml.tensors import TensOps
        >>> from tensopsciml.utils.averaging import m3
        >>>
        >>> values = torch.rand((100, 1, 10, 10, 10))
        >>> f = TensOps(values, domain_dim=3, codomain_dim=[1], order=[0, 0])
        >>> m3(f).values.shape
        torch.Size([100, 1, 10, 10, 9])
    """
    validate_tensops(f)

    if sum(f.order) != 0:
        raise ValueError(
            "Directional averaging requires a scalar (order-0) field.")

    values = f.values
    average = Average()
    backend = get_backend()

    if f.domain == 1:
        raise ValueError(
            "Cannot average along the third dimension for a 1D domain.")
    if f.domain == 2:
        raise ValueError(
            "Cannot average along the third dimension for a 2D domain.")
    elif f.domain == 3:
        kernel = average.three_dimensions()[2]
        kernel = kernel.to(device=values.device, dtype=values.dtype)
        averaged_values = backend.conv3d(values, kernel, stride=1, padding="valid")
    else:
        raise ValueError(f"Unsupported space dimension: {f.domain}.")

    return TensOps(
        averaged_values,
        domain_dim=f.domain,
        codomain_dim=f.codomain,
        order=f.order,
        contravariant_indices=f.contravariant,
        covariant_indices=f.covariant,
    )


def m(T: TensOps) -> TensOps:
    """Apply directional averaging along every spatial axis present.

    Sequentially applies ``m1``, ``m2``, and ``m3`` (as many as the domain
    has) to each codomain component, so unlike the per-axis helpers this
    accepts a tensor field of arbitrary variance.

    A ``MetrOps`` or ``VolFormOps`` input is restricted onto the coarser grid
    and returned as the same specialized type (not degraded to a plain
    ``TensOps``), so the result can still be passed to functions that require
    it -- e.g. ``lower_index``/``curl`` for a restricted metric, or
    ``cov_curl`` for a restricted volume form. This is the same restriction
    used internally to move a metric or volume form from one operator's grid
    to a coarser one (see :meth:`~tensopsciml.calculus.connections.ConnectOps.levi_civita`
    and :func:`~tensopsciml.calculus.operators.laplacian`); it does not
    recompute the metric-derived quantity (e.g. does not re-invert or
    re-derive ``sqrt(det(g))``), it only restricts the values already held.

    Args:
        T (TensOps): Tensor field to average across all spatial directions.
            May be a ``MetrOps`` or ``VolFormOps``.

    Returns:
        TensOps: Tensor field averaged across all spatial directions, with
            the same specialized type as ``T`` when ``T`` is a ``MetrOps``
            or ``VolFormOps``.

    Raises:
        ValueError: If the domain dimension is outside ``{1, 2, 3}``.
    """
    validate_tensops(T)

    N_data = T.values.shape[0]
    domain, codomain = T.domain, T.codomain
    discretization = list(T.values.shape[-T.domain:])
    backend = get_backend()

    codomain_dims = codomain if isinstance(
        codomain, (list, tuple)) else [codomain]
    target_shape = [N_data] + \
        list(codomain_dims) + [dim - 1 for dim in discretization]

    zeros_kwargs: dict[str, Any] = {}
    dtype = getattr(T.values, "dtype", None)
    if dtype is not None:
        zeros_kwargs["dtype"] = dtype
    device = getattr(T.values, "device", None)
    if device is not None:
        zeros_kwargs["device"] = device

    avg = backend.zeros(target_shape, **zeros_kwargs)

    # Compute the averaging along each spatial direction.
    codomain_range = ([codomain] if isinstance(codomain, int) else codomain)
    index_iter = itertools.product(*map(range, codomain_range))

    if domain == 1:
        for indices_i in index_iter:
            avg[:, *indices_i] = backend.squeeze(m1(
                TensOps(
                    backend.unsqueeze(T.values[:, *indices_i], 1),
                    domain, [1], [0, 0], None, None
                )
            ).values, dim=1)

    elif domain == 2:
        for indices_i in itertools.product(*map(range, codomain_range)):
            avg[:, *indices_i] = backend.squeeze(m1(
                m2(
                    TensOps(
                        backend.unsqueeze(T.values[:, *indices_i], 1),
                        domain, [1], [0, 0], None, None
                    )
                )
            ).values, dim=1)

    elif domain == 3:
        for indices_i in itertools.product(*map(range, codomain_range)):
            avg[:, *indices_i] = backend.squeeze(m1(
                m2(
                    m3(
                        TensOps(
                            backend.unsqueeze(T.values[:, *indices_i], 1),
                            domain, [1], [0, 0], None, None
                        )
                    )
                )
            ).values, dim=1)
    else:
        raise ValueError(f"Unsupported space dimension: {domain}.")

    if isinstance(T, MetrOps):
        return MetrOps(avg, domain_dim=T.domain, codomain_dim=T.codomain)

    if isinstance(T, VolFormOps):
        return VolFormOps._from_values(avg, domain_dim=T.domain)

    return TensOps(
        avg,
        domain_dim=T.domain,
        codomain_dim=T.codomain,
        order=T.order,
        contravariant_indices=T.contravariant,
        covariant_indices=T.covariant,
    )
