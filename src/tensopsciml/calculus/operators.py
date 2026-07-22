"""Differential operators implemented on top of backend convolutions."""

import itertools
from typing import Any

from ..algebra import (
    contraction,
    lower_index,
    raise_index,
    self_contraction,
)
from ..backends import get_backend
from ..tensors.metrics import MetrOps, VolFormOps
from ..tensors.tensor import TensOps
from ..utils.averaging import m
from ..utils.validation import validate_kernels, validate_tensops


def d1(f: TensOps, D: Any) -> TensOps:
    """Compute the finite-difference derivative along the first axis.

    Args:
        f (TensOps): Scalar field defined on a one-, two-, or three-
            dimensional grid.
        D (Any): Bank of derivative kernels whose count matches the field's
            domain.

    Returns:
        TensOps: Field containing the derivative df/dx1 with reduced spatial
            resolution.

    Raises:
        ValueError: If ``f`` is not scalar or the kernels are incompatible.

    Examples:
        >>> import torch
        >>> from tensopsciml.calculus import d1
        >>> from tensopsciml.kernels import Derivative
        >>> from tensopsciml.tensors import TensOps
        >>> values = torch.rand((100, 1, 10, 10, 10))
        >>> f = TensOps(values, domain_dim=3, codomain_dim=[1], order=[0, 0])
        >>> kernels = Derivative(d1=1, d2=None, d3=None).three_dimensions()
        >>> d1(f, kernels).values.shape
        torch.Size([100, 1, 8, 8, 8])
    """
    validate_kernels(f, D)
    if not f.is_scalar:
        raise ValueError("d1() requires a scalar (order-0) field.")

    backend = get_backend()

    conv_funcs = {1: backend.conv1d,
                  2: backend.conv2d, 3: backend.conv3d}[f.domain]
    dfd1 = conv_funcs(f.values, D[0], stride=1, padding="valid")

    return TensOps(
        dfd1,
        domain_dim=f.domain,
        codomain_dim=f.codomain,
        order=f.order,
        contravariant_indices=f.contravariant,
        covariant_indices=f.covariant,
    )


def d2(f: TensOps, D: Any) -> TensOps:
    """Compute the finite-difference derivative along the second axis.

    Args:
        f (TensOps): Scalar field defined on a two- or three-dimensional
            grid.
        D (Any): Bank of derivative kernels whose count matches the field's
            domain.

    Returns:
        TensOps: Field containing the derivative df/dx2.

    Raises:
        ValueError: If the domain is one-dimensional, ``f`` is not scalar, or
            the kernels mismatch.

    Examples:
        >>> import torch
        >>> from tensopsciml.calculus import d2
        >>> from tensopsciml.kernels import Derivative
        >>> from tensopsciml.tensors import TensOps
        >>> values = torch.rand((100, 1, 10, 10, 10))
        >>> f = TensOps(values, domain_dim=3, codomain_dim=[1], order=[0, 0])
        >>> kernels = Derivative(d1=1, d2=1, d3=1).three_dimensions()
        >>> d2(f, kernels).values.shape
        torch.Size([100, 1, 8, 8, 8])
    """
    validate_kernels(f, D)
    if not f.is_scalar:
        raise ValueError("d2() requires a scalar (order-0) field.")

    backend = get_backend()

    if f.domain == 1:
        raise ValueError(
            "d2() cannot differentiate along the second dimension for a 1D "
            "domain.")

    conv_funcs = {2: backend.conv2d, 3: backend.conv3d}
    dfd2 = conv_funcs[f.domain](f.values, D[1], stride=1, padding="valid")

    return TensOps(
        dfd2,
        domain_dim=f.domain,
        codomain_dim=f.codomain,
        order=f.order,
        contravariant_indices=f.contravariant,
        covariant_indices=f.covariant,
    )


def d3(f: TensOps, D: Any) -> TensOps:
    """Compute the finite-difference derivative along the third axis.

    Args:
        f (TensOps): Scalar field defined on a three-dimensional grid.
        D (Any): Bank of derivative kernels whose count matches the field's
            domain.

    Returns:
        TensOps: Field containing the derivative df/dx3.

    Raises:
        ValueError: If the domain is not three-dimensional, ``f`` is not
            scalar, or the kernels mismatch.

    Examples:
        >>> import torch
        >>> from tensopsciml.calculus import d3
        >>> from tensopsciml.kernels import Derivative
        >>> from tensopsciml.tensors import TensOps
        >>> values = torch.rand((100, 1, 10, 10, 10))
        >>> f = TensOps(values, domain_dim=3, codomain_dim=[1], order=[0, 0])
        >>> kernels = Derivative(d1=1, d2=1, d3=1).three_dimensions()
        >>> d3(f, kernels).values.shape
        torch.Size([100, 1, 8, 8, 8])
    """
    validate_kernels(f, D)
    if not f.is_scalar:
        raise ValueError("d3() requires a scalar (order-0) field.")

    backend = get_backend()

    if f.domain in {1, 2}:
        raise ValueError(
            "d3() cannot differentiate along the third dimension for a "
            f"{f.domain}D domain.")

    dfd3 = backend.conv3d(f.values, D[2], stride=1, padding="valid")

    return TensOps(
        dfd3,
        domain_dim=f.domain,
        codomain_dim=f.codomain,
        order=f.order,
        contravariant_indices=f.contravariant,
        covariant_indices=f.covariant,
    )


def partial_derivative(T: TensOps, D: Any) -> TensOps:
    """Compute the coordinate partial derivative ``d_c T`` of a tensor field.

    Differentiates every component along each spatial direction and appends a
    new COVARIANT (derivative) index ``c``::

        (d T)^{a...}_{b..., c} = d_c T^{a...}_{b...}

    This is the coordinate/partial derivative ``d``, NOT the gradient -- the
    gradient is defined only for a scalar field (see :func:`gradient`). In
    curvilinear coordinates ``d T`` is not tensorial; the tensorial version that
    adds the Christoffel corrections is :func:`cov_derivative`. Under the
    library's Cartesian hypothesis (``g = delta``) both coincide.

    For a scalar field ``f`` the result is the differential ``df = d_i f`` (a
    covariant 1-form).

    Args:
        T (TensOps): Tensor field of arbitrary variance.
        D (Any): Stack of first-derivative kernels whose count matches the
            domain.

    Returns:
        TensOps: Partial-derivative tensor with one extra covariant (derivative)
            index appended.

    Raises:
        ValueError: If the kernels do not match the field's domain dimension.

    Examples:
        >>> import torch
        >>> from tensopsciml.calculus import partial_derivative
        >>> from tensopsciml.kernels import Derivative
        >>> from tensopsciml.tensors import TensOps
        >>> values = torch.rand((100, 2, 2, 10, 10, 10))
        >>> tensor = TensOps(
        ...     values,
        ...     domain_dim=3,
        ...     codomain_dim=[2, 2],
        ...     order=[1, 1],
        ...     contravariant_indices=[0],
        ...     covariant_indices=[1],
        ... )
        >>> kernels = Derivative(d1=1, d2=1, d3=1).three_dimensions()
        >>> partial_derivative(tensor, kernels).order
        [1, 2]
    """

    validate_kernels(T, D)

    backend = get_backend()

    N_data = T.values.shape[0]
    order = sum(T.order)
    domain, codomain = T.domain, T.codomain

    # Compute the shape of the partial-derivative tensor based on the tensor order and
    # dimensions. The derivative introduces a trailing codomain axis of size
    # ``domain`` (a covariant index). For a scalar field the result is exactly
    # that axis; otherwise it is appended to the existing codomain. This holds
    # for every domain, including 1D, where the derivative axis has size 1.
    new_codomain = [domain] if order == 0 else list(codomain) + [domain]
    discretization = [dim - 1 for dim in T.values.shape[-domain:]]
    dimensions = [N_data] + new_codomain + discretization

    zeros_kwargs = {}
    if hasattr(T.values, "dtype"):
        zeros_kwargs["dtype"] = T.values.dtype
    if hasattr(T.values, "device"):
        zeros_kwargs["device"] = T.values.device

    derivative_values = backend.zeros(dimensions, **zeros_kwargs)

    # Handle convolution and compute the partial derivative
    if domain == 1:
        if order == 0:
            derivative_values = backend.conv1d(
                T.values, D[0], stride=1, padding="valid", groups=1
            )
        else:
            for indices_i in itertools.product(
                *map(range, ([codomain] if isinstance(codomain, int)
                             else codomain))
            ):
                T_ = backend.unsqueeze(T.values[:, *(indices_i)], 1)
                derivative_values[:, *list(indices_i)] = backend.conv1d(
                    T_, D[0], stride=1, padding="valid", groups=1
                )

    elif domain == 2:
        for indices_i in itertools.product(
            *map(range, ([codomain] if isinstance(codomain, int)
                         else codomain))
        ):
            if order == 0:
                T_ = backend.unsqueeze(T.values[:, *list(indices_i)], 1)
                derivative_values = backend.conv2d(
                    T_, backend.unsqueeze(backend.squeeze(D), 1), stride=1,
                    padding="valid", groups=1
                )
            else:
                T_ = backend.unsqueeze(T.values[:, *list(indices_i)], 1)
                derivative_values[:, *list(indices_i)] = backend.conv2d(
                    T_, backend.unsqueeze(backend.squeeze(D), 1), stride=1,
                    padding="valid", groups=1
                )

    else:  # domain == 3; validate_kernels() already ruled out anything else.
        for indices_i in itertools.product(
            *map(range, ([codomain] if isinstance(codomain, int)
                         else codomain))
        ):
            if order == 0:
                T_ = backend.unsqueeze(T.values[:, *list(indices_i)], 1)
                derivative_values = backend.conv3d(
                    T_, backend.unsqueeze(backend.squeeze(D), 1), stride=1,
                    padding="valid", groups=1
                )
            else:
                T_ = backend.unsqueeze(T.values[:, *(indices_i)], 1)
                derivative_values[:, *list(indices_i)] = backend.conv3d(
                    T_, backend.unsqueeze(backend.squeeze(D), 1), stride=1,
                    padding="valid", groups=1
                )

    new_contravariant = T.contravariant
    new_covariant = (T.covariant or []) + [sum(T.order)]

    return TensOps(
        derivative_values,
        domain_dim=T.domain,
        codomain_dim=new_codomain,
        order=T.order[:-1] + [T.order[-1] + 1],
        contravariant_indices=new_contravariant,
        covariant_indices=new_covariant)


def gradient(f: TensOps, inv_metric: Any, D: Any) -> TensOps:
    """Compute the gradient ``grad f = nabla f`` of a SCALAR field.

    The gradient is defined only for a scalar field ``f``. It is the
    CONTRAVARIANT vector ``grad f = nabla f = g^{ij} d_j f``, i.e. the
    coordinate partial derivative ``d_i f`` (the covariant differential ``df``
    returned by :func:`partial_derivative`) with its index raised by the
    inverse metric. Raising an index is only well-defined given a metric --
    even the Euclidean/Cartesian case must be supplied explicitly, so
    ``inv_metric`` has no default. For the coordinate partial derivative of a
    general tensor field use :func:`partial_derivative` instead.

    Args:
        f (TensOps): Scalar field (order ``[0, 0]``).
        inv_metric (MetrOps): Inverse (contravariant) metric ``g^{ij}``
            (order ``[2, 0]``) used to raise the derivative index. Obtain it
            via ``inverse(metric)``; pass the Euclidean metric's inverse
            (identity) explicitly for the Cartesian case.
        D (Any): Stack of first-derivative kernels whose count matches the
            domain.

    Returns:
        TensOps: Contravariant gradient vector, order ``[1, 0]``.

    Raises:
        ValueError: If ``f`` is not a scalar field, or if ``inv_metric`` is not
            a valid inverse metric (order ``[2, 0]``).

    Examples:
        >>> import torch
        >>> from tensopsciml.calculus import gradient
        >>> from tensopsciml.kernels import Derivative
        >>> from tensopsciml.tensors import TensOps
        >>> values = torch.rand((100, 1, 10, 10, 10))
        >>> f = TensOps(values, domain_dim=3, codomain_dim=[1], order=[0, 0])
        >>> kernels = Derivative(d1=1, d2=1, d3=1).three_dimensions()
        >>> gradient(f, inv_metric, kernels).order
        [1, 0]
    """
    validate_tensops(f)

    if not f.is_scalar:
        raise ValueError(
            "gradient() is only defined for a scalar field (order [0, 0]); "
            f"received order {f.order}. For the coordinate partial derivative "
            "of a tensor field use partial_derivative().")

    df = partial_derivative(f, D)  # covariant differential d_i f (df)

    return raise_index(inv_metric, df)


def curl(u: TensOps, metric: Any, lc_symbol: TensOps, volume_form: VolFormOps, D: Any) -> TensOps:
    """Compute the curl of a vector field using the Levi-Civita tensor.

    Formula (valid in any coordinate system, metric ``g_{ij}``)::

        (curl u)^i = (1/sqrt(g)) E^{ijk} d_j u_k,   u_k = g_{kl} u^l

    The fully contravariant Levi-Civita tensor ``E = (1/sqrt(g)) eps^{ijk}`` is
    built here by dividing the raw symbol ``lc_symbol`` by ``volume_form``; do
    NOT pass an already ``sqrt(g)``-scaled tensor. ``u`` is lowered to its
    covariant form ``u_k = g_{kl} u^l`` *before* differentiating, not after: in
    curvilinear coordinates ``d_j(g_{kl} u^l) = (d_j g_{kl}) u^l + g_{kl} d_j
    u^l`` picks up an extra metric-derivative term that lowering the
    already-differentiated ``partial_derivative(u, D)`` with the metric at a
    single point would silently drop.

    Args:
        u (TensOps): Contravariant vector field (order ``[1, 0]``).
        metric (MetrOps): Covariant metric ``g_{ij}``, on ``u``'s own grid,
            used to lower ``u``'s vector index before differentiating.
        lc_symbol (TensOps): Raw Levi-Civita symbol ``eps^{ijk}`` (values in
            ``{-1, 0, 1}``), on the grid of ``partial_derivative(u, D)`` (one
            node smaller than ``u`` per spatial direction). It is scaled by
            ``volume_form`` internally.
        volume_form (VolFormOps): Volume form ``sqrt(det(g))`` used to scale
            ``lc_symbol`` into the Levi-Civita tensor.
        D (Any): Derivative kernels used to compute the partial derivative.

    Returns:
        TensOps: Curl tensor whose rank depends on the spatial dimension.

    Raises:
        ValueError: If ``u`` is not a contravariant vector field (order
            ``[1, 0]``), or its component count does not match the domain
            dimension.
    """
    validate_tensops(u)

    if u.order != [1, 0]:
        raise ValueError(
            "curl() requires a contravariant vector field (order [1, 0]); "
            f"received order {u.order}.")

    vec_dim = u.codomain[0]
    if vec_dim < 2:
        raise ValueError(
            "curl() requires a vector field with at least 2 components; "
            f"received {vec_dim}.")
    if vec_dim != u.domain:
        raise ValueError(
            "curl() requires the vector field's component count to match the "
            f"spatial domain dimension; received {vec_dim} components over a "
            f"{u.domain}D domain.")

    E = lc_symbol / volume_form

    u_lowered = lower_index(metric, u)
    grad = partial_derivative(u_lowered, D)  # d_j u_k, fully covariant [0, 2]

    if vec_dim == 2:
        # 2D curl: E^{jk} d_j u_k -> scalar; contract both indices of E.
        result = contraction(E, grad, [[0, 1], [None]], [[None], [1, 0]])
    else:
        # 3D curl: E^{ijk} d_j u_k -> vector free in i; leave E's position 0.
        result = contraction(E, grad, [[1, 2], [None]], [[None], [1, 0]])

    return result


def autograd(
    T: TensOps,
    S: TensOps,
    retain_graph: bool = True,
    allow_unused: bool = True,
    create_graph: bool = False,
) -> TensOps:
    """Compute the analytical Jacobian dT/dS using autograd.

    ``torch.autograd.grad`` with ``grad_outputs=ones_like(y)`` is a
    vector-Jacobian product: it sums over every component of ``T`` before
    differentiating. To get the true per-component Jacobian, one
    vector-Jacobian product is evaluated per component of ``T``'s codomain,
    each selecting a single component of ``T`` via a one-hot
    ``grad_outputs`` mask. Batch and domain (spatial) positions are still
    summed jointly per call, which is exact as long as ``T`` is a pointwise
    function of ``S`` (no coupling across batch elements or grid points) —
    the same assumption already relied upon for the batch dimension.

    Args:
        T (TensOps): Numerator tensor.
        S (TensOps): Denominator tensor sharing the same domain dimension.
        retain_graph (bool): Whether to retain the computation graph.
        allow_unused (bool): Whether unused tensors are permitted.
        create_graph (bool): Whether to construct higher-order computation
            graphs.

    Returns:
        TensOps: Tensor containing dT/dS with combined codomain. If ``T``
            is scalar, its degenerate codomain axis is squeezed away so the
            result keeps ``S``'s own shape, matching the scalar-``T``
            convention.

    Raises:
        ValueError: If the tensors do not share the same domain dimension.

    Examples:
        >>> dTS = autograd(T, S)
    """
    if T.domain != S.domain:
        raise ValueError(
            "T and S must share the same domain dimension for autograd dT/dS.")

    if not S.values.requires_grad:
        S.values.requires_grad_(True)

    N_data = T.values.shape[0]
    discretization = list(T.values.shape[-T.domain:])

    dT_codomain = [T.codomain] if isinstance(T.codomain, int) else T.codomain
    dS_codomain = [S.codomain] if isinstance(S.codomain, int) else S.codomain

    backend = get_backend()

    zeros_kwargs = {}
    if hasattr(S.values, "dtype"):
        zeros_kwargs["dtype"] = S.values.dtype
    if hasattr(S.values, "device"):
        zeros_kwargs["device"] = S.values.device

    derivate = backend.zeros(
        [N_data] + dT_codomain + dS_codomain + discretization, **zeros_kwargs)

    for t_indices in itertools.product(*map(range, dT_codomain)):
        mask = backend.zeros_like(T.values)
        mask[:, *t_indices] = 1
        component_grad = backend.autograd(
            T.values,
            S.values,
            grad_outputs=mask,
            retain_graph=retain_graph,
            allow_unused=allow_unused,
            create_graph=create_graph,
        )
        if component_grad is None:
            raise ValueError(
                "autograd() got None for dT/dS: S is not connected to T's "
                "computation graph (no path from S.values to T.values)."
            )
        derivate[:, *t_indices] = component_grad

    # A scalar operand (tensorial order 0) contributes a degenerate size-1
    # codomain axis that doesn't correspond to a real tensorial index; squeeze
    # it away so the result's shape matches its declared order. If both T and S
    # are scalar, keep exactly one size-1 channel (S's), matching the convention
    # that codomain == [1] is a literal axis. Scalarness must be tested on the
    # tensorial ORDER, not on codomain == [1]: in a 1D domain a genuine
    # vector/covector also has codomain [1], and its single index is real -- it
    # must not be squeezed (otherwise order [1, 1] would collapse to codomain
    # [1] and fail validation).
    t_scalar = sum(T.order) == 0
    s_scalar = sum(S.order) == 0

    if t_scalar and s_scalar:
        derivate = backend.squeeze(derivate, dim=1)
        codomain = [1]
    elif t_scalar:
        derivate = backend.squeeze(derivate, dim=1)
        codomain = dS_codomain
    elif s_scalar:
        derivate = backend.squeeze(derivate, dim=1 + len(dT_codomain))
        codomain = dT_codomain
    else:
        codomain = dT_codomain + dS_codomain

    # Differentiating w.r.t. S INVERTS the variance of S's indices: a derivative
    # ∂/∂S^i (contravariant) is covariant, and ∂/∂S_i (covariant) is
    # contravariant (the standard "∂ w.r.t. a covariant index → contravariant"
    # rule, valid in any coordinate system — no metric involved). The numerator
    # T keeps its own variance; only the denominator's indices flip.
    new_contravariant = (T.contravariant if T.contravariant is not None else []) + \
        [x + sum(T.order)
         for x in (S.covariant if S.covariant is not None else [])]
    new_covariant = (T.covariant if T.covariant is not None else []) + \
        [x + sum(T.order)
         for x in (S.contravariant if S.contravariant is not None else [])]

    return TensOps(
        derivate,
        domain_dim=T.domain,
        codomain_dim=codomain,
        order=[T.order[0]+S.order[1], T.order[1]+S.order[0]],
        contravariant_indices=None if new_contravariant == [] else new_contravariant,
        covariant_indices=None if new_covariant == [] else new_covariant
    )

def divergence(u: TensOps, volume_form: VolFormOps, D: Any) -> TensOps:
    """Compute the divergence of a contravariant vector field.

    Uses the Voss-Weyl identity, valid in any coordinate system with metric
    ``g_{ij}``::

        div u = (1/sqrt(g)) d_i(sqrt(g) u^i)

    This avoids needing an explicit connection: the Christoffel trace
    ``Gamma^i_{ij} = d_j(sqrt(g)) / sqrt(g)`` that a full covariant
    divergence would require is already accounted for by scaling ``u`` with
    the volume form ``sqrt(g)`` before differentiating, then contracting the
    derivative index with ``u``'s vector index and undoing the scaling.
    ``volume_form`` is needed at two different grid resolutions -- once on
    ``u``'s own grid to scale it, and once (restricted via :func:`m`) on the
    differentiated grid, one node smaller per spatial direction -- the same
    pattern used in :func:`laplacian`.

    Args:
        u (TensOps): Contravariant vector field, order ``[1, 0]``.
        volume_form (VolFormOps): Volume form ``sqrt(det(g))`` associated with
            the metric in use, on ``u``'s own grid.
        D (Any): Derivative kernels used to compute the partial derivative.

    Returns:
        TensOps: Scalar divergence field, order ``[0, 0]``.

    Raises:
        ValueError: If ``u`` is not a contravariant vector field (order
            ``[1, 0]``).
    """
    validate_tensops(u)

    if u.order != [1, 0]:
        raise ValueError(
            "divergence() requires a contravariant vector field (order "
            f"[1, 0]); received order {u.order}."
        )

    scaled = partial_derivative(volume_form * u, D)
    div = self_contraction(
        scaled,
        [len(scaled.contravariant) - 1],
        [len(scaled.covariant) - 1],
    )

    return div / m(volume_form)


def advection(u: TensOps, f: TensOps, D: Any) -> TensOps:
    """Compute the advection term u dot grad f.

    Args:
        u (TensOps): Advecting vector field (first order).
        f (TensOps): Advected scalar field.
        D (Any): Derivative kernels used in the partial derivative.

    Returns:
        TensOps: Contraction result representing the advection term.

    Raises:
        ValueError: If u or f are not TensOps, u is not first-order, or f is not a scalar.
    """

    validate_tensops(u, f)

    if u.order[0] != 1:
        raise ValueError(
            "advection() requires the advecting field u to be first order "
            f"(order [1, 0]); received order {u.order}.")

    if f.order != [0, 0]:
        raise ValueError(
            "advection() requires the advected field f to be a scalar "
            f"(order [0, 0]); received order {f.order}.")

    grad = partial_derivative(f, D)
    return contraction(m(u), grad, [[0], [None]], [[None], [len(grad.covariant) - 1]])


def laplacian(f: TensOps, D: Any, inv_metric: MetrOps, volume_form: VolFormOps) -> TensOps:
    """Compute the Laplacian of a scalar field.

    Uses the Laplace-Beltrami identity ``Delta f = (1/sqrt(g)) d_i(sqrt(g)
    g^{ij} d_j f)``: ``gradient(f, inv_metric, D)`` gives the contravariant
    ``g^{ij} d_j f``, which is then scaled by the volume form ``sqrt(g)``,
    differentiated again, and self-contracted (that second ``d_i`` only
    appends a derivative axis; it doesn't sum over it on its own -- the
    self-contraction is what turns it into the divergence ``d_i(...)``).
    ``volume_form`` is needed at two different grid resolutions -- once to
    scale ``grad`` (its own grid) and once to divide the twice-differentiated
    result (one node smaller) -- so it's restricted with :func:`m` for the
    second use, the same way :func:`ConnectOps.levi_civita` restricts the
    metric before inverting it.

    Args:
        f (TensOps): Zero-order tensor representing the scalar field.
        D (Any): Derivative kernels used in partial-derivative computations.
        inv_metric (MetrOps): Inverse (contravariant) metric ``g^{ij}``
            (order ``[2, 0]``), on the grid of ``partial_derivative(f, D)``
            (one node smaller than ``f`` per spatial direction), used to
            raise :func:`gradient`'s index. Obtain it via ``inverse(metric)``.
        volume_form (VolFormOps): Volume form ``sqrt(det(g))`` associated
            with the same metric, on the grid of ``gradient(f, inv_metric,
            D)``. Obtain it via ``VolFormOps(metric)``.

    Returns:
        TensOps: Laplacian tensor representing Delta f.
    """

    if not f.is_scalar:
        raise ValueError(
            "laplacian() requires a scalar field (order [0, 0]); received "
            f"order {f.order}.")

    grad = gradient(f, inv_metric, D)
    scaled_grad = partial_derivative(volume_form * grad, D)
    div_scaled_grad = self_contraction(
        scaled_grad,
        [len(scaled_grad.contravariant) - 1],
        [len(scaled_grad.covariant) - 1],
    )

    return div_scaled_grad / m(volume_form)


__all__ = [
    "d1", "d2", "d3", "partial_derivative", "gradient", "curl",
    "divergence", "laplacian",
    "autograd", "advection",
]
