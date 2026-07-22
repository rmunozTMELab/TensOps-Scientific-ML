"""Covariant differential operators for tensor fields on a connection."""

from typing import Any

from ..algebra import contraction, self_contraction
from ..backends import get_backend
from ..tensors.metrics import VolFormOps
from ..tensors.tensor import TensOps
from ..utils.averaging import m
from ..utils.validation import (
    validate_codomain_matches_connection,
    validate_connectops,
    validate_tensops,
    validate_volformops,
)
from .connections import ConnectOps
from .operators import partial_derivative


def cov_derivative(T: TensOps, connection: ConnectOps, D: Any) -> TensOps:
    """Compute the covariant derivative of a tensor field.

    For a ``(p, q)``-type tensor (``p`` contravariant, ``q`` covariant
    indices), the general formula is::

        nabla_a T^{b1...bp}_{d1...dq} = d_a T^{b1...bp}_{d1...dq}
            + sum_i Gamma^{bi}_{a r} T^{...r(at bi)...}_{d1...dq}
            - sum_i Gamma^{r}_{a di} T^{b1...bp}_{...r(at di)...}

    ``connection`` (a :class:`ConnectOps`, e.g. ``ConnectOps.levi_civita(metric,
    D)``) is not a tensor, so each correction term is built with a direct
    ``einsum`` against ``connection.values`` instead of going through
    :func:`~tensopsciml.algebra.contraction`, which requires both operands
    to be ``TensOps``.

    Args:
        T (TensOps): Tensor field to differentiate.
        connection (ConnectOps): Affine connection providing ``Gamma^k_{ij}``.
        D (Any): Stack of first-derivative kernels matching ``T``'s domain
            (see :func:`partial_derivative`).

    Returns:
        TensOps: Covariant derivative of ``T``, with one extra covariant
            (derivative) index appended -- same shape as
            ``partial_derivative(T, D)``.

    Raises:
        ValueError: If ``T``'s codomain dimensions don't match the
            connection's, or if the kernel/domain setup is invalid (see
            :func:`partial_derivative`).
    """
    validate_tensops(T)
    validate_connectops(connection)
    validate_codomain_matches_connection(T, connection)

    n_free = sum(T.order)
    grad = partial_derivative(T, D)
    T_m = m(T)

    backend = get_backend()
    free_letters = 'ijklmpqabcdef'[:n_free]
    domain_letters = {1: 'x', 2: 'xy', 3: 'xyz'}[T.domain]
    # CONTRACT: dummy shared by T and Gamma; DERIV: appended derivative axis
    # (Gamma's derivative-direction lower index); NEW: Gamma's other index,
    # which takes over T's slot at ``pos``.
    CONTRACT, DERIV, NEW = 'r', 's', 'u'

    derivative = grad.values

    # + Gamma^{bi}_{a r} T^{...r...}: Gamma's lower index r contracts with
    # T's i-th contravariant axis; Gamma's upper (bi) replaces it in place,
    # and Gamma's other lower (a) becomes the new derivative axis.
    for pos in (T.contravariant or []):
        T_letters = list(free_letters)
        T_letters[pos] = CONTRACT
        T_eq = 'n' + ''.join(T_letters) + domain_letters
        connection_eq = 'n' + NEW + DERIV + CONTRACT + domain_letters

        out_letters = list(free_letters)
        out_letters[pos] = NEW
        out_eq = 'n' + ''.join(out_letters) + DERIV + domain_letters

        derivative = derivative + backend.einsum(
            f'{T_eq},{connection_eq}->{out_eq}', T_m.values, connection.values
        )

    # - Gamma^{r}_{a di} T^{...r...}: Gamma's upper index r contracts with
    # T's i-th covariant axis; Gamma's lower (di) replaces it in place, and
    # Gamma's other lower (a) becomes the new derivative axis.
    for pos in (T.covariant or []):
        T_letters = list(free_letters)
        T_letters[pos] = CONTRACT
        T_eq = 'n' + ''.join(T_letters) + domain_letters
        connection_eq = 'n' + CONTRACT + DERIV + NEW + domain_letters

        out_letters = list(free_letters)
        out_letters[pos] = NEW
        out_eq = 'n' + ''.join(out_letters) + DERIV + domain_letters

        derivative = derivative - backend.einsum(
            f'{T_eq},{connection_eq}->{out_eq}', T_m.values, connection.values
        )

    return TensOps(
        derivative,
        domain_dim=grad.domain,
        codomain_dim=grad.codomain,
        order=grad.order,
        contravariant_indices=grad.contravariant,
        covariant_indices=grad.covariant,
    )


def cov_divergence(T: TensOps, connection: ConnectOps, D: Any, index: int = 0) -> TensOps:
    """Compute the divergence of a tensor field via covariant-derivative contraction.

    The derivative index appended by :func:`cov_derivative` is contracted
    with one CONTRAVARIANT index of ``T``, selected by ``index``. Using the
    covariant derivative (instead of the bare :func:`partial_derivative`) makes
    this tensorial in any coordinate system: the Christoffel corrections in
    ``nabla_a T^{...a...}`` are exactly what is required for the contraction to
    transform as a scalar/tensor rather than depending on the coordinate chart.
    ``connection`` therefore has no default, even for the Cartesian case (see
    :func:`gradient` for the same convention with ``inv_metric``) — pass
    ``ConnectOps.levi_civita(metric, D)`` for the metric in use.

    For a vector ``v^i`` there is only one choice and ``index=0`` gives the
    usual ``nabla_i v^i``. For a higher-order tensor the contracted index
    matters whenever the tensor is not symmetric: different contravariant
    indices generally yield different divergences, so ``index`` selects which
    one. For a symmetric tensor the choices coincide and the default
    ``index=0`` is fine.

    Args:
        T (TensOps): Tensor field whose indices are compatible with
            contraction.
        connection (ConnectOps): Affine connection providing ``Gamma^k_{ij}``
            (see :func:`cov_derivative`). Obtain the metric-compatible
            one via ``ConnectOps.levi_civita(metric, D)``.
        D (Any): Derivative kernels used to compute the covariant derivative.
        index (int): Position, within ``T``'s contravariant index list, of the
            index to contract with the derivative. Defaults to 0 (the first
            contravariant index).

    Returns:
        TensOps: Divergence tensor with appropriately lowered order.

    Examples:
        >>> import torch
        >>> from tensopsciml.calculus import ConnectOps, cov_divergence
        >>> from tensopsciml.kernels import Derivative
        >>> from tensopsciml.tensors import TensOps
        >>> T_torch = torch.rand((100, 2, 10, 10, 10))
        >>> T = TensOps(
        ...     T_torch,
        ...     domain_dim=3,
        ...     codomain_dim=[2],
        ...     order=[1, 0],
        ...     contravariant_indices=[0],
        ...     covariant_indices=None,
        ... )
        >>> D_kernels = Derivative(d1=1, d2=1, d3=1).three_dimensions()
        >>> connection = ConnectOps.levi_civita(metric, D_kernels)
        >>> div_T = cov_divergence(T, connection, D_kernels)
        >>> print(div_T.order)
        [0, 0]
    """

    validate_tensops(T)
    validate_connectops(connection)

    grad = cov_derivative(T, connection, D)
    div = self_contraction(grad,
                           contravariant_index=[index],
                           covariant_index=[len(grad.covariant)-1])

    return div


def cov_curl(T: TensOps, connection: ConnectOps, lc_symbol: Any, volume_form: VolFormOps, D: Any, index: int = 0) -> TensOps:
    """Compute the curl of a general ``(p, q)`` tensor field over one covariant index.

    Generalizes :func:`~tensopsciml.calculus.operators.curl` (which only
    accepts a plain contravariant vector) to an arbitrary tensor::

        (curl T)^{i, ...spectators...} = E^{i j k} nabla_j T_{...k(at index)...}

    ``index`` selects, within ``T.covariant``, the single covariant slot to
    curl over -- it must range over the same dimension as the domain (see
    ``vec_dim`` below). Every other index of ``T`` (contravariant or
    covariant) is left as a free "spectator" index in the output, exactly
    like :func:`cov_divergence`'s own ``index``. Because the correction is
    built from the full :func:`cov_derivative`, those spectator indices get
    their own (correct) Christoffel corrections too -- this is what a naive
    extension of the flat ``curl`` (bare ``partial_derivative`` + Levi-Civita
    contraction) cannot give you once ``T`` carries indices besides the one
    being curled.

    ``T`` must already be fully covariant on the slot being curled -- unlike
    ``curl``, this function does NOT lower any index for you (there is no
    single "the" contravariant index to lower on a general tensor). Call
    ``lower_index(metric, T)`` yourself first if you start from a
    contravariant vector; e.g. ``cov_curl(lower_index(metric, u), ...,
    index=0)`` reproduces plain ``curl(u, metric, ..., ...)`` bit-for-bit --
    see the note on exactness below.

    Exactness note: when ``index`` is the ONLY free index of ``T`` (the
    plain-vector case), this matches the flat ``curl`` to floating-point
    round-off, not just in the small-grid-spacing limit: the Levi-Civita
    connection is torsion-free, so ``Gamma^l_{jk}`` is symmetric in its two
    lower indices, while ``E^{ijk}`` is totally antisymmetric in
    ``(i, j, k)`` -- contracting a symmetric pair against an antisymmetric
    one over the same two slots vanishes identically, ``E^{ijk}
    Gamma^l_{jk} T_l = 0``. This cancellation only removes the Christoffel
    term attached to the curled slot itself; any spectator indices still
    pick up their own (non-cancelling) corrections, so for a genuine
    ``(p, q)`` tensor with spectators, ``cov_curl`` is NOT numerically
    equivalent to a naive flat construction -- the connection is genuinely
    needed there.

    Args:
        T (TensOps): Tensor field with at least one covariant index, on the
            grid of ``T``'s domain.
        connection (ConnectOps): Affine connection providing ``Gamma^k_{ij}``
            (see :func:`cov_derivative`). Obtain the metric-compatible one
            via ``ConnectOps.levi_civita(metric, D)``.
        lc_symbol (TensOps): RAW Levi-Civita symbol ``eps^{ijk}`` (values in
            ``{-1, 0, 1}``, order ``[vec_dim, 0]``), with NO metric scaling
            applied -- this function divides it by ``volume_form`` itself.
            Passing an already ``sqrt(g)``-scaled tensor here (e.g. from
            ``levi_civita_tensor(..., inv_metric=...)``) silently
            double-scales and gives the wrong answer.
        volume_form (VolFormOps): Volume form ``sqrt(det(g))``, on the grid of
            ``cov_derivative(T, connection, D)`` (one node smaller than
            ``T`` per spatial direction).
        D (Any): Derivative kernels used to compute the covariant derivative.
        index (int): Position, within ``T.covariant``, of the index to curl
            over. Defaults to 0.

    Returns:
        TensOps: Curl tensor: the curled slot is replaced by ``vec_dim - 2``
            new contravariant indices from ``E`` (none in 2D, one in 3D);
            every other index of ``T`` is kept unchanged.

    Raises:
        ValueError: If ``T`` has no covariant indices, ``index`` is out of
            range, the curled slot's dimension doesn't match the domain
            dimension, or that dimension isn't 2 or 3.
    """
    validate_tensops(T)
    validate_volformops(volume_form)
    validate_connectops(connection)

    if not T.covariant or index < 0 or index >= len(T.covariant):
        raise ValueError(
            "cov_curl() requires 'index' to select a valid covariant slot of "
            f"T; received index={index}, T.covariant={T.covariant}."
        )

    vec_dim = T.domain
    curled_dim = T.codomain[T.covariant[index]]
    if curled_dim != vec_dim:
        raise ValueError(
            "cov_curl() requires the curled covariant index to range over the "
            f"domain dimension ({vec_dim}); received dimension {curled_dim}."
        )
    if vec_dim not in (2, 3):
        raise ValueError(
            f"cov_curl() only supports a 2D or 3D domain; received {vec_dim}."
        )

    E = lc_symbol / volume_form

    grad = cov_derivative(T, connection, D)  # nabla_j T_{...k(at index)...}
    deriv_position = len(grad.covariant) - 1

    if vec_dim == 2:
        result = contraction(E, grad, [[0, 1], [None]], [[None], [deriv_position, index]])
    else:
        result = contraction(E, grad, [[1, 2], [None]], [[None], [deriv_position, index]])

    return result


def cov_directional_derivative(X: TensOps, T: TensOps, connection: ConnectOps, D: Any) -> TensOps:
    """Compute the covariant directional derivative ``X^c nabla_c T``.

    Contracts the derivative index of :func:`cov_derivative` with a vector
    field ``X`` to differentiate ``T`` along ``X``. Because it uses the
    covariant derivative, the result is tensorial in any coordinate system;
    under the Cartesian hypothesis it reduces to the plain advection
    ``X . grad T``.

    Args:
        X (TensOps): Vector field to differentiate along (first order,
            order [1, 0]).
        T (TensOps): Tensor field to differentiate.
        connection (ConnectOps): Affine connection providing ``Gamma^k_{ij}``
            (see :func:`cov_derivative`).
        D (Any): Derivative kernels used to compute the covariant derivative.

    Returns:
        TensOps: Directional derivative ``X^c nabla_c T``, same type as ``T``.

    Raises:
        ValueError: If ``X`` or ``T`` are not TensOps or ``X`` is not a
            first-order vector.
    """

    validate_tensops(X, T)
    validate_connectops(connection)

    if X.order[0] != 1:
        raise ValueError(
            "cov_directional_derivative() requires X to be a vectorial field "
            f"(order [1, 0]); received order {X.order}.")

    derivative = cov_derivative(T, connection, D)
    return contraction(m(X), derivative, [[0], [None]], [[None], [len(derivative.covariant) - 1]])


def lie_derivative(X: TensOps, T: TensOps, connection: ConnectOps, D: Any) -> TensOps:
    """Compute the Lie derivative of a tensor field ``T`` along a vector ``X``.

    For a ``(p, q)`` tensor the Lie derivative is::

        (L_X T)^{a1...ap}_{b1...bq} = X^c nabla_c T^{a1...ap}_{b1...bq}
            - sum_i (nabla_c X^{ai}) T^{a1...c(at ai)...ap}_{b1...bq}
            + sum_j (nabla_{bj} X^c)  T^{a1...ap}_{b1...c(at bj)...bq}

    Written with covariant derivatives it is manifestly independent of the
    ``connection`` as long as it is TORSION-FREE (the Christoffel terms cancel
    between the three pieces), so passing ``ConnectOps.levi_civita(metric, D)``
    reproduces the usual coordinate Lie derivative. Unlike :func:`cov_derivative`
    it does NOT append a derivative index: the result has the SAME type as ``T``
    (e.g. ``lie_derivative`` of a vector is again a vector -- the Lie bracket
    ``[X, T]``).

    The first term is ``cov_directional_derivative(X, T, connection, D)``; the
    corrections are built from ``nabla X = cov_derivative(X, connection, D)`` (a
    ``(1, 1)`` tensor) contracted against ``m(T)`` -- both live on the reduced
    derivative grid, so the three pieces line up node-for-node.

    Args:
        X (TensOps): Vector field to flow along (first order, order [1, 0]).
        T (TensOps): Tensor field to differentiate.
        connection (ConnectOps): Torsion-free affine connection providing
            ``Gamma^k_{ij}`` (e.g. ``ConnectOps.levi_civita(metric, D)``).
        D (Any): Derivative kernels used to compute the covariant derivatives.

    Returns:
        TensOps: Lie derivative ``L_X T``, same type as ``T``, on the reduced
            derivative grid.

    Raises:
        ValueError: If ``X`` or ``T`` are not TensOps, ``X`` is not a
            first-order vector, or ``T``'s codomain dimensions don't match the
            connection's.
    """
    validate_tensops(X, T)
    validate_connectops(connection)
    validate_codomain_matches_connection(T, connection)

    if X.order[0] != 1:
        raise ValueError(
            "lie_derivative() requires X to be a vectorial field "
            f"(order [1, 0]); received order {X.order}.")

    n_free = sum(T.order)

    # First term X^c nabla_c T. For a scalar this is the entire derivative.
    first_term = cov_directional_derivative(X, T, connection, D)
    if n_free == 0:
        return first_term

    T_m = m(T)
    cov_X = cov_derivative(X, connection, D)  # (nabla X)^a_c: axis 0 = a, axis 1 = c

    backend = get_backend()
    free_letters = 'ijklmpqabcdef'[:n_free]
    domain_letters = {1: 'x', 2: 'xy', 3: 'xyz'}[T.domain]
    CONTRACT, NEW = 'r', 's'

    result = first_term.values

    # - (nabla_c X^{ai}) T^{...c(at ai)...}: nabla X's derivative index c
    # contracts with T's i-th contravariant axis; X's upper index a replaces it.
    for pos in (T.contravariant or []):
        T_letters = list(free_letters)
        T_letters[pos] = CONTRACT
        T_eq = 'n' + ''.join(T_letters) + domain_letters
        cov_X_eq = 'n' + NEW + CONTRACT + domain_letters

        out_letters = list(free_letters)
        out_letters[pos] = NEW
        out_eq = 'n' + ''.join(out_letters) + domain_letters

        result = result - backend.einsum(
            f'{T_eq},{cov_X_eq}->{out_eq}', T_m.values, cov_X.values
        )

    # + (nabla_{bj} X^c) T^{...c(at bj)...}: X's upper index c contracts with
    # T's j-th covariant axis; nabla X's derivative index bj replaces it.
    for pos in (T.covariant or []):
        T_letters = list(free_letters)
        T_letters[pos] = CONTRACT
        T_eq = 'n' + ''.join(T_letters) + domain_letters
        cov_X_eq = 'n' + CONTRACT + NEW + domain_letters

        out_letters = list(free_letters)
        out_letters[pos] = NEW
        out_eq = 'n' + ''.join(out_letters) + domain_letters

        result = result + backend.einsum(
            f'{T_eq},{cov_X_eq}->{out_eq}', T_m.values, cov_X.values
        )

    return TensOps(
        result,
        domain_dim=T.domain,
        codomain_dim=T.codomain,
        order=T.order,
        contravariant_indices=T.contravariant,
        covariant_indices=T.covariant,
    )