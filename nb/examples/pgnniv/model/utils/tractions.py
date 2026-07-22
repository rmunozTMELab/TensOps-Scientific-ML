"""Boundary traction computation via P . N for the 2D model.

Computes reference-configuration traction vectors T^i = P^{ij} N_j at domain
boundaries by contracting the first Piola-Kirchhoff stress P^{ij} (order [2, 0])
with a covariant reference boundary normal N_j. This is the 2D form of the
P . N traction used in the psi_pgnniv reference. Contracting with the full
stress (not just its diagonal) makes the traction a genuine vector, so its
tangential (shear) component is exposed on the boundary.

The bulk spatial layout is (Ny, Nx): trailing axis -2 is y, axis -1 is x, so the
positive faces x=xN and y=yN are the LAST index along each axis. The sliced axis
is kept as a singleton so the traction stays a 2D-domain TensOps, matching the
ground-truth Tx / Ty laid out in the notebook.
"""

import tensopsciml as to
from tensopsciml.tensors import TensOps

from .normals import make_boundary_normals


def _traction_on_face(p_pred: TensOps, normal: TensOps,
                      slice_dim: int, slice_idx: int) -> TensOps:
    """Compute T^i = P^{ij} N_j and extract one boundary-face slice.

    Args:
        p_pred:    First Piola-Kirchhoff stress P^{ij} [2, 0], shape
                   (N, 2, 2, Ny, Nx).
        normal:    Covariant reference unit normal N_j [0, 1], shape
                   (N, 2, Ny, Nx).
        slice_dim: Spatial axis to slice (0=y, 1=x).
        slice_idx: Index along that axis (-1 for the positive/last face).

    Returns:
        TensOps: Traction vector T^i [1, 0] at the requested face, with a
                 singleton dimension reinserted at slice_dim.
    """
    # T^i = P^{ij} N_j: contract P's reference (second) contravariant index j
    # with the normal's covariant index, leaving P's spatial index i free.
    t = to.contraction(p_pred, normal, [[1], [None]], [[None], [0]])

    # Map slice_dim (0, 1) -> trailing tensor axis (-2, -1).
    axis = -(2 - slice_dim)
    idx = slice_idx if slice_idx >= 0 else t.values.shape[axis] - 1
    t_ = t.values.select(axis, idx).unsqueeze(axis)

    return TensOps(t_, domain_dim=2, codomain_dim=[2], order=[1, 0],
                   contravariant_indices=[0], covariant_indices=None)


def compute_boundary_tractions(p_pred: TensOps):
    """Compute predicted tractions at the positive x and y boundaries.

    Implements T^i = P^{ij} N_j with covariant reference boundary normals. The
    x-face traction is sampled at x=xN and the y-face traction at y=yN, matching
    where the ground-truth Tx / Ty are defined in this dataset.

    Args:
        p_pred: Predicted first Piola-Kirchhoff stress P^{ij} [2, 0], shape
            (N, 2, 2, Ny, Nx).

    Returns:
        Tuple (tx_pred, ty_pred) as TensOps [1, 0]:
            - tx_pred: shape (N, 2, Ny, 1) -- (P_xx, P_yx) at x=xN, along y.
            - ty_pred: shape (N, 2, 1, Nx) -- (P_xy, P_yy) at y=yN, along x.
    """
    N = p_pred.values.shape[0]
    d0 = p_pred.values.shape[-2]  # y (Ny)
    d1 = p_pred.values.shape[-1]  # x (Nx)
    device = p_pred.values.device
    dtype = p_pred.values.dtype

    normal_x, normal_y = make_boundary_normals(N, d0, d1, device, dtype)

    # x-face (N=(1,0)) lives on the last x index (axis -1 -> slice_dim=1).
    tx_pred = _traction_on_face(p_pred, normal_x, slice_dim=1, slice_idx=-1)
    # y-face (N=(0,1)) lives on the last y index (axis -2 -> slice_dim=0).
    ty_pred = _traction_on_face(p_pred, normal_y, slice_dim=0, slice_idx=-1)

    return tx_pred, ty_pred
