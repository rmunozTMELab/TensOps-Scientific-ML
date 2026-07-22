"""Reference boundary normal covectors for 2D traction computation.

Provides covariant unit normal covectors N_j for the two positive-face
boundaries of the 2D reference domain, ready for T^i = P^{ij} N_j contractions
with the first Piola-Kirchhoff stress P^{ij} (order [2, 0]) produced by this
model. This mirrors the psi_pgnniv reference, where covariant normals are
contracted against a fully contravariant stress; here the normal is the
reference (Lagrangian) normal.
"""

import torch
from tensopsciml.tensors import TensOps


def make_boundary_normals(N, Nx, Ny, device, dtype):
    """Build covariant reference unit normals for the two positive boundaries.

    Returns N_j as TensOps with order=[0, 1] (covariant) so they contract
    naturally with the reference (second) contravariant index of the first
    Piola-Kirchhoff stress P^{ij} [2, 0]:
        T^i = P^{ij} N_j

    Args:
        N: Batch size.
        Nx, Ny: Spatial grid dimensions.
        device: Torch device.
        dtype: Torch dtype.

    Returns:
        Tuple (normal_x, normal_y):
            - normal_x: N = (1, 0), x boundary (positive x face).
            - normal_y: N = (0, 1), y boundary (positive y face).
    """
    def _normal(component: int) -> TensOps:
        v = torch.zeros(N, 2, Nx, Ny, device=device, dtype=dtype)
        v[:, component] = 1.0
        return TensOps(v, domain_dim=2, codomain_dim=[2], order=[0, 1],
                       contravariant_indices=None, covariant_indices=[0])

    return _normal(0), _normal(1)
