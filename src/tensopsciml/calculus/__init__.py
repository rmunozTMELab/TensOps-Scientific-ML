"""Calculus subpackage: differential operators on tensor fields."""

from .connections import ConnectOps
from .cov_operators import (
    cov_curl,
    cov_derivative,
    cov_directional_derivative,
    cov_divergence,
    lie_derivative,
)
from .operators import (
    d1,
    d2,
    d3,
    partial_derivative,
    gradient,
    curl,
    divergence,
    laplacian,
    autograd,
    advection,
)

__all__ = [
    "ConnectOps",
    "d1",
    "d2",
    "d3",
    "partial_derivative",
    "gradient",
    "curl",
    "cov_curl",
    "cov_derivative",
    "cov_directional_derivative",
    "cov_divergence",
    "lie_derivative",
    "divergence",
    "laplacian",
    "autograd",
    "advection",
]
