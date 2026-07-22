# Copyright 2026:
#   Ruben Muñoz-Sierra (rmunoz@unizar.es),
#   Jacobo Ayensa-Jiménez (jacobo.ayensa@upm.es),
#   Manuel Doblaré (mdoblare@unizar.es).
#
# Licensed under the Apache License, Version 2.0
# See the LICENSE file for details.

"""TensOps: backend-agnostic tensor calculus on discretized fields."""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("tensopsciml")
except PackageNotFoundError:  # package not installed (e.g. running from a source checkout)
    __version__ = "0.0.0"

from .backends import (
    Backend,
    TorchBackend,
    get_backend,
    set_backend,
    use_backend,
)

from .tensors.tensor import TensOps
from .tensors.metrics import MetrOps, VolFormOps

from .kernels.average import Average
from .kernels.derivative import Derivative

from .algebra.invariants import (
    first_invariant,
    second_invariant,
    third_invariant,
)
from .algebra.matrix_algebra import (
    determinant,
    inverse,
    transpose,
    trace,
    spherical,
    deviatoric,
)
from .algebra.tensor_algebra import (
    tensor_product,
    contraction,
    self_contraction,
    symmetrize,
    antisymmetrize,
    symmetrization_covariant,
    symmetrization_contravariant,
    antisymmetrization_covariant,
    antisymmetrization_contravariant,
    raise_index,
    lower_index,
    cross_prod,
    kronecker_delta,
    levi_civita_tensor,
)
from .utils.levi_civita_symbol import levi_civita_symbol

from .calculus import (
    ConnectOps,
    curl,
    cov_curl,
    cov_derivative,
    cov_directional_derivative,
    cov_divergence,
    lie_derivative,
    d1,
    d2,
    d3,
    divergence,
    partial_derivative,
    gradient,
    laplacian,
    advection,
    autograd,
)

from .utils.averaging import m, m1, m2, m3

__all__ = [
    "__version__",
    "TensOps",
    "MetrOps",
    "VolFormOps",
    "Backend",
    "TorchBackend",
    "get_backend",
    "set_backend",
    "use_backend",
    "Average",
    "Derivative",
    "ConnectOps",
    "d1",
    "d2",
    "d3",
    "m",
    "m1",
    "m2",
    "m3",
    "tensor_product",
    "contraction",
    "self_contraction",
    "transpose",
    "trace",
    "symmetrize",
    "antisymmetrize",
    "symmetrization_covariant",
    "symmetrization_contravariant",
    "antisymmetrization_covariant",
    "antisymmetrization_contravariant",
    "first_invariant",
    "second_invariant",
    "third_invariant",
    "determinant",
    "inverse",
    "spherical",
    "deviatoric",
    "partial_derivative",
    "gradient",
    "divergence",
    "curl",
    "cov_curl",
    "cov_derivative",
    "cov_directional_derivative",
    "cov_divergence",
    "lie_derivative",
    "laplacian",
    "raise_index",
    "lower_index",
    "cross_prod",
    "kronecker_delta",
    "levi_civita_symbol",
    "levi_civita_tensor",
    "advection",
    "autograd",
]
