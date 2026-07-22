"""Pytest configuration and shared test fixtures for TensOps test suite.

This module provides common test data, backend setup, and reusable tensor
fixtures for all test modules. It generates synthetic scalar and vector fields
in 3D space with known mathematical properties (trigonometric functions) and
constructs higher-order tensor fields from these base fields.

Module-level fixtures:
    backend: The active compute backend (PyTorch).
    N_data: Number of data samples (10).
    d1, d2, d3: Grid spacing in each dimension (0.1).
    phi_: Raw scalar field tensor (sin(x) + cos(y)).
    psi_: Raw scalar field tensor (x).
    u_, v_, w_, x_: Raw vector field tensors with trigonometric expressions.
    A_, B_, C_: Raw second-order tensor fields (stacked vectors).
    phi, psi: TensOps scalar field objects.
    u, v, w, x: TensOps vector field objects.
    A, B, C: TensOps second-order tensor field objects.
"""

import os
import sys

SRC_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)

from data.data_generator import ExampleScalarFieldThreeDimensions, ExampleVectorialFieldThreeDimensions
from tensopsciml import TensOps
from tensopsciml.backends import get_backend

backend = get_backend()

N_data = 10
d1 = d2 = d3 = 0.1

phi_ = ExampleScalarFieldThreeDimensions(
    N_data,
    f_char='sin(x) + cos(y)',
    dx=d1, dy=d2, dz=d3
).func()

psi_ = ExampleScalarFieldThreeDimensions(
    N_data,
    f_char='x',
    dx=d1, dy=d2, dz=d3
).func()

u_ = ExampleVectorialFieldThreeDimensions(
    N_data,
    Fx_char='sin(x)**2 + cos(y)',
    Fy_char='sin(x)',
    Fz_char='sin(z)',
    dx=d1, dy=d2, dz=d3
).Func()

v_ = ExampleVectorialFieldThreeDimensions(
    N_data,
    Fx_char='sin(x**2) + cos(y**2)',
    Fy_char='cos(x*y)',
    Fz_char='cos(z)',
    dx=d1, dy=d2, dz=d3
).Func()

w_ = ExampleVectorialFieldThreeDimensions(
    N_data, 
    Fx_char='sin(x) * cos(y) + tan(z)', 
    Fy_char='sin(x**2 + y**2)', 
    Fz_char='cos(z) - sin(x*y)',
    dx=d1, dy=d2, dz=d3
).Func()

x_ = ExampleVectorialFieldThreeDimensions(
    N_data, 
    Fx_char='sin(x + y) * cos(z)', 
    Fy_char='sin(x) - cos(y)',
    Fz_char='tan(z) + sin(x + y)',
    dx=d1, dy=d2, dz=d3
).Func()

A_ = backend.stack((u_, v_, w_), dim=2)
B_ = backend.stack((w_, v_, x_), dim=2)
C_ = backend.stack((w_, x_, x_), dim=2)

phi = TensOps(phi_, domain_dim=3, codomain_dim=[1], order=[0, 0], contravariant_indices=None, covariant_indices=None)
psi = TensOps(psi_, domain_dim=3, codomain_dim=[1], order=[0, 0], contravariant_indices=None, covariant_indices=None)

u = TensOps(u_, domain_dim=3, codomain_dim=[3], order=[1, 0], contravariant_indices=[0], covariant_indices=None)
v = TensOps(v_, domain_dim=3, codomain_dim=[3], order=[1, 0], contravariant_indices=[0], covariant_indices=None)
w = TensOps(w_, domain_dim=3, codomain_dim=[3], order=[1, 0], contravariant_indices=[0], covariant_indices=None)
x = TensOps(x_, domain_dim=3, codomain_dim=[3], order=[1, 0], contravariant_indices=[0], covariant_indices=None)

A = TensOps(A_, domain_dim=3, codomain_dim=[3, 3], order=[1, 1], contravariant_indices=[0], covariant_indices=[1])
B = TensOps(B_, domain_dim=3, codomain_dim=[3, 3], order=[1, 1], contravariant_indices=[0], covariant_indices=[1])
C = TensOps(C_, domain_dim=3, codomain_dim=[3, 3], order=[1, 1], contravariant_indices=[0], covariant_indices=[1])