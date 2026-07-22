"""Tests for the calculus operators module."""

import unittest
from typing import Any

import torch

from tensopsciml.algebra.matrix_algebra import inverse
from tensopsciml.backends import get_backend
from tensopsciml.calculus.operators import (
    advection,
    autograd,
    curl,
    d1,
    d2,
    d3,
    divergence,
    gradient,
    laplacian,
    partial_derivative,
)
from tensopsciml.kernels.derivative import Derivative
from tensopsciml.tensors import MetrOps, TensOps, VolFormOps
from tensopsciml.utils.levi_civita_symbol import levi_civita_symbol

backend = get_backend()


def _diff_x(field: Any) -> Any:
    """Forward difference along the x-axis using the derivative stencil."""

    return 0.5 * (
        field[:, :, :-1, 1:] - field[:, :, :-1, :-1]
        + field[:, :, 1:, 1:] - field[:, :, 1:, :-1]
    )


def _diff_y(field: Any) -> Any:
    """Forward difference along the y-axis using the derivative stencil."""

    return 0.5 * (
        field[:, :, 1:, :-1] - field[:, :, :-1, :-1]
        + field[:, :, 1:, 1:] - field[:, :, :-1, 1:]
    )


def _euclidean_metric(
    n_samples: int,
    dimension: int,
    discretization: list[int],
    *,
    dtype=None,
) -> MetrOps:
    """Build the Euclidean metric g_ij = delta_ij on the given grid.

    Every metric-dependent operator takes its metric explicitly -- even the
    Cartesian case, which the library refuses to fabricate internally -- so the
    tests must build it. The grid matters: partial_derivative shrinks the domain
    by one node per spatial axis, so an operator that differentiates before
    using the metric needs it on the *smaller* grid.
    """
    dtype = dtype if dtype is not None else backend.float32
    identity = backend.eye(dimension, dtype=dtype)
    shape = [n_samples, dimension, dimension] + list(discretization)
    values = identity.reshape(
        1, dimension, dimension, *([1] * len(discretization))
    ).expand(*shape).clone()

    return MetrOps(
        values,
        domain_dim=len(discretization),
        codomain_dim=[dimension, dimension],
    )


class TestPartialDerivative(unittest.TestCase):
    """Test suite for the coordinate partial derivative.

    This is the operator formerly called ``gradient``: it differentiates a
    tensor field of any variance and appends a covariant derivative index. The
    name ``gradient`` now denotes the metric-raised gradient of a scalar (see
    :class:`TestGradient`).
    """

    def setUp(self) -> None:
        self.dtype = backend.float32

    def test_partial_derivative_scalar_field(self) -> None:
        """The partial derivative of a scalar field matches discrete differences."""

        values_data = [
            [[[float(row * 4 + col) for col in range(4)] for row in range(4)]]
        ]
        values = backend.tensor(values_data, dtype=self.dtype)
        tensor = TensOps(
            values,
            domain_dim=2,
            codomain_dim=[1],
            order=[0, 0],
            contravariant_indices=None,
            covariant_indices=None,
        )
        kernels = Derivative(1.0, 1.0, dtype=self.dtype).two_dimensions()

        result = partial_derivative(tensor, kernels)

        expected = backend.stack(
            [
                backend.squeeze(_diff_y(values), 1),
                backend.squeeze(_diff_x(values), 1),
            ],
            dim=1,
        )

        self.assertEqual(result.order, [0, 1])
        self.assertEqual(result.codomain, [2])
        self.assertIsNone(result.contravariant)
        self.assertEqual(result.covariant, [0])
        self.assertTrue(backend.allclose(result.values, expected))

    def test_partial_derivative_vector_field(self) -> None:
        """The partial derivative of a vector field appends a covariant index."""

        values_data = [
            [
                [[float(2 * row) for col in range(4)] for row in range(3)],
                [[float(3 * col) for col in range(4)] for row in range(3)],
            ]
        ]
        values = backend.tensor(values_data, dtype=self.dtype)
        tensor = TensOps(
            values,
            domain_dim=2,
            codomain_dim=[2],
            order=[1, 0],
            contravariant_indices=[0],
            covariant_indices=None,
        )
        kernels = Derivative(1.0, 1.0, dtype=self.dtype).two_dimensions()

        result = partial_derivative(tensor, kernels)

        expected = backend.stack([_diff_y(values), _diff_x(values)], dim=2)

        self.assertEqual(result.order, [1, 1])
        self.assertEqual(result.codomain, [2, 2])
        self.assertEqual(result.contravariant, [0])
        self.assertEqual(result.covariant, [1])
        self.assertTrue(backend.allclose(result.values, expected))

    def test_partial_derivative_scalar_field_1d(self) -> None:
        """A 1D scalar field differentiates into a covector with a size-1 axis."""

        values = torch.arange(6, dtype=self.dtype).view(1, 1, 6)
        tensor = TensOps(values, 1, [1], [0, 0], None, None)
        kernels = Derivative(1.0, dtype=self.dtype).one_dimension()

        result = partial_derivative(tensor, kernels)

        self.assertEqual(result.codomain, [1])
        self.assertEqual(result.order, [0, 1])
        self.assertIsNone(result.contravariant)
        self.assertEqual(result.covariant, [0])
        self.assertEqual(list(result.values.shape), [1, 1, 5])
        self.assertTrue(
            backend.allclose(result.values, torch.ones_like(result.values))
        )

    def test_partial_derivative_vector_field_1d(self) -> None:
        """A 1D vector field keeps its index and gains the derivative one."""

        component_0 = 2.0 * torch.arange(4, dtype=self.dtype)
        component_1 = 3.0 * torch.arange(4, dtype=self.dtype)
        values = torch.stack([component_0, component_1]).view(1, 2, 4)
        tensor = TensOps(values, 1, [2], [1, 0], [0], None)
        kernels = Derivative(1.0, dtype=self.dtype).one_dimension()

        result = partial_derivative(tensor, kernels)

        self.assertEqual(result.codomain, [2, 1])
        self.assertEqual(result.order, [1, 1])
        self.assertEqual(result.contravariant, [0])
        self.assertEqual(result.covariant, [1])

        expected = torch.tensor(
            [[[[2.0, 2.0, 2.0]], [[3.0, 3.0, 3.0]]]], dtype=self.dtype
        )
        self.assertTrue(backend.allclose(result.values, expected))

    def test_partial_derivative_kernel_mismatch_raises(self) -> None:
        """The kernel bank must match the field's domain dimension."""

        values = backend.zeros([1, 1, 3, 3, 3], dtype=self.dtype)
        tensor = TensOps(values, 3, [1], [0, 0], None, None)
        kernels = Derivative(1.0, 1.0, dtype=self.dtype).two_dimensions()

        with self.assertRaisesRegex(ValueError, "number of derivative kernels"):
            partial_derivative(tensor, kernels)


class TestGradient(unittest.TestCase):
    """Test suite for the metric-raised gradient of a scalar field."""

    def setUp(self) -> None:
        self.dtype = backend.float32

    def test_gradient_raises_the_derivative_index(self) -> None:
        """grad f = g^{ij} d_j f is contravariant, unlike the partial derivative."""

        values = torch.arange(6, dtype=self.dtype).view(1, 1, 6)
        scalar = TensOps(values, 1, [1], [0, 0], None, None)
        kernels = Derivative(1.0, dtype=self.dtype).one_dimension()

        # The metric raises the index of d_j f, which lives on the
        # differentiated (one node smaller) grid.
        inv_metric = inverse(_euclidean_metric(1, 1, [5], dtype=self.dtype))

        result = gradient(scalar, inv_metric, kernels)

        self.assertEqual(result.order, [1, 0])
        self.assertEqual(result.contravariant, [0])
        self.assertIsNone(result.covariant)
        # With g = I the components coincide with the partial derivative.
        self.assertTrue(
            backend.allclose(result.values, torch.ones_like(result.values))
        )

    def test_gradient_rejects_non_scalar_field(self) -> None:
        """gradient() is only defined for scalars; tensors use partial_derivative."""

        values = torch.ones(1, 2, 5, dtype=self.dtype)
        vector = TensOps(values, 1, [2], [1, 0], [0], None)
        kernels = Derivative(1.0, dtype=self.dtype).one_dimension()
        inv_metric = inverse(_euclidean_metric(1, 2, [4], dtype=self.dtype))

        with self.assertRaisesRegex(ValueError, "only defined for a scalar field"):
            gradient(vector, inv_metric, kernels)


class TestDivergence(unittest.TestCase):
    """Test suite for the Voss-Weyl divergence of a contravariant vector field."""

    def setUp(self) -> None:
        self.dtype = backend.float32

    def test_divergence_vector_field(self) -> None:
        """Divergence matches the sum of directional derivatives."""

        values_data = [
            [
                [[float(2 * row) for col in range(4)] for row in range(3)],
                [[float(3 * col) for col in range(4)] for row in range(3)],
            ]
        ]
        values = backend.tensor(values_data, dtype=self.dtype)
        tensor = TensOps(values, 2, [2], [1, 0], [0], None)
        kernels = Derivative(1.0, 1.0, dtype=self.dtype).two_dimensions()

        # The volume form scales u on u's own grid; divergence() restricts it
        # internally for the differentiated one.
        volume_form = VolFormOps(
            _euclidean_metric(1, 2, [3, 4], dtype=self.dtype))

        result = divergence(tensor, volume_form, kernels)

        expected = _diff_y(values[:, 0:1]) + _diff_x(values[:, 1:2])

        self.assertEqual(result.order, [0, 0])
        self.assertEqual(result.codomain, [1])
        self.assertIsNone(result.contravariant)
        self.assertIsNone(result.covariant)
        self.assertTrue(backend.allclose(result.values, expected, atol=1e-5))

    def test_divergence_non_vector_raises(self) -> None:
        """Divergence requires a contravariant vector field."""

        values = backend.ones([1, 1, 3, 3], dtype=self.dtype)
        scalar = TensOps(values, 2, [1], [0, 0], None, None)
        kernels = Derivative(1.0, 1.0, dtype=self.dtype).two_dimensions()
        volume_form = VolFormOps(
            _euclidean_metric(1, 2, [3, 3], dtype=self.dtype))

        with self.assertRaisesRegex(ValueError, "contravariant vector field"):
            divergence(scalar, volume_form, kernels)


class TestCurl(unittest.TestCase):
    """Test suite for the Levi-Civita curl."""

    def setUp(self) -> None:
        self.dtype = backend.float32

    def test_curl_of_a_gradient_vanishes_in_3d(self) -> None:
        """curl(grad f) = 0 for any scalar field."""

        scalar_values = torch.tensor(
            [[[[[float(depth * 9 + row * 3 + col) for col in range(3)]
                for row in range(3)]
               for depth in range(3)]]],
            dtype=self.dtype,
        )
        scalar = TensOps(scalar_values, 3, [1], [0, 0], None, None)
        kernels = Derivative(1.0, 1.0, 1.0, dtype=self.dtype).three_dimensions()

        # grad lives on the 2^3 grid (one node smaller than the 3^3 field).
        inv_metric = inverse(
            _euclidean_metric(1, 3, [2, 2, 2], dtype=self.dtype))
        vector_field = gradient(scalar, inv_metric, kernels)

        # curl lowers the vector on its own grid, then differentiates: the
        # symbol and volume form live on the once-more differentiated grid.
        metric = _euclidean_metric(1, 3, [2, 2, 2], dtype=self.dtype)
        symbol_values = levi_civita_symbol(1, 3, [1, 1, 1], dtype=self.dtype)
        lc_symbol = TensOps(
            symbol_values,
            domain_dim=3,
            codomain_dim=[3, 3, 3],
            order=[3, 0],
            contravariant_indices=[0, 1, 2],
            covariant_indices=None,
        )
        volume_form = VolFormOps(
            _euclidean_metric(1, 3, [1, 1, 1], dtype=self.dtype))

        result = curl(vector_field, metric, lc_symbol, volume_form, kernels)

        self.assertEqual(result.order, [1, 0])
        self.assertEqual(result.codomain, [3])
        self.assertTrue(backend.allclose(
            result.values, backend.zeros_like(result.values), atol=1e-5))

    def test_curl_2d_is_a_scalar(self) -> None:
        """The 2D curl contracts both Levi-Civita indices into a scalar."""

        torch.manual_seed(0)
        values = torch.rand(1, 2, 4, 4, dtype=self.dtype)
        tensor = TensOps(values, 2, [2], [1, 0], [0], None)
        kernels = Derivative(1.0, 1.0, dtype=self.dtype).two_dimensions()

        metric = _euclidean_metric(1, 2, [4, 4], dtype=self.dtype)
        symbol_values = levi_civita_symbol(1, 2, [3, 3], dtype=self.dtype)
        lc_symbol = TensOps(
            symbol_values,
            domain_dim=2,
            codomain_dim=[2, 2],
            order=[2, 0],
            contravariant_indices=[0, 1],
            covariant_indices=None,
        )
        volume_form = VolFormOps(
            _euclidean_metric(1, 2, [3, 3], dtype=self.dtype))

        result = curl(tensor, metric, lc_symbol, volume_form, kernels)

        # 2D curl = d(A_y)/dx - d(A_x)/dy, with the Euclidean metric.
        Ax = TensOps(values[:, 0:1], 2, [1], [0, 0], None, None)
        Ay = TensOps(values[:, 1:2], 2, [1], [0, 0], None, None)
        expected = d1(Ay, kernels).values - d2(Ax, kernels).values

        self.assertEqual(result.order, [0, 0])
        self.assertEqual(result.codomain, [1])
        self.assertTrue(backend.allclose(
            result.values, expected, atol=1e-5))

    def test_curl_requires_a_contravariant_vector(self) -> None:
        """Curl rejects anything that is not a vector field."""

        values = torch.rand(1, 3, 3, 4, 4, 4, dtype=self.dtype)
        tensor = TensOps(values, 3, [3, 3], [2, 0], [0, 1], None)
        kernels = Derivative(1.0, 1.0, 1.0, dtype=self.dtype).three_dimensions()
        metric = _euclidean_metric(1, 3, [4, 4, 4], dtype=self.dtype)
        volume_form = VolFormOps(
            _euclidean_metric(1, 3, [3, 3, 3], dtype=self.dtype))

        with self.assertRaisesRegex(ValueError, "contravariant vector field"):
            curl(tensor, metric, tensor, volume_form, kernels)

    def test_curl_requires_components_to_match_the_domain(self) -> None:
        """A 3-component vector over a 2D domain has no well-defined curl."""

        values = torch.rand(1, 3, 4, 4, dtype=self.dtype)
        tensor = TensOps(values, 2, [3], [1, 0], [0], None)
        kernels = Derivative(1.0, 1.0, dtype=self.dtype).two_dimensions()
        metric = _euclidean_metric(1, 3, [4, 4], dtype=self.dtype)
        volume_form = VolFormOps(
            _euclidean_metric(1, 2, [3, 3], dtype=self.dtype))

        with self.assertRaisesRegex(ValueError, "match the spatial domain dimension"):
            curl(tensor, metric, tensor, volume_form, kernels)

    def test_curl_requires_at_least_two_components(self) -> None:
        """A single-component vector field has no curl."""

        values = torch.rand(1, 1, 4, dtype=self.dtype)
        tensor = TensOps(values, 1, [1], [1, 0], [0], None)
        kernels = Derivative(1.0, dtype=self.dtype).one_dimension()
        metric = _euclidean_metric(1, 1, [4], dtype=self.dtype)
        volume_form = VolFormOps(
            _euclidean_metric(1, 1, [3], dtype=self.dtype))

        with self.assertRaisesRegex(ValueError, "at least 2 components"):
            curl(tensor, metric, tensor, volume_form, kernels)


class TestLaplacian(unittest.TestCase):
    """Test suite for the Laplace-Beltrami operator."""

    def setUp(self) -> None:
        self.dtype = backend.float32

    def test_laplacian_of_a_quadratic_is_constant(self) -> None:
        """Delta(x^2) = 2 everywhere."""

        x = torch.arange(6, dtype=self.dtype)
        scalar = TensOps((x ** 2).view(1, 1, 6), 1, [1], [0, 0], None, None)
        kernels = Derivative(1.0, dtype=self.dtype).one_dimension()

        # Both live on the differentiated grid (6 -> 5 nodes).
        inv_metric = inverse(_euclidean_metric(1, 1, [5], dtype=self.dtype))
        volume_form = VolFormOps(
            _euclidean_metric(1, 1, [5], dtype=self.dtype))

        result = laplacian(scalar, kernels, inv_metric, volume_form)

        self.assertEqual(result.codomain, [1])
        self.assertEqual(result.order, [0, 0])
        self.assertTrue(result.is_scalar)
        self.assertTrue(backend.allclose(
            result.values, 2.0 * torch.ones_like(result.values), atol=1e-5))

    def test_laplacian_rejects_non_scalar_field(self) -> None:
        """The Laplace-Beltrami identity here is defined for scalar fields."""

        values = torch.ones(1, 2, 6, dtype=self.dtype)
        vector = TensOps(values, 1, [2], [1, 0], [0], None)
        kernels = Derivative(1.0, dtype=self.dtype).one_dimension()
        inv_metric = inverse(_euclidean_metric(1, 2, [5], dtype=self.dtype))
        volume_form = VolFormOps(
            _euclidean_metric(1, 2, [5], dtype=self.dtype))

        with self.assertRaisesRegex(ValueError, "requires a scalar field"):
            laplacian(vector, kernels, inv_metric, volume_form)


class TestAdvection(unittest.TestCase):
    """Test suite for the advection term u . grad f."""

    def setUp(self) -> None:
        self.dtype = backend.float32

    def test_advection_contracts_u_with_the_derivative(self) -> None:
        """u . grad f contracts u against the partial derivative's new axis."""

        # f = x^2 on a unit grid: d_x f = 1, 3, 5 at the midpoints.
        scalar_values = torch.tensor([[[0.0, 1.0, 4.0, 9.0]]], dtype=self.dtype)
        # A constant advecting field u = 2 makes the expected value 2 * d_x f.
        u_values = torch.full((1, 1, 4), 2.0, dtype=self.dtype)

        u_tensor = TensOps(u_values, 1, [1], [1, 0], [0], None)
        scalar = TensOps(scalar_values, 1, [1], [0, 0], None, None)
        kernels = Derivative(1.0, dtype=self.dtype).one_dimension()

        result = advection(u_tensor, scalar, kernels)

        expected = torch.tensor([[[2.0, 6.0, 10.0]]], dtype=self.dtype)

        self.assertEqual(result.order, [0, 0])
        self.assertTrue(backend.allclose(result.values, expected, atol=1e-5))

    def test_advection_requires_first_order_u(self) -> None:
        """The advecting field must be a vector."""

        scalar = TensOps(torch.ones(1, 1, 4, dtype=self.dtype),
                         1, [1], [0, 0], None, None)
        kernels = Derivative(1.0, dtype=self.dtype).one_dimension()

        with self.assertRaisesRegex(ValueError, "advecting field u to be first order"):
            advection(scalar, scalar, kernels)

    def test_advection_requires_scalar_f(self) -> None:
        """The advected field must be a scalar."""

        u_tensor = TensOps(torch.ones(1, 1, 4, dtype=self.dtype),
                           1, [1], [1, 0], [0], None)
        kernels = Derivative(1.0, dtype=self.dtype).one_dimension()

        with self.assertRaisesRegex(ValueError, "advected field f to be a scalar"):
            advection(u_tensor, u_tensor, kernels)


class TestDirectionalDerivatives(unittest.TestCase):
    """Test suite for the per-axis derivative helpers d1/d2/d3."""

    def setUp(self) -> None:
        self.dtype = backend.float32

    def test_d1_matches_forward_difference(self) -> None:
        """The d1 helper should match a direct forward difference."""

        values = torch.arange(6, dtype=self.dtype).view(1, 1, 6)
        tensor = TensOps(values, 1, [1], [0, 0], None, None)
        kernels = Derivative(1.0, dtype=self.dtype).one_dimension()

        result = d1(tensor, kernels)
        expected = values[:, :, 1:] - values[:, :, :-1]

        self.assertEqual(result.values.shape, (1, 1, 5))
        self.assertTrue(backend.allclose(result.values, expected))

    def test_d2_and_d3_validate_domains(self) -> None:
        """Higher-axis derivatives should reject insufficient domains."""

        line = torch.ones(1, 1, 6, dtype=self.dtype)
        tensor_1d = TensOps(line, 1, [1], [0, 0], None, None)
        kernels_2d = Derivative(1.0, 1.0, dtype=self.dtype).two_dimensions()
        kernels_3d = Derivative(
            1.0, 1.0, 1.0, dtype=self.dtype).three_dimensions()

        with self.assertRaisesRegex(ValueError, "number of derivative kernels"):
            d2(tensor_1d, kernels_2d)
        with self.assertRaisesRegex(ValueError, "number of derivative kernels"):
            d3(tensor_1d, kernels_3d)

    def test_d1_d2_d3_require_a_scalar_field(self) -> None:
        """The per-axis helpers only differentiate scalar fields."""

        vector_1d = TensOps(torch.ones(1, 2, 6, dtype=self.dtype),
                            1, [2], [1, 0], [0], None)
        vector_2d = TensOps(torch.ones(1, 2, 6, 6, dtype=self.dtype),
                            2, [2], [1, 0], [0], None)
        vector_3d = TensOps(torch.ones(1, 3, 4, 4, 4, dtype=self.dtype),
                            3, [3], [1, 0], [0], None)

        kernels_1d = Derivative(1.0, dtype=self.dtype).one_dimension()
        kernels_2d = Derivative(1.0, 1.0, dtype=self.dtype).two_dimensions()
        kernels_3d = Derivative(
            1.0, 1.0, 1.0, dtype=self.dtype).three_dimensions()

        with self.assertRaisesRegex(ValueError, r"d1\(\) requires a scalar"):
            d1(vector_1d, kernels_1d)
        with self.assertRaisesRegex(ValueError, r"d2\(\) requires a scalar"):
            d2(vector_2d, kernels_2d)
        with self.assertRaisesRegex(ValueError, r"d3\(\) requires a scalar"):
            d3(vector_3d, kernels_3d)

    def test_d2_rejects_a_one_dimensional_domain(self) -> None:
        """There is no second axis to differentiate along in 1D."""

        # A 1D scalar field whose domain is 2 kernels' worth: the kernel bank
        # matches, so the guard under test is the one that fires.
        scalar_2d = TensOps(torch.ones(1, 1, 6, 6, dtype=self.dtype),
                            2, [1], [0, 0], None, None)
        object.__setattr__(scalar_2d, "_TensOps__domain", 1)
        kernels = Derivative(1.0, dtype=self.dtype).one_dimension()

        with self.assertRaisesRegex(ValueError, "second dimension for a 1D"):
            d2(scalar_2d, kernels)

    def test_d3_rejects_domains_below_three(self) -> None:
        """There is no third axis to differentiate along in 1D or 2D."""

        scalar_3d = TensOps(torch.ones(1, 1, 4, 4, 4, dtype=self.dtype),
                            3, [1], [0, 0], None, None)
        object.__setattr__(scalar_3d, "_TensOps__domain", 2)
        kernels = Derivative(1.0, 1.0, dtype=self.dtype).two_dimensions()

        with self.assertRaisesRegex(ValueError, "third dimension for a 2D"):
            d3(scalar_3d, kernels)

    def test_d2_and_d3_compute_forward_difference(self) -> None:
        """d2 and d3 should match the backend convolutions they wrap."""

        grid_2d = torch.arange(16, dtype=self.dtype).view(1, 1, 4, 4)
        tensor_2d = TensOps(grid_2d, 2, [1], [0, 0], None, None)
        kernels_2d = Derivative(1.0, 1.0, dtype=self.dtype).two_dimensions()
        result_d2 = d2(tensor_2d, kernels_2d)
        expected_d2 = backend.conv2d(
            grid_2d, kernels_2d[1], stride=1, padding="valid")

        grid_3d = torch.arange(27, dtype=self.dtype).view(1, 1, 3, 3, 3)
        tensor_3d = TensOps(grid_3d, 3, [1], [0, 0], None, None)
        kernels_3d = Derivative(
            1.0, 1.0, 1.0, dtype=self.dtype).three_dimensions()
        result_d3 = d3(tensor_3d, kernels_3d)
        expected_d3 = backend.conv3d(
            grid_3d, kernels_3d[2], stride=1, padding="valid")

        self.assertTrue(backend.allclose(result_d2.values, expected_d2))
        self.assertTrue(backend.allclose(result_d3.values, expected_d3))


class TestAutograd(unittest.TestCase):
    """Test suite for the autograd-based derivative."""

    def setUp(self) -> None:
        self.dtype = backend.float32

    def test_autograd_matches_manual_gradient(self) -> None:
        """Autograd dA/dB should match the analytical derivative."""

        axis = torch.linspace(0, 4, steps=5, dtype=self.dtype).view(1, 1, 5)
        axis.requires_grad_(True)
        tensor_b = TensOps(axis, 1, [1], [0, 0], None, None)
        tensor_a = TensOps(tensor_b.values ** 2, 1, [1], [0, 0], None, None)

        result = autograd(tensor_a, tensor_b,
                          retain_graph=False, allow_unused=False)
        expected = 2 * tensor_b.values

        self.assertEqual(result.codomain, [1])
        self.assertEqual(result.order, [0, 0])
        self.assertTrue(backend.allclose(result.values, expected))

    def test_autograd_vector_A_gives_full_jacobian(self) -> None:
        """Non-scalar A must get a true per-component Jacobian, not a VJP sum."""

        t = torch.linspace(0, 4, steps=5, dtype=self.dtype).view(1, 1, 5)
        t.requires_grad_(True)
        tensor_b = TensOps(t, 1, [1], [0, 0], None, None)
        a_values = torch.cat(
            [tensor_b.values, tensor_b.values ** 2, tensor_b.values ** 3], dim=1)
        tensor_a = TensOps(a_values, 1, [3], [1, 0], [0], None)

        result = autograd(tensor_a, tensor_b,
                          retain_graph=True, allow_unused=False)

        tt = tensor_b.values
        expected = torch.stack(
            [torch.ones_like(tt[:, 0]), 2 * tt[:, 0], 3 * tt[:, 0] ** 2], dim=1)

        self.assertEqual(result.order, [1, 0])
        self.assertEqual(result.codomain, [3])
        self.assertEqual(result.contravariant, [0])
        self.assertIsNone(result.covariant)
        self.assertTrue(backend.allclose(result.values, expected))

    def test_autograd_matrix_A_and_B_gives_full_jacobian(self) -> None:
        """Both A and B non-scalar must yield the full rank-4 Jacobian."""

        c = torch.rand(2, 2, 2, 6, dtype=self.dtype)
        c.requires_grad_(True)
        tensor_b = TensOps(c, 1, [2, 2], [2, 0], [0, 1], None)
        tensor_a = TensOps(tensor_b.values ** 2, 1,
                           [2, 2], [2, 0], [0, 1], None)

        result = autograd(tensor_a, tensor_b,
                          retain_graph=True, allow_unused=True)

        cc = tensor_b.values
        expected = torch.zeros(2, 2, 2, 2, 2, 6, dtype=self.dtype)
        for i in range(2):
            for j in range(2):
                expected[:, i, j, i, j] = 2 * cc[:, i, j]

        self.assertEqual(result.order, [2, 2])
        self.assertEqual(result.codomain, [2, 2, 2, 2])
        self.assertEqual(result.contravariant, [0, 1])
        self.assertEqual(result.covariant, [2, 3])
        self.assertTrue(backend.allclose(result.values, expected, atol=1e-5))

    def test_autograd_scalar_T_against_vector_S(self) -> None:
        """A scalar T drops its degenerate channel, keeping only S's indices.

        dT/dS where T is a scalar and S a vector: the result carries S's index
        alone, so T's size-1 codomain axis must be squeezed away.
        """
        s_values = torch.rand(1, 2, 5, dtype=self.dtype, requires_grad=True)
        tensor_s = TensOps(s_values, 1, [2], [1, 0], [0], None)

        # T = sum of S's squared components: a genuine scalar field.
        t_values = (tensor_s.values ** 2).sum(dim=1, keepdim=True)
        tensor_t = TensOps(t_values, 1, [1], [0, 0], None, None)

        result = autograd(tensor_t, tensor_s, retain_graph=True)

        self.assertEqual(result.codomain, [2])
        self.assertEqual(list(result.values.shape), [1, 2, 5])
        self.assertTrue(backend.allclose(
            result.values, 2 * tensor_s.values, atol=1e-5))

    def test_autograd_vector_T_against_vector_S_1d(self) -> None:
        """Differentiating a 1D vector by a 1D vector keeps both indices.

        Regression: a genuine vector/covector in a 1D domain has codomain [1],
        but its single index is real and must NOT be squeezed as if it were a
        scalar. dT^i/dt^j must come out as an order [1, 1] tensor, not collapse
        to codomain [1] and fail tensorial-order validation.
        """
        t = torch.linspace(1, 5, steps=5, dtype=self.dtype).view(1, 1, 5)
        t.requires_grad_(True)
        tensor_s = TensOps(t, 1, [1], [1, 0], [0], None)
        tensor_t = TensOps(tensor_s.values ** 2, 1, [1], [1, 0], [0], None)

        result = autograd(tensor_t, tensor_s, retain_graph=True)

        self.assertEqual(result.order, [1, 1])
        self.assertEqual(result.codomain, [1, 1])
        self.assertEqual(result.contravariant, [0])
        self.assertEqual(result.covariant, [1])
        self.assertTrue(backend.allclose(
            result.values.squeeze(), (2 * t).squeeze(), atol=1e-5))

    def test_autograd_domain_mismatch_raises(self) -> None:
        """Autograd should reject tensors with different domains."""

        values = torch.ones(1, 1, 4, dtype=self.dtype)
        tensor_a = TensOps(values, 1, [1], [0, 0], None, None)
        tensor_b = TensOps(values.unsqueeze(-1), 2, [1], [0, 0], None, None)

        with self.assertRaisesRegex(ValueError, "same domain dimension"):
            autograd(tensor_a, tensor_b)

    def test_autograd_disconnected_b_raises_explicit_error(self) -> None:
        """B not on A's graph should raise a clear error, not a raw None-assign crash."""

        c = torch.linspace(0, 4, steps=5, dtype=self.dtype).view(1, 1, 5)
        c.requires_grad_(True)
        tensor_a = TensOps(c * 2, 1, [1], [0, 0], None, None)
        tensor_b = TensOps(
            torch.linspace(0, 4, steps=5, dtype=self.dtype).view(1, 1, 5),
            1, [1], [0, 0], None, None,
        )

        with self.assertRaisesRegex(ValueError, "not connected to T's"):
            autograd(tensor_a, tensor_b)


if __name__ == "__main__":
    unittest.main()
