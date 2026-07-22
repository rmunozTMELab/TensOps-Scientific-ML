"""Tests for the covariant (connection-aware) differential operators.

The strategy throughout is to pin each operator against a property that holds
independently of the implementation:

* On a flat (constant) metric the Christoffel symbols vanish, so every
  covariant operator must collapse onto its plain coordinate counterpart
  (``cov_derivative`` -> ``partial_derivative``, and so on). That catches sign
  and index-placement errors in the Gamma corrections without hand-computing
  them.
* ``cov_curl`` of an already-lowered vector must reproduce the flat ``curl``
  bit-for-bit, which its docstring claims explicitly.
* The Lie derivative of a field along itself is the Lie bracket [X, X] = 0,
  which must hold exactly regardless of the metric.
"""

import unittest

import torch

from tensopsciml.algebra.tensor_algebra import lower_index, self_contraction
from tensopsciml.backends import get_backend
from tensopsciml.calculus.connections import ConnectOps
from tensopsciml.calculus.cov_operators import (
    cov_curl,
    cov_derivative,
    cov_directional_derivative,
    cov_divergence,
    lie_derivative,
)
from tensopsciml.calculus.operators import (
    advection,
    curl,
    partial_derivative,
)
from tensopsciml.kernels.derivative import Derivative
from tensopsciml.tensors import MetrOps, TensOps, VolFormOps
from tensopsciml.utils.levi_civita_symbol import levi_civita_symbol

backend = get_backend()
DTYPE = backend.float32


def _euclidean_metric(n_samples: int, dimension: int, discretization: list[int]) -> MetrOps:
    """Build the Euclidean metric on the given grid."""
    identity = backend.eye(dimension, dtype=DTYPE)
    shape = [n_samples, dimension, dimension] + list(discretization)
    values = identity.reshape(
        1, dimension, dimension, *([1] * len(discretization))
    ).expand(*shape).clone()

    return MetrOps(
        values,
        domain_dim=len(discretization),
        codomain_dim=[dimension, dimension],
    )


class TestCovDerivative(unittest.TestCase):
    """Test suite for the covariant derivative."""

    def setUp(self) -> None:
        torch.manual_seed(0)
        self.kernels = Derivative(1.0, 1.0, dtype=DTYPE).two_dimensions()
        self.metric = _euclidean_metric(1, 2, [5, 5])
        self.connection = ConnectOps.levi_civita(self.metric, self.kernels)

    def _vector(self) -> TensOps:
        return TensOps(
            torch.rand(1, 2, 5, 5, dtype=DTYPE),
            domain_dim=2,
            codomain_dim=[2],
            order=[1, 0],
            contravariant_indices=[0],
            covariant_indices=None,
        )

    def test_flat_connection_reduces_to_the_partial_derivative(self) -> None:
        """With Gamma = 0 the covariant derivative is the coordinate one."""

        vector = self._vector()

        result = cov_derivative(vector, self.connection, self.kernels)
        expected = partial_derivative(vector, self.kernels)

        self.assertEqual(result.order, expected.order)
        self.assertEqual(result.codomain, expected.codomain)
        self.assertEqual(result.covariant, expected.covariant)
        self.assertTrue(backend.allclose(
            result.values, expected.values, atol=1e-6))

    def test_appends_a_covariant_derivative_index(self) -> None:
        """The covariant derivative of a vector is a mixed (1, 1) tensor."""

        result = cov_derivative(self._vector(), self.connection, self.kernels)

        self.assertEqual(result.order, [1, 1])
        self.assertEqual(result.codomain, [2, 2])
        self.assertEqual(result.contravariant, [0])
        self.assertEqual(result.covariant, [1])

    def test_covariant_field_also_reduces_to_the_partial_derivative(self) -> None:
        """The lower-index correction term is exercised by a covector too."""

        covector = TensOps(
            torch.rand(1, 2, 5, 5, dtype=DTYPE),
            domain_dim=2,
            codomain_dim=[2],
            order=[0, 1],
            contravariant_indices=None,
            covariant_indices=[0],
        )

        result = cov_derivative(covector, self.connection, self.kernels)
        expected = partial_derivative(covector, self.kernels)

        self.assertEqual(result.order, [0, 2])
        self.assertTrue(backend.allclose(
            result.values, expected.values, atol=1e-6))

    def test_scalar_field_has_no_christoffel_correction(self) -> None:
        """A scalar has no indices, so nabla f is just d_i f for any connection."""

        scalar = TensOps(
            torch.rand(1, 1, 5, 5, dtype=DTYPE),
            domain_dim=2,
            codomain_dim=[1],
            order=[0, 0],
            contravariant_indices=None,
            covariant_indices=None,
        )

        result = cov_derivative(scalar, self.connection, self.kernels)
        expected = partial_derivative(scalar, self.kernels)

        self.assertTrue(backend.allclose(
            result.values, expected.values, atol=1e-6))

    def test_codomain_must_match_the_connection(self) -> None:
        """A field whose indices don't range over the connection's dimension."""

        vector_3d = TensOps(
            torch.rand(1, 3, 5, 5, dtype=DTYPE),
            domain_dim=2,
            codomain_dim=[3],
            order=[1, 0],
            contravariant_indices=[0],
            covariant_indices=None,
        )

        with self.assertRaisesRegex(ValueError, "codomain dimensions must match"):
            cov_derivative(vector_3d, self.connection, self.kernels)

    def test_rejects_a_non_connection(self) -> None:
        """The connection argument must be a ConnectOps."""

        with self.assertRaisesRegex(ValueError, "Expected ConnectOps"):
            cov_derivative(self._vector(), self.metric, self.kernels)


class TestCovDivergence(unittest.TestCase):
    """Test suite for the covariant divergence."""

    def setUp(self) -> None:
        torch.manual_seed(0)
        self.kernels = Derivative(1.0, 1.0, dtype=DTYPE).two_dimensions()
        self.metric = _euclidean_metric(1, 2, [5, 5])
        self.connection = ConnectOps.levi_civita(self.metric, self.kernels)

    def test_divergence_of_a_vector_is_a_scalar(self) -> None:
        """nabla_i v^i contracts the derivative index into a scalar field."""

        vector = TensOps(
            torch.rand(1, 2, 5, 5, dtype=DTYPE),
            domain_dim=2,
            codomain_dim=[2],
            order=[1, 0],
            contravariant_indices=[0],
            covariant_indices=None,
        )

        result = cov_divergence(vector, self.connection, self.kernels)

        # With Gamma = 0 this is the trace of the coordinate derivative.
        expected = self_contraction(
            partial_derivative(vector, self.kernels), [0], [0])

        self.assertEqual(result.order, [0, 0])
        self.assertTrue(result.is_scalar)
        self.assertTrue(backend.allclose(
            result.values, expected.values, atol=1e-6))

    def test_index_selects_the_contracted_slot(self) -> None:
        """On a non-symmetric tensor, different indices give different divergences."""

        values = torch.rand(1, 2, 2, 5, 5, dtype=DTYPE)
        tensor = TensOps(
            values,
            domain_dim=2,
            codomain_dim=[2, 2],
            order=[2, 0],
            contravariant_indices=[0, 1],
            covariant_indices=None,
        )

        div_0 = cov_divergence(tensor, self.connection, self.kernels, index=0)
        div_1 = cov_divergence(tensor, self.connection, self.kernels, index=1)

        self.assertEqual(div_0.order, [1, 0])
        self.assertEqual(div_1.order, [1, 0])
        self.assertFalse(backend.allclose(div_0.values, div_1.values))


class TestCovCurl(unittest.TestCase):
    """Test suite for the covariant curl."""

    def setUp(self) -> None:
        torch.manual_seed(0)
        self.kernels = Derivative(1.0, 1.0, dtype=DTYPE).two_dimensions()
        self.metric = _euclidean_metric(1, 2, [5, 5])
        self.connection = ConnectOps.levi_civita(self.metric, self.kernels)

        # E and the volume form live on the differentiated grid (5 -> 4).
        symbol_values = levi_civita_symbol(1, 2, [4, 4], dtype=DTYPE)
        self.lc_symbol = TensOps(
            symbol_values,
            domain_dim=2,
            codomain_dim=[2, 2],
            order=[2, 0],
            contravariant_indices=[0, 1],
            covariant_indices=None,
        )
        self.volume_form = VolFormOps(_euclidean_metric(1, 2, [4, 4]))

        self.vector = TensOps(
            torch.rand(1, 2, 5, 5, dtype=DTYPE),
            domain_dim=2,
            codomain_dim=[2],
            order=[1, 0],
            contravariant_indices=[0],
            covariant_indices=None,
        )

    def test_matches_the_flat_curl_on_a_lowered_vector(self) -> None:
        """cov_curl(lower_index(g, u), ...) reproduces curl(u, g, ...) exactly.

        This is the equivalence the docstring promises: on the plain-vector case
        the two operators agree to floating-point round-off, not merely in the
        fine-grid limit.
        """
        lowered = lower_index(self.metric, self.vector)

        result = cov_curl(
            lowered, self.connection, self.lc_symbol,
            self.volume_form, self.kernels, index=0,
        )
        expected = curl(
            self.vector, self.metric, self.lc_symbol,
            self.volume_form, self.kernels,
        )

        self.assertEqual(result.order, expected.order)
        self.assertTrue(backend.allclose(
            result.values, expected.values, atol=1e-6))

    def test_three_dimensional_curl_leaves_a_free_contravariant_index(self) -> None:
        """In 3D only two of E's indices contract, so the curl is a vector."""

        kernels_3d = Derivative(
            1.0, 1.0, 1.0, dtype=DTYPE).three_dimensions()
        metric_3d = _euclidean_metric(1, 3, [4, 4, 4])
        connection_3d = ConnectOps.levi_civita(metric_3d, kernels_3d)

        vector_3d = TensOps(
            torch.rand(1, 3, 4, 4, 4, dtype=DTYPE),
            domain_dim=3,
            codomain_dim=[3],
            order=[1, 0],
            contravariant_indices=[0],
            covariant_indices=None,
        )
        lowered = lower_index(metric_3d, vector_3d)

        symbol_values = levi_civita_symbol(1, 3, [3, 3, 3], dtype=DTYPE)
        lc_symbol_3d = TensOps(
            symbol_values,
            domain_dim=3,
            codomain_dim=[3, 3, 3],
            order=[3, 0],
            contravariant_indices=[0, 1, 2],
            covariant_indices=None,
        )
        volume_form_3d = VolFormOps(_euclidean_metric(1, 3, [3, 3, 3]))

        result = cov_curl(
            lowered, connection_3d, lc_symbol_3d,
            volume_form_3d, kernels_3d, index=0,
        )
        expected = curl(
            vector_3d, metric_3d, lc_symbol_3d,
            volume_form_3d, kernels_3d,
        )

        self.assertEqual(result.order, [1, 0])
        self.assertEqual(result.codomain, [3])
        self.assertTrue(backend.allclose(
            result.values, expected.values, atol=1e-6))

    def test_requires_a_valid_covariant_slot(self) -> None:
        """'index' must select an existing covariant index of T."""

        lowered = lower_index(self.metric, self.vector)

        with self.assertRaisesRegex(ValueError, "valid covariant slot"):
            cov_curl(
                lowered, self.connection, self.lc_symbol,
                self.volume_form, self.kernels, index=3,
            )

    def test_rejects_a_tensor_without_covariant_indices(self) -> None:
        """A purely contravariant field has no slot to curl over."""

        with self.assertRaisesRegex(ValueError, "valid covariant slot"):
            cov_curl(
                self.vector, self.connection, self.lc_symbol,
                self.volume_form, self.kernels, index=0,
            )

    def test_curled_index_must_span_the_domain(self) -> None:
        """The curled slot must range over the domain dimension."""

        covector_3 = TensOps(
            torch.rand(1, 3, 5, 5, dtype=DTYPE),
            domain_dim=2,
            codomain_dim=[3],
            order=[0, 1],
            contravariant_indices=None,
            covariant_indices=[0],
        )

        with self.assertRaisesRegex(ValueError, "range over the domain dimension"):
            cov_curl(
                covector_3, self.connection, self.lc_symbol,
                self.volume_form, self.kernels, index=0,
            )

    def test_rejects_a_one_dimensional_domain(self) -> None:
        """Only 2D and 3D domains have a curl."""

        kernels_1d = Derivative(1.0, dtype=DTYPE).one_dimension()
        metric_1d = _euclidean_metric(1, 1, [5])
        connection_1d = ConnectOps.levi_civita(metric_1d, kernels_1d)
        covector_1d = TensOps(
            torch.rand(1, 1, 5, dtype=DTYPE),
            domain_dim=1,
            codomain_dim=[1],
            order=[0, 1],
            contravariant_indices=None,
            covariant_indices=[0],
        )
        volume_form_1d = VolFormOps(_euclidean_metric(1, 1, [4]))

        with self.assertRaisesRegex(ValueError, "only supports a 2D or 3D domain"):
            cov_curl(
                covector_1d, connection_1d, covector_1d,
                volume_form_1d, kernels_1d, index=0,
            )


class TestCovDirectionalDerivative(unittest.TestCase):
    """Test suite for the covariant directional derivative X^c nabla_c T."""

    def setUp(self) -> None:
        torch.manual_seed(0)
        self.kernels = Derivative(1.0, 1.0, dtype=DTYPE).two_dimensions()
        self.metric = _euclidean_metric(1, 2, [5, 5])
        self.connection = ConnectOps.levi_civita(self.metric, self.kernels)

        self.X = TensOps(
            torch.rand(1, 2, 5, 5, dtype=DTYPE),
            domain_dim=2,
            codomain_dim=[2],
            order=[1, 0],
            contravariant_indices=[0],
            covariant_indices=None,
        )

    def test_on_a_scalar_it_reduces_to_advection(self) -> None:
        """Under the flat hypothesis X^c nabla_c f is the plain advection."""

        scalar = TensOps(
            torch.rand(1, 1, 5, 5, dtype=DTYPE),
            domain_dim=2,
            codomain_dim=[1],
            order=[0, 0],
            contravariant_indices=None,
            covariant_indices=None,
        )

        result = cov_directional_derivative(
            self.X, scalar, self.connection, self.kernels)
        expected = advection(self.X, scalar, self.kernels)

        self.assertEqual(result.order, [0, 0])
        self.assertTrue(backend.allclose(
            result.values, expected.values, atol=1e-6))

    def test_preserves_the_type_of_the_differentiated_tensor(self) -> None:
        """Contracting the derivative axis leaves T's own indices free."""

        result = cov_directional_derivative(
            self.X, self.X, self.connection, self.kernels)

        self.assertEqual(result.order, [1, 0])
        self.assertEqual(result.codomain, [2])

    def test_requires_a_vector_field(self) -> None:
        """X must be a first-order contravariant field."""

        scalar = TensOps(
            torch.rand(1, 1, 5, 5, dtype=DTYPE),
            domain_dim=2,
            codomain_dim=[1],
            order=[0, 0],
            contravariant_indices=None,
            covariant_indices=None,
        )

        with self.assertRaisesRegex(ValueError, "requires X to be a vectorial field"):
            cov_directional_derivative(
                scalar, scalar, self.connection, self.kernels)


class TestLieDerivative(unittest.TestCase):
    """Test suite for the Lie derivative."""

    def setUp(self) -> None:
        torch.manual_seed(0)
        self.kernels = Derivative(1.0, 1.0, dtype=DTYPE).two_dimensions()
        self.metric = _euclidean_metric(1, 2, [5, 5])
        self.connection = ConnectOps.levi_civita(self.metric, self.kernels)

        self.X = TensOps(
            torch.rand(1, 2, 5, 5, dtype=DTYPE),
            domain_dim=2,
            codomain_dim=[2],
            order=[1, 0],
            contravariant_indices=[0],
            covariant_indices=None,
        )

    def test_lie_bracket_of_a_field_with_itself_vanishes(self) -> None:
        """[X, X] = 0 exactly: the directional term cancels the correction."""

        result = lie_derivative(self.X, self.X, self.connection, self.kernels)

        self.assertEqual(result.order, [1, 0])
        self.assertEqual(result.codomain, [2])
        self.assertTrue(backend.allclose(
            result.values,
            backend.zeros_like(result.values),
            atol=1e-6,
        ))

    def test_does_not_append_a_derivative_index(self) -> None:
        """Unlike cov_derivative, the Lie derivative preserves T's type."""

        covector = TensOps(
            torch.rand(1, 2, 5, 5, dtype=DTYPE),
            domain_dim=2,
            codomain_dim=[2],
            order=[0, 1],
            contravariant_indices=None,
            covariant_indices=[0],
        )

        result = lie_derivative(
            self.X, covector, self.connection, self.kernels)

        self.assertEqual(result.order, covector.order)
        self.assertEqual(result.codomain, covector.codomain)
        self.assertEqual(result.covariant, [0])

    def test_on_a_scalar_it_is_the_directional_derivative(self) -> None:
        """For a scalar there are no correction terms: L_X f = X^c d_c f."""

        scalar = TensOps(
            torch.rand(1, 1, 5, 5, dtype=DTYPE),
            domain_dim=2,
            codomain_dim=[1],
            order=[0, 0],
            contravariant_indices=None,
            covariant_indices=None,
        )

        result = lie_derivative(self.X, scalar, self.connection, self.kernels)
        expected = cov_directional_derivative(
            self.X, scalar, self.connection, self.kernels)

        self.assertEqual(result.order, [0, 0])
        self.assertTrue(backend.allclose(
            result.values, expected.values, atol=1e-6))

    def test_requires_a_vector_field(self) -> None:
        """X must be a first-order contravariant field."""

        scalar = TensOps(
            torch.rand(1, 1, 5, 5, dtype=DTYPE),
            domain_dim=2,
            codomain_dim=[1],
            order=[0, 0],
            contravariant_indices=None,
            covariant_indices=None,
        )

        with self.assertRaisesRegex(ValueError, "requires X to be a vectorial field"):
            lie_derivative(scalar, scalar, self.connection, self.kernels)

    def test_codomain_must_match_the_connection(self) -> None:
        """T's indices must range over the connection's dimension."""

        vector_3d = TensOps(
            torch.rand(1, 3, 5, 5, dtype=DTYPE),
            domain_dim=2,
            codomain_dim=[3],
            order=[1, 0],
            contravariant_indices=[0],
            covariant_indices=None,
        )

        with self.assertRaisesRegex(ValueError, "codomain dimensions must match"):
            lie_derivative(self.X, vector_3d, self.connection, self.kernels)


if __name__ == "__main__":
    unittest.main()
