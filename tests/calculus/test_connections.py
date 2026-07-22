"""Tests for the affine connection (ConnectOps).

A connection is deliberately not a TensOps: its coefficients Gamma^k_{ij} do
not transform as a tensor. These tests cover the container's own validation,
its index-layout vocabulary (upper/lower rather than contravariant/covariant),
and the Levi-Civita factory that derives the Christoffel symbols of a metric.
"""

import unittest

from tensopsciml.backends import get_backend
from tensopsciml.calculus.connections import ConnectOps
from tensopsciml.kernels.derivative import Derivative
from tensopsciml.tensors import MetrOps, TensOps

backend = get_backend()


def _diagonal_metric(
    diagonal: list[float],
    n_samples: int,
    discretization: list[int],
    *,
    dtype=None,
) -> MetrOps:
    """Build a constant diagonal metric g_ij on the given grid."""
    dtype = dtype if dtype is not None else backend.float32
    dimension = len(diagonal)

    matrix = backend.tensor(
        [[diagonal[i] if i == j else 0.0 for j in range(dimension)]
         for i in range(dimension)],
        dtype=dtype,
    )
    shape = [n_samples, dimension, dimension] + list(discretization)
    values = matrix.reshape(
        1, dimension, dimension, *([1] * len(discretization))
    ).expand(*shape).clone()

    return MetrOps(
        values,
        domain_dim=len(discretization),
        codomain_dim=[dimension, dimension],
    )


class TestConnectOpsInitialization(unittest.TestCase):
    """Test suite for the ConnectOps container and its index layout."""

    def setUp(self) -> None:
        self.connection = ConnectOps(
            backend.zeros((2, 2, 2, 2, 5, 5)),
            domain_dim=2,
            codomain_dim=[2, 2, 2],
        )

    def test_index_layout(self) -> None:
        """One upper index (axis 0) and two lower ones (axes 1 and 2)."""

        self.assertEqual(self.connection.upper_indices, [0])
        self.assertEqual(self.connection.lower_indices, [1, 2])

    def test_metadata_properties(self) -> None:
        """Domain, codomain and the space dimension are exposed."""

        self.assertEqual(self.connection.domain, 2)
        self.assertEqual(self.connection.codomain, [2, 2, 2])
        self.assertEqual(self.connection.dimension, 2)
        self.assertEqual(tuple(self.connection.values.shape), (2, 2, 2, 2, 5, 5))

    def test_is_not_a_tensops(self) -> None:
        """A connection is not a tensor, so it must not pass as a TensOps."""

        self.assertNotIsInstance(self.connection, TensOps)

    def test_string_representation(self) -> None:
        """The repr summarizes the connection's structure."""

        description = str(self.connection)

        self.assertIn("ConnectOps", description)
        self.assertEqual(repr(self.connection), description)


class TestConnectOpsValidation(unittest.TestCase):
    """Test suite for the connection's metadata and shape validation."""

    def test_unsupported_domain_raises(self) -> None:
        """The domain must be 1, 2 or 3."""

        with self.assertRaisesRegex(ValueError, "must be one of"):
            ConnectOps(
                backend.zeros((1, 2, 2, 2, 4, 4, 4, 4)),
                domain_dim=4,
                codomain_dim=[2, 2, 2],
            )

    def test_codomain_must_have_three_entries(self) -> None:
        """A connection carries exactly three indices."""

        with self.assertRaisesRegex(ValueError, "three integer"):
            ConnectOps(
                backend.zeros((1, 2, 2, 4, 4)),
                domain_dim=2,
                codomain_dim=[2, 2],
            )

    def test_codomain_must_be_a_list_of_ints(self) -> None:
        """Non-integer codomain entries are rejected."""

        with self.assertRaisesRegex(ValueError, "three integer"):
            ConnectOps(
                backend.zeros((1, 2, 2, 2, 4, 4)),
                domain_dim=2,
                codomain_dim=[2, 2, 2.0],
            )

    def test_codomain_dimensions_must_be_equal(self) -> None:
        """The three indices must range over the same dimension."""

        with self.assertRaisesRegex(ValueError, "same dimension"):
            ConnectOps(
                backend.zeros((1, 2, 2, 3, 4, 4)),
                domain_dim=2,
                codomain_dim=[2, 2, 3],
            )

    def test_shape_must_match_the_metadata(self) -> None:
        """The coefficient tensor must match the declared index structure."""

        with self.assertRaisesRegex(ValueError, "Inconsistent connection shape"):
            ConnectOps(
                backend.zeros((1, 3, 3, 3, 4, 4)),
                domain_dim=2,
                codomain_dim=[2, 2, 2],
            )


class TestLeviCivitaConnection(unittest.TestCase):
    """Test suite for the Levi-Civita (Christoffel) factory."""

    def setUp(self) -> None:
        self.dtype = backend.float32
        self.kernels = Derivative(
            1.0, 1.0, dtype=self.dtype).two_dimensions()

    def test_christoffel_symbols_of_a_flat_metric_vanish(self) -> None:
        """A constant (Cartesian) metric has zero Christoffel symbols.

        Gamma is built from derivatives of g, so a metric that is constant
        across the grid must produce an identically zero connection.
        """
        metric = _diagonal_metric([1.0, 1.0], 1, [5, 5], dtype=self.dtype)

        connection = ConnectOps.levi_civita(metric, self.kernels)

        self.assertIsInstance(connection, ConnectOps)
        self.assertEqual(connection.dimension, 2)
        self.assertEqual(connection.codomain, [2, 2, 2])
        # The connection lives on the differentiated grid, one node smaller.
        self.assertEqual(tuple(connection.values.shape), (1, 2, 2, 2, 4, 4))
        self.assertTrue(backend.allclose(
            connection.values,
            backend.zeros_like(connection.values),
            atol=1e-6,
        ))

    def test_constant_non_identity_metric_also_vanishes(self) -> None:
        """Flatness, not unit scale, is what makes Gamma vanish."""

        metric = _diagonal_metric([4.0, 9.0], 1, [5, 5], dtype=self.dtype)

        connection = ConnectOps.levi_civita(metric, self.kernels)

        self.assertTrue(backend.allclose(
            connection.values,
            backend.zeros_like(connection.values),
            atol=1e-6,
        ))

    def test_levi_civita_rejects_non_metrops(self) -> None:
        """The factory needs a MetrOps, not a plain TensOps."""

        plain = TensOps(
            backend.ones((1, 2, 2, 5, 5), dtype=self.dtype),
            domain_dim=2,
            codomain_dim=[2, 2],
            order=[0, 2],
            contravariant_indices=None,
            covariant_indices=[0, 1],
        )

        with self.assertRaisesRegex(ValueError, "Expected a MetrOps"):
            ConnectOps.levi_civita(plain, self.kernels)

    def test_levi_civita_requires_codomain_to_match_domain(self) -> None:
        """The connection's indices must range over the tangent space."""

        # A 3x3 metric over a 2D domain: the tangent space rank disagrees with
        # the number of spatial dimensions.
        metric = _diagonal_metric([1.0, 1.0, 1.0], 1, [5, 5], dtype=self.dtype)

        with self.assertRaisesRegex(ValueError, "match its domain dimension"):
            ConnectOps.levi_civita(metric, self.kernels)


if __name__ == "__main__":
    unittest.main()
