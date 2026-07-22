"""Tests for the MetrOps and VolFormOps tensor classes.

This module covers the metric tensor specialization of TensOps: the fixed
covariant order and base of MetrOps, its square/symmetric/positive-definite
validation, the internal _from_inverse bypass used to build its contravariant
counterpart, its blanket rejection of arithmetic, and VolFormOps' sqrt(det(g))
computation together with its own _from_values bypass.

Classes:
    TestMetrOpsInitialization: Tests for MetrOps order, base, and happy path.
    TestMetrOpsValidation: Tests for square/symmetry/positive-definite checks.
    TestMetrOpsFromInverse: Tests for the internal contravariant bypass.
    TestMetrOpsArithmeticRejected: Tests that all arithmetic operators raise.
    TestVolFormOps: Tests for the sqrt(det(g)) scalar field.
    TestVolFormOpsFromValues: Tests for the internal restriction bypass.
"""

import unittest

import torch

from tests.conftest import backend
from tensopsciml.tensors import MetrOps, TensOps, VolFormOps


def _diag_metric_values(diag, *, n_samples=2, grid_size=4, domain_dim=1):
    """Build a [n_samples, d, d, *grid] tensor with `diag` at every grid point."""
    d = len(diag)
    base = torch.diag(torch.tensor(diag, dtype=torch.float32))
    shape = [n_samples, d, d] + [grid_size] * domain_dim
    return base.reshape(1, d, d, *([1] * domain_dim)).expand(*shape).clone()


class TestMetrOpsInitialization(unittest.TestCase):
    """Test suite for MetrOps order, base, and the happy path."""

    def test_metric_order_and_base(self) -> None:
        """A metric always has order [0, 2], no contravariant indices, and base V* x V*."""
        metric = MetrOps(
            _diag_metric_values([1.0, 1.0, 1.0], domain_dim=3, grid_size=3),
            domain_dim=3,
            codomain_dim=[3, 3],
        )
        self.assertEqual(metric.order, [0, 2])
        self.assertIsNone(metric.contravariant)
        self.assertEqual(metric.covariant, [0, 1])
        self.assertEqual(metric.base, "V* x V*")

    def test_euclidean_identity_metric_is_valid(self) -> None:
        """The Euclidean (identity) metric is square, symmetric, and positive definite."""
        values = _diag_metric_values([1.0, 1.0], domain_dim=1, grid_size=4)
        metric = MetrOps(values, domain_dim=1, codomain_dim=[2, 2])
        self.assertTrue(backend.allclose(metric.values, values))


class TestMetrOpsValidation(unittest.TestCase):
    """Test suite for MetrOps square/symmetry/positive-definite validation."""

    def test_non_square_codomain_rejected(self) -> None:
        """A non-square codomain [2, 3] must be rejected."""
        values = torch.randn(2, 2, 3, 4)
        with self.assertRaises(ValueError):
            MetrOps(values, domain_dim=1, codomain_dim=[2, 3])

    def test_asymmetric_metric_rejected(self) -> None:
        """A metric with g_{01} != g_{10} must be rejected."""
        base = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        values = base.reshape(1, 2, 2, 1).expand(2, 2, 2, 4).clone()
        with self.assertRaises(ValueError):
            MetrOps(values, domain_dim=1, codomain_dim=[2, 2])

    def test_non_positive_definite_metric_rejected(self) -> None:
        """A symmetric metric with a negative leading principal minor must be rejected."""
        values = _diag_metric_values([-1.0, 1.0], domain_dim=1, grid_size=4)
        with self.assertRaises(ValueError):
            MetrOps(values, domain_dim=1, codomain_dim=[2, 2])

    def test_degenerate_metric_rejected(self) -> None:
        """A symmetric metric with a zero determinant (degenerate) must be rejected."""
        values = _diag_metric_values([1.0, 0.0], domain_dim=1, grid_size=4)
        with self.assertRaises(ValueError):
            MetrOps(values, domain_dim=1, codomain_dim=[2, 2])


class TestMetrOpsFromInverse(unittest.TestCase):
    """Test suite for MetrOps._from_inverse, the contravariant bypass."""

    def test_from_inverse_builds_contravariant_metric(self) -> None:
        """_from_inverse builds a valid MetrOps with order [2, 0] and base V x V."""
        diag = [2.0, 5.0]
        metric = MetrOps(
            _diag_metric_values(diag, domain_dim=1, grid_size=4),
            domain_dim=1,
            codomain_dim=[2, 2],
        )
        inverse_values = _diag_metric_values(
            [1.0 / value for value in diag], domain_dim=1, grid_size=4
        )

        inv_metric = MetrOps._from_inverse(
            inverse_values,
            domain_dim=metric.domain,
            codomain_dim=metric.codomain,
            order=[2, 0],
            contravariant_indices=[0, 1],
            covariant_indices=None,
        )

        self.assertIsInstance(inv_metric, MetrOps)
        self.assertEqual(inv_metric.order, [2, 0])
        self.assertEqual(inv_metric.contravariant, [0, 1])
        self.assertIsNone(inv_metric.covariant)
        self.assertEqual(inv_metric.base, "V x V")
        self.assertTrue(backend.allclose(inv_metric.values, inverse_values))

    def test_from_inverse_still_validates_symmetry(self) -> None:
        """_from_inverse still runs the square/symmetry/positive-definite checks."""
        base = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        asymmetric_values = base.reshape(1, 2, 2, 1).expand(2, 2, 2, 4).clone()

        with self.assertRaises(ValueError):
            MetrOps._from_inverse(
                asymmetric_values,
                domain_dim=1,
                codomain_dim=[2, 2],
                order=[2, 0],
                contravariant_indices=[0, 1],
                covariant_indices=None,
            )


class TestMetrOpsArithmeticRejected(unittest.TestCase):
    """Test suite covering the blanket rejection of arithmetic on MetrOps."""

    def setUp(self) -> None:
        self.metric = MetrOps(
            _diag_metric_values([1.0, 1.0], domain_dim=1, grid_size=4),
            domain_dim=1,
            codomain_dim=[2, 2],
        )
        self.other = MetrOps(
            _diag_metric_values([2.0, 2.0], domain_dim=1, grid_size=4),
            domain_dim=1,
            codomain_dim=[2, 2],
        )

    def test_addition_rejected(self) -> None:
        with self.assertRaises(TypeError) as ctx:
            _ = self.metric + self.other
        self.assertIn("MetrOps", str(ctx.exception))

    def test_subtraction_rejected(self) -> None:
        with self.assertRaises(TypeError):
            _ = self.metric - self.other

    def test_multiplication_by_scalar_rejected(self) -> None:
        with self.assertRaises(TypeError):
            _ = self.metric * 2.0

    def test_right_multiplication_by_scalar_rejected(self) -> None:
        with self.assertRaises(TypeError):
            _ = 2.0 * self.metric

    def test_division_by_scalar_rejected(self) -> None:
        with self.assertRaises(TypeError):
            _ = self.metric / 2.0

    def test_negation_rejected(self) -> None:
        with self.assertRaises(TypeError):
            _ = -self.metric

    def test_unary_plus_rejected(self) -> None:
        with self.assertRaises(TypeError):
            _ = +self.metric


class TestVolFormOps(unittest.TestCase):
    """Test suite for the VolFormOps sqrt(det(g)) scalar field."""

    def setUp(self) -> None:
        self.diag = [4.0, 9.0]
        self.metric = MetrOps(
            _diag_metric_values(self.diag, domain_dim=1, grid_size=4),
            domain_dim=1,
            codomain_dim=[2, 2],
        )

    def test_volume_form_matches_sqrt_det(self) -> None:
        """The volume form equals sqrt(det(g)) = sqrt(4 * 9) = 6 everywhere."""
        volume_form = VolFormOps(self.metric)
        expected = torch.full((2, 1, 4), 6.0, dtype=torch.float32)

        self.assertTrue(backend.allclose(volume_form.values, expected))

    def test_volume_form_scalar_properties(self) -> None:
        """The volume form is a scalar field: order [0, 0], codomain [1]."""
        volume_form = VolFormOps(self.metric)

        self.assertEqual(volume_form.order, [0, 0])
        self.assertEqual(volume_form.codomain, [1])
        self.assertEqual(volume_form.domain, self.metric.domain)
        self.assertIsNone(volume_form.contravariant)
        self.assertIsNone(volume_form.covariant)

    def test_rejects_non_metrops_input(self) -> None:
        """A plain TensOps (not a MetrOps) must be rejected."""
        plain_tensor = TensOps(
            torch.randn(2, 2, 2, 4),
            domain_dim=1,
            codomain_dim=[2, 2],
            order=[1, 1],
            contravariant_indices=[0],
            covariant_indices=[1],
        )
        with self.assertRaises(ValueError):
            VolFormOps(plain_tensor)


class TestVolFormOpsFromValues(unittest.TestCase):
    """Test suite for VolFormOps._from_values, the restriction bypass."""

    def test_from_values_bypasses_sqrt_det_computation(self) -> None:
        """_from_values wraps data directly, without recomputing sqrt(det(g))."""
        data = torch.full((2, 1, 4), 3.0, dtype=torch.float32)
        volume_form = VolFormOps._from_values(data, domain_dim=1)

        self.assertIsInstance(volume_form, VolFormOps)
        self.assertEqual(volume_form.order, [0, 0])
        self.assertEqual(volume_form.codomain, [1])
        self.assertEqual(volume_form.domain, 1)
        self.assertTrue(backend.allclose(volume_form.values, data))


if __name__ == "__main__":
    unittest.main()
