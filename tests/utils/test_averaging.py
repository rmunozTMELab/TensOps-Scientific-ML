"""Tests for the averaging module."""

import unittest

from tensopsciml.backends import get_backend
from tensopsciml.tensors import MetrOps, TensOps, VolFormOps
from tensopsciml.utils.averaging import m, m1, m2, m3

backend = get_backend()


def _identity_metric(domain_dim: int, dimension: int, grid_size: int = 10) -> MetrOps:
    """Build an identity metric on a `domain_dim`-dimensional grid."""
    shape = [10, dimension, dimension] + [grid_size] * domain_dim
    values = backend.eye(dimension).reshape(
        1, dimension, dimension, *([1] * domain_dim)
    ).expand(*shape).clone()
    return MetrOps(values, domain_dim=domain_dim, codomain_dim=[dimension, dimension])


class TestAveraging(unittest.TestCase):
    """Test suite covering directional averaging helpers."""

    def test_m1_valid_1d(self) -> None:
        """Ensure m1 averages a valid one-dimensional tensor."""
        data = backend.ones((1, 1, 10))
        tensor = TensOps(data, 1, [1], [0, 0], None, None)
        
        result = m1(tensor)
        expected = backend.ones((1, 1, 9))
        
        self.assertEqual(result.values.shape, expected.shape)
        self.assertTrue(backend.allclose(result.values, expected))

    def test_m1_valid_2d(self) -> None:
        """Ensure m1 averages a valid two-dimensional tensor."""
        data = backend.ones((1, 1, 10, 10))
        tensor = TensOps(data, 2, [1], [0, 0], None, None)
        
        result = m1(tensor)
        expected = backend.ones((1, 1, 9, 10))
        
        self.assertEqual(result.values.shape, expected.shape)
        self.assertTrue(backend.allclose(result.values, expected))

    def test_m1_valid_3d(self) -> None:
        """Ensure m1 averages a valid three-dimensional tensor."""
        data = backend.ones((1, 1, 10, 10, 10))
        tensor = TensOps(data, 3, [1], [0, 0], None, None)
        
        result = m1(tensor)
        expected = backend.ones((1, 1, 9, 10, 10))
        
        self.assertEqual(result.values.shape, expected.shape)
        self.assertTrue(backend.allclose(result.values, expected))

    def test_m2_valid_2d(self) -> None:
        """Ensure m2 averages a valid two-dimensional tensor."""
        data = backend.ones((1, 1, 10, 10))
        tensor = TensOps(data, 2, [1], [0, 0], None, None)
        
        result = m2(tensor)
        expected = backend.ones((1, 1, 10, 9))
        
        self.assertEqual(result.values.shape, expected.shape)
        self.assertTrue(backend.allclose(result.values, expected))

    def test_m2_valid_3d(self) -> None:
        """Ensure m2 averages a valid three-dimensional tensor."""
        data = backend.ones((1, 1, 10, 10, 10))
        tensor = TensOps(data, 3, [1], [0, 0], None, None)
        
        result = m2(tensor)
        expected = backend.ones((1, 1, 10, 9, 10))
        
        self.assertEqual(result.values.shape, expected.shape)
        self.assertTrue(backend.allclose(result.values, expected))

    def test_m2_invalid_1d(self) -> None:
        """Ensure m2 rejects one-dimensional tensors."""
        tensor = TensOps(backend.zeros((1, 1, 5)), 1, [1], [0, 0], None, None)
        with self.assertRaisesRegex(ValueError, "Cannot average along the second dimension"):
            m2(tensor)

    def test_m3_valid_3d(self) -> None:
        """Ensure m3 averages a valid three-dimensional tensor."""
        data = backend.ones((1, 1, 10, 10, 10))
        tensor = TensOps(data, 3, [1], [0, 0], None, None)
        
        result = m3(tensor)
        expected = backend.ones((1, 1, 10, 10, 9))
        
        self.assertEqual(result.values.shape, expected.shape)
        self.assertTrue(backend.allclose(result.values, expected))

    def test_m3_invalid_1d(self) -> None:
        """Ensure m3 rejects one-dimensional tensors."""
        tensor = TensOps(backend.zeros((1, 1, 5)), 1, [1], [0, 0], None, None)
        with self.assertRaisesRegex(ValueError, "Cannot average along the third dimension"):
            m3(tensor)

    def test_m3_invalid_2d(self) -> None:
        """Ensure m3 rejects two-dimensional tensors."""
        tensor = TensOps(backend.zeros((1, 1, 5, 5)), 2, [1], [0, 0], None, None)
        with self.assertRaisesRegex(ValueError, "Cannot average along the third dimension"):
            m3(tensor)

    def test_m_full_averaging_1d(self) -> None:
        """Ensure m averages across a one-dimensional domain."""
        tensor = TensOps(backend.ones((10, 1, 10)), 1, [1], [0, 0], None, None)
        result = m(tensor)
        expected = backend.ones((10, 1, 9))
        
        self.assertEqual(result.values.shape, expected.shape)
        self.assertTrue(backend.allclose(result.values, expected))

    def test_m_full_averaging_2d(self) -> None:
        """Ensure m averages across a two-dimensional domain."""
        tensor = TensOps(backend.ones((10, 2, 2, 10, 10)), 2, [2, 2], [1, 1], [0], [1])
        result = m(tensor)
        expected = backend.ones((10, 2, 2, 9, 9))
        
        self.assertEqual(result.values.shape, expected.shape)
        self.assertTrue(backend.allclose(result.values, expected))

    def test_m_full_averaging_3d(self) -> None:
        """Ensure m averages across a three-dimensional domain."""
        tensor = TensOps(backend.ones((10, 3, 3, 10, 10, 10)), 3, [3, 3], [1, 1], [0], [1])
        result = m(tensor)
        expected = backend.ones((10, 3, 3, 9, 9, 9))
        
        self.assertEqual(result.values.shape, expected.shape)
        self.assertTrue(backend.allclose(result.values, expected))
    
    def test_m_single_sample_batch(self) -> None:
        """Ensure m handles a batch of one sample without collapsing axes."""
        tensor = TensOps(backend.ones((1, 2, 2, 10, 10)), 2, [2, 2], [1, 1], [0], [1])
        result = m(tensor)
        expected = backend.ones((1, 2, 2, 9, 9))

        self.assertEqual(result.values.shape, expected.shape)
        self.assertTrue(backend.allclose(result.values, expected))

    def test_m_singleton_spatial_dimension(self) -> None:
        """Ensure m handles spatial dimensions that reduce to size one."""
        tensor = TensOps(backend.ones((10, 2, 2, 2)), 2, [2], [1, 0], [0], None)
        result = m(tensor)
        expected = backend.ones((10, 2, 1, 1))

        self.assertEqual(result.values.shape, expected.shape)
        self.assertTrue(backend.allclose(result.values, expected))

    def test_m1_invalid_order(self) -> None:
        """Ensure m1 rejects non-zero-order tensors."""
        tensor = TensOps(backend.zeros((1, 3, 5)), 1, [3], [1, 0], [0], None)
        with self.assertRaisesRegex(ValueError, r"scalar \(order-0\) field"):
            m1(tensor)

    def test_m2_invalid_order(self) -> None:
        """Ensure m2 rejects non-zero-order tensors."""
        tensor = TensOps(backend.zeros((1, 3, 5, 5)), 2, [3], [1, 0], [0], None)
        with self.assertRaisesRegex(ValueError, r"scalar \(order-0\) field"):
            m2(tensor)

    def test_m3_invalid_order(self) -> None:
        """Ensure m3 rejects non-zero-order tensors."""
        tensor = TensOps(backend.zeros((1, 3, 5, 5, 5)), 3, [3], [1, 0], [0], None)
        with self.assertRaisesRegex(ValueError, r"scalar \(order-0\) field"):
            m3(tensor)

    def test_m1_unsupported_domain(self) -> None:
        """m1 rejects tensors with unsupported spatial dimensions."""
        tensor = TensOps(backend.ones((1, 1, 4, 4, 4)), 3, [1], [0, 0], None, None)
        object.__setattr__(tensor, "_TensOps__domain", 4)
        with self.assertRaisesRegex(ValueError, "Unsupported space dimension"):
            m1(tensor)

    def test_m2_unsupported_domain(self) -> None:
        """m2 rejects tensors with unsupported spatial dimensions."""
        tensor = TensOps(backend.ones((1, 1, 4, 4, 4)), 3, [1], [0, 0], None, None)
        object.__setattr__(tensor, "_TensOps__domain", 4)
        with self.assertRaisesRegex(ValueError, "Unsupported space dimension"):
            m2(tensor)

    def test_m3_unsupported_domain(self) -> None:
        """m3 rejects tensors with unsupported spatial dimensions."""
        tensor = TensOps(backend.ones((1, 1, 4, 4, 4)), 3, [1], [0, 0], None, None)
        object.__setattr__(tensor, "_TensOps__domain", 4)
        with self.assertRaisesRegex(ValueError, "Unsupported space dimension"):
            m3(tensor)

    def test_m_unsupported_domain(self) -> None:
        """m rejects tensors outside the supported domain range."""
        tensor = TensOps(backend.ones((1, 1, 4, 4, 4)), 3, [1], [0, 0], None, None)
        object.__setattr__(tensor, "_TensOps__domain", 4)
        with self.assertRaisesRegex(ValueError, "Unsupported space dimension"):
            m(tensor)


class TestAveragingPreservesSpecializedTypes(unittest.TestCase):
    """Test suite covering m()'s restriction of MetrOps and VolFormOps."""

    def test_m_returns_metrops_for_metrops(self) -> None:
        """A restricted MetrOps stays a MetrOps, not a plain TensOps."""
        metric = _identity_metric(domain_dim=2, dimension=2)
        result = m(metric)

        self.assertIsInstance(result, MetrOps)
        self.assertEqual(result.values.shape, (10, 2, 2, 9, 9))
        self.assertEqual(result.order, [0, 2])
        self.assertEqual(result.covariant, [0, 1])
        self.assertEqual(result.domain, 2)
        self.assertEqual(result.codomain, [2, 2])

    def test_m_restricts_metric_values_onto_coarser_grid(self) -> None:
        """Averaging an identity metric leaves its values unchanged."""
        metric = _identity_metric(domain_dim=2, dimension=2)
        result = m(metric)
        expected = backend.eye(2).reshape(1, 2, 2, 1, 1).expand(10, 2, 2, 9, 9)

        self.assertTrue(backend.allclose(result.values, expected))

    def test_m_returns_volformops_for_volformops(self) -> None:
        """A restricted VolFormOps stays a VolFormOps, not a plain TensOps."""
        volume_form = VolFormOps(_identity_metric(domain_dim=2, dimension=2))
        result = m(volume_form)

        self.assertIsInstance(result, VolFormOps)
        self.assertEqual(result.values.shape, (10, 1, 9, 9))
        self.assertEqual(result.order, [0, 0])
        self.assertEqual(result.codomain, [1])
        self.assertEqual(result.domain, 2)

    def test_m_does_not_recompute_the_volume_form(self) -> None:
        """m only restricts the values already held: sqrt(det(I)) = 1 everywhere."""
        volume_form = VolFormOps(_identity_metric(domain_dim=2, dimension=2))
        result = m(volume_form)
        expected = backend.ones((10, 1, 9, 9))

        self.assertTrue(backend.allclose(result.values, expected))

    def test_m_returns_plain_tensops_for_plain_tensops(self) -> None:
        """A plain TensOps is not promoted to a specialized type."""
        tensor = TensOps(backend.ones((10, 2, 2, 10, 10)), 2, [2, 2], [1, 1], [0], [1])
        result = m(tensor)

        self.assertIsInstance(result, TensOps)
        self.assertNotIsInstance(result, MetrOps)
        self.assertNotIsInstance(result, VolFormOps)


if __name__ == "__main__":
    unittest.main()
