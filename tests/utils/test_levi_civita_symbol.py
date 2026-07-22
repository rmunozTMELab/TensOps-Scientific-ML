"""Tests for the Levi-Civita symbol generator.

This module covers levi_civita_symbol(), which builds the raw backend array of
+1/-1/0 values (a tensor density, not a tensor). The per-index sign logic it
delegates to is covered separately in test_indices.py.
"""

import unittest

from tensopsciml.backends import get_backend
from tensopsciml.utils.levi_civita_symbol import levi_civita_symbol

backend = get_backend()


class TestLeviCivitaSymbol(unittest.TestCase):
    """Test suite covering the Levi-Civita symbol generator."""

    def test_shape_for_order_three(self) -> None:
        """The symbol is shaped [N_data] + order * [order] + discretization."""
        e = levi_civita_symbol(10, 3, [5, 5, 5])
        self.assertEqual(tuple(e.shape), (10, 3, 3, 3, 5, 5, 5))

    def test_shape_for_order_two(self) -> None:
        """A second-order symbol carries two index axes of size two."""
        e = levi_civita_symbol(4, 2, [6, 6])
        self.assertEqual(tuple(e.shape), (4, 2, 2, 6, 6))

    def test_values_for_order_two(self) -> None:
        """The 2D symbol is the antisymmetric [[0, 1], [-1, 0]]."""
        e = levi_civita_symbol(1, 2, [3, 3])

        self.assertEqual(e[0, 0, 0, 0, 0].item(), 0)
        self.assertEqual(e[0, 0, 1, 0, 0].item(), 1)
        self.assertEqual(e[0, 1, 0, 0, 0].item(), -1)
        self.assertEqual(e[0, 1, 1, 0, 0].item(), 0)

    def test_values_for_order_three(self) -> None:
        """The 3D symbol is +1 on even permutations, -1 on odd, 0 on repeats."""
        e = levi_civita_symbol(1, 3, [2, 2, 2])

        for indices in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
            with self.subTest(indices=indices, parity="even"):
                self.assertEqual(e[0, *indices, 0, 0, 0].item(), 1)

        for indices in ((0, 2, 1), (2, 1, 0), (1, 0, 2)):
            with self.subTest(indices=indices, parity="odd"):
                self.assertEqual(e[0, *indices, 0, 0, 0].item(), -1)

        for indices in ((0, 0, 0), (1, 1, 2), (2, 2, 2), (0, 1, 1)):
            with self.subTest(indices=indices, parity="repeated"):
                self.assertEqual(e[0, *indices, 0, 0, 0].item(), 0)

    def test_values_are_constant_across_batch_and_grid(self) -> None:
        """The symbol is the same at every sample and every grid point."""
        e = levi_civita_symbol(4, 3, [3, 3, 3])
        reference = e[0, ..., 0, 0, 0]

        expected = reference.reshape(1, 3, 3, 3, 1, 1, 1).expand(4, 3, 3, 3, 3, 3, 3)
        self.assertTrue(backend.allclose(e, expected))

    def test_accepts_non_list_sequence_discretization(self) -> None:
        """A tuple discretization is coerced to a list rather than rejected."""
        e = levi_civita_symbol(2, 2, (4, 4))
        self.assertEqual(tuple(e.shape), (2, 2, 2, 4, 4))

    def test_rejects_non_sequence_discretization(self) -> None:
        """A non-sequence discretization is rejected."""
        with self.assertRaisesRegex(ValueError, "Discretization must be a list"):
            levi_civita_symbol(2, 2, 4)

    def test_rejects_string_discretization(self) -> None:
        """A string is a sequence, but not a valid discretization."""
        with self.assertRaisesRegex(ValueError, "Discretization must be a list"):
            levi_civita_symbol(2, 2, "44")

    def test_rejects_non_integer_order(self) -> None:
        """A non-integer order is rejected."""
        with self.assertRaisesRegex(ValueError, "must be an integer"):
            levi_civita_symbol(2, 2.0, [4, 4])

    def test_rejects_order_below_two(self) -> None:
        """An order of 1 or less is rejected."""
        for order in (1, 0, -1):
            with self.subTest(order=order):
                with self.assertRaisesRegex(ValueError, "greater than 1"):
                    levi_civita_symbol(2, order, [4, 4])

    def test_dtype_is_honoured(self) -> None:
        """An explicit dtype is passed through to the backend array."""
        e = levi_civita_symbol(2, 2, [3, 3], dtype=backend.float64)
        self.assertEqual(e.dtype, backend.float64)


if __name__ == "__main__":
    unittest.main()