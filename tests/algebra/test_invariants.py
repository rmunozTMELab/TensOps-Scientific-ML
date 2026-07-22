"""Tests for the tensor invariants module."""

import os
import sys
import unittest
from typing import Sequence

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from conftest import A
from tensopsciml.backends import get_backend
from tensopsciml.tensors import TensOps
from tensopsciml.algebra.invariants import (
    first_invariant,
    second_invariant,
    third_invariant,
)

backend = get_backend()


class TestInvariants(unittest.TestCase):
    """Test suite for tensor invariant computations."""

    @staticmethod
    def _make_tensor(
        entries: Sequence[Sequence[Sequence[float]]],
        *,
        domain_dim: int = 1,
        order: Sequence[int] = [1, 1],
        contravariant: Sequence[int] | None = [0],
        covariant: Sequence[int] | None = [1],
    ) -> TensOps:
        """Wrap raw entries into a ``TensOps`` tensor with the requested variance."""

        data = backend.tensor(entries, dtype=backend.float64)
        for _ in range(domain_dim):
            data = backend.unsqueeze(data, -1)

        contra_list = list(contravariant) if contravariant is not None else None
        covar_list = list(covariant) if covariant is not None else None
        total_order = sum(order)
        codomain = list(data.shape[1 : 1 + total_order])

        return TensOps(
            data,
            domain_dim=domain_dim,
            codomain_dim=codomain,
            order=list(order),
            contravariant_indices=contra_list,
            covariant_indices=covar_list,
        )

    def test_first_invariant_matches_trace(self) -> None:
        """First invariant matches the explicit trace computation."""

        entries = [[[1.0, 2.0], [3.0, 5.0]]]
        tensor = self._make_tensor(entries)

        result = first_invariant(tensor)

        flattened = backend.squeeze(tensor.values, -1)
        expected = backend.einsum("nii->n", flattened)
        expected = backend.unsqueeze(expected, 1)
        expected = backend.unsqueeze(expected, -1)

        self.assertEqual(result.order, [0, 0])
        self.assertEqual(result.codomain, [1])
        self.assertIsNone(result.contravariant)
        self.assertIsNone(result.covariant)
        self.assertTrue(backend.allclose(result.values, expected))

    def test_first_invariant_non_square_raises(self) -> None:
        """First invariant rejects non-square matrices."""

        entries = [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]
        tensor = self._make_tensor(entries, order=[1, 1])

        with self.assertRaisesRegex(ValueError, "square second-order tensor"):
            first_invariant(tensor)

    def test_first_invariant_with_fixture_tensor(self) -> None:
        """First invariant agrees with backend trace for fixture tensor."""

        result = first_invariant(A)
        expected = backend.einsum("niixyz->nxyz", A.values)
        expected = backend.unsqueeze(expected, 1)

        self.assertTrue(backend.allclose(result.values, expected))
        self.assertEqual(result.order, [0, 0])
        self.assertEqual(result.codomain, [1])
        self.assertIsNone(result.contravariant)
        self.assertIsNone(result.covariant)

    def test_second_invariant_matches_formula(self) -> None:
        """Second invariant matches the analytical ``0.5*(tr(A)^2 - tr(A^2))``."""

        entries = [[[4.0, 7.0], [2.0, 6.0]]]
        tensor = self._make_tensor(entries)

        result = second_invariant(tensor)

        flattened = backend.squeeze(tensor.values, -1)
        trace_a = backend.einsum("nii->n", flattened)
        squared = backend.einsum("nij,njk->nik", flattened, flattened)
        trace_squared = backend.einsum("nii->n", squared)
        expected = 0.5 * (trace_a * trace_a - trace_squared)
        expected = backend.unsqueeze(expected, 1)
        expected = backend.unsqueeze(expected, -1)

        self.assertTrue(backend.allclose(result.values, expected, atol=1e-6))
        self.assertEqual(result.order, [0, 0])
        self.assertEqual(result.codomain, [1])
        self.assertIsNone(result.contravariant)
        self.assertIsNone(result.covariant)

    def test_second_invariant_contravariant_tensor_raises(self) -> None:
        """Second invariant rejects fully contravariant tensors."""

        entries = [[[2.0, 1.0], [3.0, 4.0]]]
        tensor = self._make_tensor(
            entries,
            order=[2, 0],
            contravariant=[0, 1],
            covariant=None,
        )

        with self.assertRaisesRegex(ValueError, "contract"):
            second_invariant(tensor)

    def test_second_invariant_covariant_tensor_raises(self) -> None:
        """Second invariant rejects fully covariant tensors."""

        entries = [[[3.0, -1.0], [2.0, 5.0]]]
        tensor = self._make_tensor(
            entries,
            order=[0, 2],
            contravariant=None,
            covariant=[0, 1],
        )

        with self.assertRaisesRegex(ValueError, "contract"):
            second_invariant(tensor)

    def test_second_invariant_invalid_order_raises(self) -> None:
        """Second invariant rejects tensors that are not second order."""

        entries = [[1.0, 2.0]]
        tensor = self._make_tensor(
            entries,
            order=[1, 0],
            contravariant=[0],
            covariant=None,
        )

        with self.assertRaisesRegex(ValueError, "second-order"):
            second_invariant(tensor)

    def test_third_invariant_two_by_two(self) -> None:
        """Third invariant matches the determinant for 2x2 matrices."""

        entries = [[[4.0, 7.0], [2.0, 6.0]]]
        tensor = self._make_tensor(entries)

        result = third_invariant(tensor)

        flattened = backend.squeeze(tensor.values, -1)
        expected = backend.det(flattened)
        expected = backend.unsqueeze(expected, 1)
        expected = backend.unsqueeze(expected, -1)

        self.assertTrue(backend.allclose(result.values, expected, atol=1e-6))
        self.assertEqual(result.order, [0, 0])
        self.assertEqual(result.codomain, [1])
        self.assertIsNone(result.contravariant)
        self.assertIsNone(result.covariant)

    def test_third_invariant_three_by_three(self) -> None:
        """Third invariant matches the determinant for 3x3 matrices."""

        entries = [
            [
                [1.0, 2.0, 3.0],
                [0.0, 1.0, 4.0],
                [5.0, 6.0, 0.0],
            ]
        ]
        tensor = self._make_tensor(entries)

        result = third_invariant(tensor)

        flattened = backend.squeeze(tensor.values, -1)
        expected = backend.det(flattened)
        expected = backend.unsqueeze(expected, 1)
        expected = backend.unsqueeze(expected, -1)

        self.assertTrue(backend.allclose(result.values, expected, atol=1e-6))
        self.assertEqual(result.order, [0, 0])
        self.assertEqual(result.codomain, [1])
        self.assertIsNone(result.contravariant)
        self.assertIsNone(result.covariant)

    def test_third_invariant_non_square_raises(self) -> None:
        """Third invariant rejects non-square tensors."""

        entries = [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]
        tensor = self._make_tensor(entries)

        with self.assertRaisesRegex(ValueError, "square second-order tensor"):
            third_invariant(tensor)


if __name__ == "__main__":
    unittest.main()
