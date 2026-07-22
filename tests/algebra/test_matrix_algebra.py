"""Matrix algebra unit tests following Google style guidelines."""

import os
import sys
import unittest
from typing import Sequence

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from conftest import A, u
from tensopsciml.backends import get_backend
from tensopsciml.tensors import MetrOps, TensOps
from tensopsciml.algebra.matrix_algebra import (
    determinant,
    deviatoric,
    inverse,
    spherical,
    trace,
    transpose,
)
from tensopsciml.algebra.tensor_algebra import kronecker_delta

backend = get_backend()


class TestMatrixAlgebra(unittest.TestCase):
    """Test suite for matrix algebra operations."""

    @staticmethod
    def _make_matrix(
        entries: Sequence[Sequence[Sequence[float]]],
        *,
        domain_dim: int = 1,
        order: Sequence[int] = [1, 1],
        contravariant: Sequence[int] | None = [0],
        covariant: Sequence[int] | None = [1],
    ) -> TensOps:
        """Wrap raw array into TensOps with matrix variance."""

        data = backend.tensor(entries, dtype=backend.float64)
        for _ in range(domain_dim):
            data = backend.unsqueeze(data, -1)

        contra_list = list(contravariant) if contravariant is not None else None
        covar_list = list(covariant) if covariant is not None else None
        total_order = sum(order)
        codomain = list(data.shape[1:1 + total_order])

        return TensOps(
            data,
            domain_dim=domain_dim,
            codomain_dim=codomain,
            order=list(order),
            contravariant_indices=contra_list,
            covariant_indices=covar_list,
        )

    def test_trace_simple_matrix(self) -> None:
        """Trace reduces a square matrix to a scalar field."""

        entries = [[[1.0, 2.0], [3.0, 5.0]]]
        matrix = self._make_matrix(entries)

        result = trace(matrix)
        expected = backend.einsum("niix->nx", matrix.values)
        expected = backend.unsqueeze(expected, 1)

        self.assertEqual(result.order, [0, 0])
        self.assertIsNone(result.contravariant)
        self.assertIsNone(result.covariant)
        self.assertTrue(backend.allclose(result.values, expected))

    def test_trace_three_dimensional_tensor(self) -> None:
        """Trace agrees with direct einsum on 3D tensor field."""

        result = trace(A)
        expected = backend.einsum("niixyz->nxyz", A.values)
        expected = backend.unsqueeze(expected, 1)

        self.assertTrue(backend.allclose(result.values, expected))
        self.assertEqual(result.order, [0, 0])
        self.assertEqual(result.codomain, [1])
        self.assertIsNone(result.contravariant)
        self.assertIsNone(result.covariant)

    def test_trace_non_square_matrix_raises(self) -> None:
        """Trace rejects non-square matrices."""

        entries = [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]
        matrix = self._make_matrix(entries)

        with self.assertRaisesRegex(ValueError, "square second-order tensor"):
            trace(matrix)

    def test_trace_wrong_order_raises(self) -> None:
        """Trace enforces that indices can be contracted."""

        entries = [[[1.0, 0.0], [0.0, 1.0]]]
        matrix = self._make_matrix(
            entries,
            order=(2, 0),
            contravariant=(0, 1),
            covariant=None,
        )

        with self.assertRaisesRegex(ValueError, "mixed tensor"):
            trace(matrix)

    def test_transpose_covariant_keeps_variance(self) -> None:
        """Transposing a fully covariant tensor is a pure axis swap.

        Both indices share the same variance, so the metadata is left
        untouched: A_{ij} -> A_{ji} is still order [0, 2].
        """
        entries = [[[1.0, 2.0], [3.0, 5.0]]]
        matrix = self._make_matrix(
            entries,
            order=(0, 2),
            contravariant=None,
            covariant=(0, 1),
        )

        result = transpose(matrix)
        expected = backend.transpose(matrix.values, (1, 2))

        self.assertTrue(backend.allclose(result.values, expected))
        self.assertEqual(result.order, [0, 2])
        self.assertIsNone(result.contravariant)
        self.assertEqual(result.covariant, [0, 1])

    def test_transpose_contravariant_keeps_variance(self) -> None:
        """Transposing a fully contravariant tensor is also a pure axis swap."""

        entries = [[[1.0, 2.0], [3.0, 5.0]]]
        matrix = self._make_matrix(
            entries,
            order=(2, 0),
            contravariant=(0, 1),
            covariant=None,
        )

        result = transpose(matrix)
        expected = backend.transpose(matrix.values, (1, 2))

        self.assertTrue(backend.allclose(result.values, expected))
        self.assertEqual(result.order, [2, 0])
        self.assertEqual(result.contravariant, [0, 1])
        self.assertIsNone(result.covariant)

    def test_transpose_mixed_tensor_raises(self) -> None:
        """Transpose rejects mixed tensors: swapping A^i_j needs the metric.

        Exchanging the variance of the two slots is only well-defined once an
        index is raised or lowered, so the mixed case (order [1, 1]) is sent to
        raise_index/lower_index instead of being silently transposed.
        """
        with self.assertRaisesRegex(ValueError, "same-variance second-order tensor"):
            transpose(A)

    def test_transpose_invalid_input_raises(self) -> None:
        """Transpose rejects tensors that are not second order."""

        with self.assertRaisesRegex(ValueError, "same-variance second-order tensor"):
            transpose(u)

    def test_determinant_two_by_two(self) -> None:
        """Determinant recovers analytic 2x2 value."""

        entries = [[[4.0, 7.0], [2.0, 6.0]]]
        matrix = self._make_matrix(entries)

        result = determinant(matrix)

        flattened = backend.squeeze(matrix.values, -1)
        expected = backend.det(flattened)
        expected = backend.unsqueeze(expected, 1)
        expected = backend.unsqueeze(expected, -1)

        self.assertTrue(backend.allclose(result.values, expected))
        self.assertEqual(result.order, [0, 0])
        self.assertEqual(result.codomain, [1])
        self.assertIsNone(result.contravariant)
        self.assertIsNone(result.covariant)
        # The internal Levi-Civita symbol must adopt the matrix's own dtype
        # instead of silently promoting/downcasting it.
        self.assertEqual(result.values.dtype, matrix.values.dtype)

    def test_determinant_three_by_three(self) -> None:
        """Determinant matches reference 3x3 computation."""

        entries = [
            [
                [1.0, 2.0, 3.0],
                [0.0, 1.0, 4.0],
                [5.0, 6.0, 0.0],
            ]
        ]
        matrix = self._make_matrix(
            entries,
            contravariant=(0,),
            covariant=(1,),
        )

        result = determinant(matrix)

        flattened = backend.squeeze(matrix.values, -1)
        expected = backend.det(flattened)
        expected = backend.unsqueeze(expected, 1)
        expected = backend.unsqueeze(expected, -1)

        self.assertTrue(backend.allclose(result.values, expected, atol=1e-6))
        self.assertEqual(result.order, [0, 0])
        self.assertEqual(result.codomain, [1])
        self.assertIsNone(result.contravariant)
        self.assertIsNone(result.covariant)

    def test_determinant_non_square_matrix_raises(self) -> None:
        """Determinant rejects non-square tensors."""

        entries = [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]
        matrix = self._make_matrix(entries)

        with self.assertRaisesRegex(ValueError, "square second-order tensor"):
            determinant(matrix)

    def test_determinant_unsupported_dimension_raises(self) -> None:
        """Determinant complains when matrix dimension exceeds support."""

        entries = [
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ]
        matrix = self._make_matrix(entries)

        with self.assertRaisesRegex(ValueError, "only supports 2x2 or 3x3"):
            determinant(matrix)

    def test_inverse_two_by_two(self) -> None:
        """Inverse mirrors backend inversion for 2x2 matrices."""

        entries = [[[4.0, 7.0], [2.0, 6.0]]]
        matrix = self._make_matrix(entries)

        result = inverse(matrix)

        flattened = backend.squeeze(matrix.values, -1)
        expected = backend.inv(flattened)
        expected = backend.unsqueeze(expected, -1)

        self.assertTrue(backend.allclose(result.values, expected, atol=1e-6))
        self.assertEqual(result.order, list(matrix.order))
        self.assertEqual(result.contravariant, matrix.contravariant)
        self.assertEqual(result.covariant, matrix.covariant)

    def test_inverse_three_by_three(self) -> None:
        """Inverse produces expected 3x3 result."""

        entries = [
            [
                [1.0, 2.0, 3.0],
                [0.0, 1.0, 4.0],
                [5.0, 6.0, 0.0],
            ]
        ]
        matrix = self._make_matrix(entries)

        result = inverse(matrix)

        flattened = backend.squeeze(matrix.values, -1)
        expected = backend.inv(flattened)
        expected = backend.unsqueeze(expected, -1)

        self.assertTrue(backend.allclose(result.values, expected, atol=1e-6))
        self.assertEqual(result.order, list(matrix.order))
        self.assertEqual(result.contravariant, matrix.contravariant)
        self.assertEqual(result.covariant, matrix.covariant)

    def test_inverse_non_square_matrix_raises(self) -> None:
        """Inverse rejects non-square matrices."""

        entries = [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]
        matrix = self._make_matrix(entries)

        with self.assertRaisesRegex(ValueError, "square matrix"):
            inverse(matrix)

    def test_inverse_singular_matrix_raises(self) -> None:
        """Inverse raises when matrix is singular."""

        entries = [[[1.0, 2.0], [2.0, 4.0]]]
        matrix = self._make_matrix(entries)

        with self.assertRaisesRegex(RuntimeError, "singular"):
            inverse(matrix)


class TestMatrixAlgebraDomainGuards(unittest.TestCase):
    """Test suite for the {1, 2, 3} domain guards of the matrix operators.

    TensOps refuses to build a tensor outside that range, so the guards can
    only be reached by forcing the domain after construction.
    """

    @staticmethod
    def _mixed_matrix() -> TensOps:
        """Build a valid 3D mixed matrix field."""
        return TensOps(
            backend.zeros((1, 2, 2, 4, 4, 4)),
            domain_dim=3,
            codomain_dim=[2, 2],
            order=[1, 1],
            contravariant_indices=[0],
            covariant_indices=[1],
        )

    @staticmethod
    def _covariant_matrix() -> TensOps:
        """Build a valid 3D fully covariant matrix field."""
        return TensOps(
            backend.zeros((1, 2, 2, 4, 4, 4)),
            domain_dim=3,
            codomain_dim=[2, 2],
            order=[0, 2],
            contravariant_indices=None,
            covariant_indices=[0, 1],
        )

    def test_trace_unsupported_domain_raises(self) -> None:
        """trace() rejects a domain outside {1, 2, 3}."""
        matrix = self._mixed_matrix()
        object.__setattr__(matrix, "_TensOps__domain", 4)

        with self.assertRaisesRegex(ValueError, "domain dimension of 1, 2, or 3"):
            trace(matrix)

    def test_transpose_unsupported_domain_raises(self) -> None:
        """transpose() rejects a domain outside {1, 2, 3}."""
        matrix = self._covariant_matrix()
        object.__setattr__(matrix, "_TensOps__domain", 4)

        with self.assertRaisesRegex(ValueError, "domain dimension of 1, 2, or 3"):
            transpose(matrix)

    def test_inverse_unsupported_domain_raises(self) -> None:
        """inverse() rejects a domain outside {1, 2, 3}."""
        matrix = self._mixed_matrix()
        object.__setattr__(matrix, "_TensOps__domain", 4)

        with self.assertRaisesRegex(ValueError, "domain dimension of 1, 2, or 3"):
            inverse(matrix)

    def test_inverse_non_matrix_raises(self) -> None:
        """inverse() requires a second-order tensor; a vector is rejected."""
        with self.assertRaisesRegex(ValueError, "second-order tensor"):
            inverse(u)


class TestSphericalDeviatoric(unittest.TestCase):
    """Test suite for the spherical/deviatoric split of a mixed tensor."""

    def setUp(self) -> None:
        """Build a 2x2 mixed tensor and the identity that scales its trace."""
        self.n_samples = 2
        self.discretization = [2, 2]

        identity = backend.eye(2).reshape(1, 2, 2, 1, 1).expand(
            self.n_samples, 2, 2, *self.discretization).clone()
        self.metric = MetrOps(
            identity, domain_dim=2, codomain_dim=[2, 2])

        # I is supplied by the caller (never fabricated inside the operator).
        self.I = kronecker_delta(self.metric)

        # tr(A) = 1 + 5 = 6, so the spherical part is (6/2) * I = 3 * I.
        entries = backend.tensor(
            [[1.0, 2.0], [3.0, 5.0]], dtype=backend.float32)
        values = entries.reshape(1, 2, 2, 1, 1).expand(
            self.n_samples, 2, 2, *self.discretization).clone()
        self.A = TensOps(
            values,
            domain_dim=2,
            codomain_dim=[2, 2],
            order=[1, 1],
            contravariant_indices=[0],
            covariant_indices=[1],
        )

    def test_spherical_is_the_scaled_trace(self) -> None:
        """spherical(A, I) = (1/n) tr(A) I."""

        result = spherical(self.A, self.I)

        self.assertEqual(result.order, [1, 1])
        self.assertEqual(result.codomain, [2, 2])

        expected = 3.0 * self.I.values
        self.assertTrue(backend.allclose(result.values, expected, atol=1e-6))

    def test_deviatoric_is_traceless(self) -> None:
        """The deviatoric part carries the whole trace away: tr(dev(A)) = 0."""

        result = deviatoric(self.A, self.I)
        result_trace = trace(result)

        self.assertEqual(result.order, [1, 1])
        self.assertTrue(backend.allclose(
            result_trace.values,
            backend.zeros(tuple(result_trace.values.shape)),
            atol=1e-5,
        ))

    def test_spherical_and_deviatoric_reconstruct_the_tensor(self) -> None:
        """A = spherical(A, I) + deviatoric(A, I)."""

        rebuilt = spherical(self.A, self.I) + deviatoric(self.A, self.I)

        self.assertTrue(backend.allclose(
            rebuilt.values, self.A.values, atol=1e-6))

    def test_spherical_rejects_same_variance_tensor(self) -> None:
        """The trace is only defined on the mixed form, so [0, 2] is rejected."""

        covariant = TensOps(
            self.A.values,
            domain_dim=2,
            codomain_dim=[2, 2],
            order=[0, 2],
            contravariant_indices=None,
            covariant_indices=[0, 1],
        )

        with self.assertRaisesRegex(ValueError, "square mixed second-order tensor"):
            spherical(covariant, self.I)

    def test_deviatoric_rejects_same_variance_tensor(self) -> None:
        """deviatoric() inherits spherical()'s mixed-tensor requirement."""

        covariant = TensOps(
            self.A.values,
            domain_dim=2,
            codomain_dim=[2, 2],
            order=[0, 2],
            contravariant_indices=None,
            covariant_indices=[0, 1],
        )

        with self.assertRaisesRegex(ValueError, "square mixed second-order tensor"):
            deviatoric(covariant, self.I)


if __name__ == "__main__":
    unittest.main()
