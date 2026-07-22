"""Tensor algebra unit tests following Google style guidelines."""

import os
import re
import sys
import unittest

import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from conftest import A, B, C
from conftest import phi, psi
from conftest import u, v, w, x
from tensopsciml.algebra.matrix_algebra import inverse
from tensopsciml.backends import get_backend
from tensopsciml.tensors import MetrOps, TensOps
from tensopsciml.utils.levi_civita_symbol import levi_civita_symbol
from tensopsciml.algebra.tensor_algebra import (
    antisymmetrization_contravariant,
    antisymmetrization_covariant,
    antisymmetrize,
    contraction,
    cross_prod,
    kronecker_delta,
    levi_civita_tensor,
    lower_index,
    raise_index,
    self_contraction,
    symmetrization_contravariant,
    symmetrization_covariant,
    symmetrize,
    tensor_product,
)

backend = get_backend()


def _diagonal_metric(
    diagonal: list[float],
    n_samples: int,
    discretization: list[int],
    *,
    dtype=None,
) -> MetrOps:
    """Build a constant diagonal metric g_ij on the given grid.

    The metric is always supplied explicitly by the caller: the library never
    fabricates one internally, so the tests must provide it too. A diagonal
    (rather than identity) metric is what proves raise_index/lower_index really
    contract against g instead of merely relabelling variance.
    """
    dimension = len(diagonal)
    dtype = dtype if dtype is not None else backend.float32

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


# Euclidean metric matching the conftest fixtures' grid (10 samples, 3D
# codomain, 40^3 discretization). Lowering/raising against it leaves values
# untouched, which is what the metadata-only assertions below rely on.
EUCLIDEAN_METRIC = _diagonal_metric(
    [1.0, 1.0, 1.0],
    u.values.shape[0],
    list(u.values.shape[-u.domain:]),
    dtype=u.values.dtype,
)
EUCLIDEAN_INV_METRIC = inverse(EUCLIDEAN_METRIC)


class TestTensorAlgebra(unittest.TestCase):
    """Test suite for tensor algebra operations."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        pass

    def _sample_rank_three_tensor(self):
        """Builds deterministic data tensor for symmetry tests."""

        values = np.arange(8, dtype=np.float64).reshape(1, 2, 2, 2, 1)
        return backend.tensor(values, dtype=backend.float64)

    def _grid_vector(
        self, components: list[float], n_samples: int, discretization: list[int]
    ) -> TensOps:
        """Build a contravariant vector field holding `components` at every point."""

        dimension = len(components)
        values = backend.tensor(
            components, dtype=backend.float32
        ).reshape(1, dimension, *([1] * len(discretization)))
        values = values.expand(
            n_samples, dimension, *discretization).clone()

        return TensOps(
            values,
            domain_dim=len(discretization),
            codomain_dim=[dimension],
            order=[1, 0],
            contravariant_indices=[0],
            covariant_indices=None,
        )

    def _make_vector(self, n_data: int, dim: int, *, contravariant: bool) -> TensOps:
        """Builds a minimal vector TensOps with a custom batch size and dimension."""

        values = backend.tensor(
            np.zeros((n_data, dim, 1)), dtype=backend.float64)
        return TensOps(
            values,
            domain_dim=1,
            codomain_dim=[dim],
            order=[1, 0] if contravariant else [0, 1],
            contravariant_indices=[0] if contravariant else None,
            covariant_indices=None if contravariant else [0],
        )

    def test_contraction_invalid(self) -> None:
        """Test invalid specifications for tensor contraction."""

        u_contravariant = u
        v_contravariant = v

        u_covariant = lower_index(EUCLIDEAN_METRIC, u)
        v_covariant = lower_index(EUCLIDEAN_METRIC, v)

        invalid_cases = [
            (
                "contravariant mismatch",
                u_contravariant,
                v_contravariant,
                [[0], [None]],
                [[0], [None]],
                "Contravariant and covariant indices must be compatible for double contraction.",
            ),
            (
                "covariant mismatch",
                u_covariant,
                v_covariant,
                [[None], [0]],
                [[None], [0]],
                "Contravariant and covariant indices must be compatible for double contraction.",
            ),
            (
                "duplicate contravariant indices",
                u_contravariant,
                v_contravariant,
                [[0, 0], [None]],
                [[None], [0]],
                "A tensor index cannot be contracted twice.",
            ),
            (
                "duplicate covariant indices",
                u_covariant,
                v_covariant,
                [[None], [0, 0]],
                [[None], [0]],
                "A tensor index cannot be contracted twice.",
            ),
            (
                "malformed indices specification",
                A,
                B,
                [0, None],
                [[None], [0]],
                "Indices to be contracted must be provided as a list shaped [[], []].",
            ),
            (
                "duplicate indices in second tensor",
                u_covariant,
                v_covariant,
                [[None], [0]],
                [[None], [0, 0]],
                "A tensor index cannot be contracted twice.",
            ),
            (
                "matrix vector variance mismatch",
                A,
                u,
                [[0], [None]],
                [[0], [None]],
                "Contravariant and covariant indices must be compatible for double contraction.",
            ),
            (
                "matrix vector duplicate indices",
                A,
                u,
                [[None], [0]],
                [[0, 0], [None]],
                "A tensor index cannot be contracted twice.",
            ),
            (
                "scalar operand",
                phi,
                u,
                [[None], [None]],
                [[None], [None]],
                "contraction() doesn't support scalar operands",
            ),
            (
                "batch size mismatch",
                self._make_vector(2, 3, contravariant=True),
                self._make_vector(3, 3, contravariant=False),
                [[0], [None]],
                [[None], [0]],
                "contraction() requires matching batch sizes",
            ),
            (
                "codomain dimension mismatch",
                self._make_vector(2, 3, contravariant=True),
                self._make_vector(2, 2, contravariant=False),
                [[0], [None]],
                [[None], [0]],
                "Contracted codomain dimensions must match",
            ),
        ]

        for name, tensor_a, tensor_b, a_indices, b_indices, message in invalid_cases:
            with self.subTest(case=name):
                with self.assertRaisesRegex(ValueError, re.escape(message)):
                    contraction(tensor_a, tensor_b, a_indices, b_indices)

    def test_contraction_valid(self) -> None:
        """Test valid tensor contraction scenarios."""

        v_covariant = lower_index(EUCLIDEAN_METRIC, v)

        result_scalar = contraction(u, v_covariant, [[0], [None]], [[None], [0]])
        expected_scalar = backend.einsum('nixyz,nixyz->nxyz', u.values, v_covariant.values)
        expected_scalar = backend.unsqueeze(expected_scalar, 1)

        self.assertEqual(result_scalar.order, [0, 0])
        self.assertIsNone(result_scalar.contravariant)
        self.assertIsNone(result_scalar.covariant)
        self.assertEqual(result_scalar.codomain, [1])
        self.assertTrue(backend.allclose(result_scalar.values, expected_scalar))

        result_vector = contraction(A, v, [[None], [0]], [[0], [None]])
        expected_vector = backend.einsum('nijxyz,njxyz->nixyz', A.values, v.values)

        self.assertEqual(result_vector.order, [1, 0])
        self.assertEqual(result_vector.contravariant, [0])
        self.assertIsNone(result_vector.covariant)
        self.assertEqual(result_vector.codomain, [3])
        self.assertTrue(backend.allclose(result_vector.values, expected_vector))

        result_matrix_vector = contraction(A, u, [[None], [0]], [[0], [None]])
        expected_matrix_vector = backend.einsum('nijxyz,njxyz->nixyz', A.values, u.values)

        self.assertEqual(result_matrix_vector.order, [1, 0])
        self.assertEqual(result_matrix_vector.contravariant, [0])
        self.assertIsNone(result_matrix_vector.covariant)
        self.assertEqual(result_matrix_vector.codomain, [3])
        self.assertTrue(backend.allclose(result_matrix_vector.values, expected_matrix_vector))

    def test_tensor_product_valid(self) -> None:
        """Tensor product produces expected variance combinations."""

        u_covariant = lower_index(EUCLIDEAN_METRIC, u)
        v_covariant = lower_index(EUCLIDEAN_METRIC, v)

        vector_outer = tensor_product(u, v)
        expected_vector_outer = backend.einsum('nixyz,njxyz->nijxyz', u.values, v.values)

        self.assertEqual(vector_outer.order, [2, 0])
        self.assertEqual(vector_outer.contravariant, [0, 1])
        self.assertIsNone(vector_outer.covariant)
        self.assertEqual(vector_outer.codomain, [3, 3])
        self.assertTrue(backend.allclose(vector_outer.values, expected_vector_outer))

        covector_outer = tensor_product(u_covariant, v_covariant)
        expected_covector_outer = backend.einsum('nixyz,njxyz->nijxyz', u_covariant.values, v_covariant.values)

        self.assertEqual(covector_outer.order, [0, 2])
        self.assertIsNone(covector_outer.contravariant)
        self.assertEqual(covector_outer.covariant, [0, 1])
        self.assertEqual(covector_outer.codomain, [3, 3])
        self.assertTrue(backend.allclose(covector_outer.values, expected_covector_outer))

        mixed_outer = tensor_product(u, v_covariant)
        expected_mixed_outer = backend.einsum('nixyz,njxyz->nijxyz', u.values, v_covariant.values)

        self.assertEqual(mixed_outer.order, [1, 1])
        self.assertEqual(mixed_outer.contravariant, [0])
        self.assertEqual(mixed_outer.covariant, [1])
        self.assertEqual(mixed_outer.codomain, [3, 3])
        self.assertTrue(backend.allclose(mixed_outer.values, expected_mixed_outer))

    def test_tensor_product_scalar_gives_pointwise_multiplication(self) -> None:
        """Scalar ⊗ T = scalar * T (pointwise multiplication)."""
        # Create a scalar field with the same domain shape as u
        scalar_shape = (u.values.shape[0], 1) + u.values.shape[2:]
        scalar_values = backend.ones(scalar_shape, dtype=u.values.dtype)
        scalar = TensOps(
            scalar_values,
            domain_dim=3,
            codomain_dim=[1],
            order=[0, 0],
            contravariant_indices=None,
            covariant_indices=None,
        )
        # scalar * u should equal u * scalar by commutativity
        result_scalar_u = tensor_product(scalar, u)
        result_u_scalar = tensor_product(u, scalar)

        # Both should preserve u's variance and shape
        self.assertEqual(result_scalar_u.order, u.order)
        self.assertEqual(result_scalar_u.contravariant, u.contravariant)
        self.assertEqual(result_u_scalar.order, u.order)
        self.assertEqual(result_u_scalar.contravariant, u.contravariant)

        # Values should match: scalar * u.values (with broadcasting)
        expected = u.values * scalar_values
        self.assertTrue(backend.allclose(result_scalar_u.values, expected))
        self.assertTrue(backend.allclose(result_u_scalar.values, expected))

    def test_tensor_product_scalar_with_rank_two_tensor(self) -> None:
        """Scalar ⊗ rank-2 tensor aligns axes instead of misbroadcasting."""

        result = tensor_product(phi, A)

        self.assertEqual(result.order, A.order)
        self.assertEqual(result.codomain, A.codomain)
        self.assertEqual(result.contravariant, A.contravariant)
        self.assertEqual(result.covariant, A.covariant)

        # phi: (N, 1, x, y, z) -> (N, 1, 1, x, y, z) to align with A's codomain
        expected = A.values * backend.unsqueeze(phi.values, 1)
        self.assertTrue(backend.allclose(result.values, expected))

        # Commutativity: T ⊗ scalar must give the same values
        result_right = tensor_product(A, phi)
        self.assertTrue(backend.allclose(result_right.values, expected))

    def test_tensor_product_scalar_batch_equal_codomain(self) -> None:
        """Scalar ⊗ rank-2 is correct even when N_data equals a codomain dim.

        With naive broadcasting, a scalar of shape (3, 1, spatial) against a
        tensor of shape (3, 3, 3, spatial) silently pairs the scalar's batch
        axis with the tensor's first codomain axis, producing wrong values
        instead of an error. This is the regression guard for that case.
        """
        n_matched = A.codomain[0]
        scalar_small = TensOps(
            phi.values[:n_matched],
            domain_dim=3,
            codomain_dim=[1],
            order=[0, 0],
            contravariant_indices=None,
            covariant_indices=None,
        )
        tensor_small = TensOps(
            A.values[:n_matched],
            domain_dim=3,
            codomain_dim=[3, 3],
            order=[1, 1],
            contravariant_indices=[0],
            covariant_indices=[1],
        )

        result = tensor_product(scalar_small, tensor_small)

        expected = tensor_small.values * backend.unsqueeze(
            scalar_small.values, 1)
        self.assertTrue(backend.allclose(result.values, expected))
    def test_tensor_product_invalid(self) -> None:
        """Tensor product rejects incompatible tensors."""

        n_samples = u.values.shape[0]
        mismatch = backend.zeros(
            [n_samples, 3, 4, 4],
            dtype=u.values.dtype,
            device=u.values.device,
        )
        tensor_2d = TensOps(
            mismatch,
            domain_dim=2,
            codomain_dim=[3],
            order=[1, 0],
            contravariant_indices=[0],
            covariant_indices=None,
        )

        with self.assertRaisesRegex(ValueError, "Not compatible dimensions between tensors."):
            tensor_product(u, tensor_2d)

    def test_self_contraction_valid(self) -> None:
        """Test valid tensor self-contraction scenarios."""

        result_scalar = self_contraction(A, [0], [0])
        expected_scalar = backend.einsum('niixyz->nxyz', A.values)
        expected_scalar = backend.unsqueeze(expected_scalar, 1)

        self.assertEqual(result_scalar.order, [0, 0])
        self.assertIsNone(result_scalar.contravariant)
        self.assertIsNone(result_scalar.covariant)
        self.assertEqual(result_scalar.codomain, [1])
        self.assertTrue(backend.allclose(result_scalar.values, expected_scalar))

    def test_self_contraction_invalid(self) -> None:
        """Invalid configurations raise descriptive errors."""

        invalid_cases = [
            (
                "missing covariant variance",
                u,
                [0],
                [0],
                ValueError,
                "Both contravariant and covariant orders of T must be non-zero for self-contraction.",
            ),
            (
                "contravariant type mismatch",
                A,
                (0,),
                [0],
                TypeError,
                "Both contravariant_index and covariant_index must be provided as lists.",
            ),
            (
                "covariant type mismatch",
                A,
                [0],
                (0,),
                TypeError,
                "Both contravariant_index and covariant_index must be provided as lists.",
            ),
            (
                "length mismatch",
                A,
                [0, 0],
                [0],
                ValueError,
                "The lengths of contravariant_index and covariant_index must match for self-contraction.",
            ),
            (
                "contravariant index out of range",
                A,
                [1],
                [0],
                ValueError,
                "All indices in contravariant_index must be within the range [0, T.order[0]).",
            ),
            (
                "covariant index out of range",
                A,
                [0],
                [1],
                ValueError,
                "All indices in covariant_index must be within the range [0, T.order[1]).",
            ),
            (
                "duplicate contravariant indices",
                A,
                [0, 0],
                [0, 0],
                ValueError,
                "Duplicate entries found in contravariant_index.",
            ),
        ]

        for name, tensor, contra_idx, cov_idx, error, message in invalid_cases:
            with self.subTest(case=name):
                with self.assertRaisesRegex(error, re.escape(message)):
                    self_contraction(tensor, contra_idx, cov_idx)

    def test_symmetrization_covariant(self) -> None:
        """Covariant symmetrization averages over selected indices."""

        data = self._sample_rank_three_tensor()
        tensor = TensOps(
            data,
            domain_dim=1,
            codomain_dim=[2, 2, 2],
            order=[1, 2],
            contravariant_indices=[0],
            covariant_indices=[1, 2],
        )

        result = symmetrization_covariant(tensor, sym_indices=[0, 1])
        expected = 0.5 * (data + backend.permute(data, (0, 1, 3, 2, 4)))

        self.assertTrue(backend.allclose(result.values, expected))
        self.assertEqual(result.order, tensor.order)
        self.assertEqual(result.contravariant, tensor.contravariant)
        self.assertEqual(result.covariant, tensor.covariant)

    def test_symmetrization_contravariant(self) -> None:
        """Contravariant symmetrization averages over selected indices."""

        data = self._sample_rank_three_tensor()
        tensor = TensOps(
            data,
            domain_dim=1,
            codomain_dim=[2, 2, 2],
            order=[2, 1],
            contravariant_indices=[0, 1],
            covariant_indices=[2],
        )

        result = symmetrization_contravariant(tensor, sym_indices=[0, 1])
        expected = 0.5 * (data + backend.permute(data, (0, 2, 1, 3, 4)))

        self.assertTrue(backend.allclose(result.values, expected))
        self.assertEqual(result.order, tensor.order)
        self.assertEqual(result.contravariant, tensor.contravariant)
        self.assertEqual(result.covariant, tensor.covariant)

    def test_antisymmetrization_covariant(self) -> None:
        """Covariant antisymmetrization produces alternating sum."""

        data = self._sample_rank_three_tensor()
        tensor = TensOps(
            data,
            domain_dim=1,
            codomain_dim=[2, 2, 2],
            order=[1, 2],
            contravariant_indices=[0],
            covariant_indices=[1, 2],
        )

        result = antisymmetrization_covariant(tensor, sym_indices=[0, 1])
        expected = 0.5 * (data - backend.permute(data, (0, 1, 3, 2, 4)))

        self.assertTrue(backend.allclose(result.values, expected))
        self.assertEqual(result.order, tensor.order)
        self.assertEqual(result.contravariant, tensor.contravariant)
        self.assertEqual(result.covariant, tensor.covariant)

    def test_antisymmetrization_contravariant(self) -> None:
        """Contravariant antisymmetrization produces alternating sum."""

        data = self._sample_rank_three_tensor()
        tensor = TensOps(
            data,
            domain_dim=1,
            codomain_dim=[2, 2, 2],
            order=[2, 1],
            contravariant_indices=[0, 1],
            covariant_indices=[2],
        )

        result = antisymmetrization_contravariant(tensor, sym_indices=[0, 1])
        expected = 0.5 * (data - backend.permute(data, (0, 2, 1, 3, 4)))

        self.assertTrue(backend.allclose(result.values, expected))
        self.assertEqual(result.order, tensor.order)
        self.assertEqual(result.contravariant, tensor.contravariant)
        self.assertEqual(result.covariant, tensor.covariant)

    def test_antisymmetrization_covariant_unsorted_indices_raises(self) -> None:
        """Unsorted sym_indices must raise instead of flipping the sign."""

        data = self._sample_rank_three_tensor()
        tensor = TensOps(
            data,
            domain_dim=1,
            codomain_dim=[2, 2, 2],
            order=[1, 2],
            contravariant_indices=[0],
            covariant_indices=[1, 2],
        )

        with self.assertRaises(ValueError):
            antisymmetrization_covariant(tensor, sym_indices=[1, 0])

    def test_antisymmetrization_contravariant_unsorted_indices_raises(self) -> None:
        """Unsorted sym_indices must raise instead of flipping the sign."""

        data = self._sample_rank_three_tensor()
        tensor = TensOps(
            data,
            domain_dim=1,
            codomain_dim=[2, 2, 2],
            order=[2, 1],
            contravariant_indices=[0, 1],
            covariant_indices=[2],
        )

        with self.assertRaises(ValueError):
            antisymmetrization_contravariant(tensor, sym_indices=[1, 0])

    def test_symmetrization_covariant_on_tensor_without_covariant_raises(self) -> None:
        """Symmetrization over covariant indices requires the tensor to have them."""
        # Contravariant-only tensor (covariant=None)
        tensor = TensOps(
            backend.ones((2, 3, 5, 5, 5)),
            domain_dim=3,
            codomain_dim=[3],
            order=[1, 0],
            contravariant_indices=[0],
            covariant_indices=None,
        )
        with self.assertRaisesRegex(ValueError, "requires T to have covariant"):
            symmetrization_covariant(tensor, [0, 1])

    def test_symmetrization_contravariant_on_tensor_without_contravariant_raises(self) -> None:
        """Symmetrization over contravariant indices requires the tensor to have them."""
        # Covariant-only tensor (contravariant=None)
        tensor = TensOps(
            backend.ones((2, 3, 5, 5, 5)),
            domain_dim=3,
            codomain_dim=[3],
            order=[0, 1],
            contravariant_indices=None,
            covariant_indices=[0],
        )
        with self.assertRaisesRegex(ValueError, "requires T to have"):
            symmetrization_contravariant(tensor, [0, 1])

    def test_antisymmetrization_covariant_on_tensor_without_covariant_raises(self) -> None:
        """Antisymmetrization over covariant indices requires the tensor to have them."""
        # Contravariant-only tensor (covariant=None)
        tensor = TensOps(
            backend.ones((2, 3, 5, 5, 5)),
            domain_dim=3,
            codomain_dim=[3],
            order=[1, 0],
            contravariant_indices=[0],
            covariant_indices=None,
        )
        with self.assertRaisesRegex(ValueError, "requires T to have covariant"):
            antisymmetrization_covariant(tensor, [0, 1])

    def test_antisymmetrization_contravariant_on_tensor_without_contravariant_raises(self) -> None:
        """Antisymmetrization over contravariant indices requires the tensor to have them."""
        # Covariant-only tensor (contravariant=None)
        tensor = TensOps(
            backend.ones((2, 3, 5, 5, 5)),
            domain_dim=3,
            codomain_dim=[3],
            order=[0, 1],
            contravariant_indices=None,
            covariant_indices=[0],
        )
        with self.assertRaisesRegex(ValueError, "requires T to have"):
            antisymmetrization_contravariant(tensor, [0, 1])
    def test_symmetrize_dispatches_on_variance(self) -> None:
        """symmetrize() delegates to the covariant/contravariant implementation."""

        data = self._sample_rank_three_tensor()
        tensor = TensOps(
            data,
            domain_dim=1,
            codomain_dim=[2, 2, 2],
            order=[1, 2],
            contravariant_indices=[0],
            covariant_indices=[1, 2],
        )

        result = symmetrize(tensor, [0, 1], "covariant")
        expected = symmetrization_covariant(tensor, sym_indices=[0, 1])

        self.assertTrue(backend.allclose(result.values, expected.values))

    def test_antisymmetrize_dispatches_on_variance(self) -> None:
        """antisymmetrize() delegates to the covariant/contravariant implementation."""

        data = self._sample_rank_three_tensor()
        tensor = TensOps(
            data,
            domain_dim=1,
            codomain_dim=[2, 2, 2],
            order=[2, 1],
            contravariant_indices=[0, 1],
            covariant_indices=[2],
        )

        result = antisymmetrize(tensor, [0, 1], "contravariant")
        expected = antisymmetrization_contravariant(tensor, sym_indices=[0, 1])

        self.assertTrue(backend.allclose(result.values, expected.values))

    def test_symmetrize_unknown_variance_raises(self) -> None:
        """symmetrize() only accepts 'covariant' or 'contravariant'."""

        with self.assertRaisesRegex(ValueError, "'covariant' or 'contravariant'"):
            symmetrize(A, [0, 1], "mixed")

    def test_antisymmetrize_unknown_variance_raises(self) -> None:
        """antisymmetrize() only accepts 'covariant' or 'contravariant'."""

        with self.assertRaisesRegex(ValueError, "'covariant' or 'contravariant'"):
            antisymmetrize(A, [0, 1], "mixed")

    def test_cross_prod_three_components(self) -> None:
        """Cross product of 3-vectors contracts the Levi-Civita tensor.

        Contracting two of E's three covariant indices leaves the third free,
        so the result is a covector (order [0, 1]), not a vector.
        """
        n_samples, discretization = 2, [2, 2, 2]
        metric = _diagonal_metric([1.0, 1.0, 1.0], n_samples, discretization)
        symbol = levi_civita_symbol(n_samples, 3, discretization)
        E = levi_civita_tensor(
            symbol,
            domain_dim=3,
            codomain_dim=[3, 3, 3],
            order=[0, 3],
            metric=metric,
        )

        e1 = self._grid_vector([1.0, 0.0, 0.0], n_samples, discretization)
        e2 = self._grid_vector([0.0, 1.0, 0.0], n_samples, discretization)

        result = cross_prod(e1, e2, E)

        self.assertEqual(result.order, [0, 1])
        self.assertIsNone(result.contravariant)
        self.assertEqual(result.covariant, [0])
        self.assertEqual(result.codomain, [3])

        # e1 x e2 = e3 in a Euclidean metric.
        expected = self._grid_vector(
            [0.0, 0.0, 1.0], n_samples, discretization).values
        self.assertTrue(backend.allclose(result.values, expected))

    def test_cross_prod_two_components(self) -> None:
        """The 2D cross product is the scalar exterior product u1*v2 - u2*v1."""

        n_samples, discretization = 2, [2, 2, 2]
        a = self._grid_vector([1.0, 0.0], n_samples, discretization)
        b = self._grid_vector([0.0, 1.0], n_samples, discretization)

        result = cross_prod(a, b)

        self.assertEqual(result.order, [0, 0])
        self.assertEqual(result.codomain, [1])
        self.assertTrue(backend.allclose(
            result.values, backend.ones(tuple(result.values.shape))))

    def test_cross_prod_requires_levi_civita_in_three_dimensions(self) -> None:
        """The Levi-Civita tensor is mandatory for 3-component vectors."""

        n_samples, discretization = 2, [2, 2, 2]
        a = self._grid_vector([1.0, 0.0, 0.0], n_samples, discretization)
        b = self._grid_vector([0.0, 1.0, 0.0], n_samples, discretization)

        with self.assertRaisesRegex(ValueError, "requires the covariant Levi-Civita"):
            cross_prod(a, b)

    def test_cross_prod_rejects_levi_civita_in_two_dimensions(self) -> None:
        """The Levi-Civita tensor must be omitted for 2-component vectors."""

        n_samples, discretization = 2, [2, 2, 2]
        metric = _diagonal_metric([1.0, 1.0, 1.0], n_samples, discretization)
        symbol = levi_civita_symbol(n_samples, 3, discretization)
        E = levi_civita_tensor(
            symbol,
            domain_dim=3,
            codomain_dim=[3, 3, 3],
            order=[0, 3],
            metric=metric,
        )

        a = self._grid_vector([1.0, 0.0], n_samples, discretization)
        b = self._grid_vector([0.0, 1.0], n_samples, discretization)

        with self.assertRaisesRegex(ValueError, "omit the 'E' argument"):
            cross_prod(a, b, E)

    def test_cross_prod_invalid(self) -> None:
        """Cross product rejects non-vector inputs and unsupported codomain."""

        with self.assertRaisesRegex(
            ValueError,
            "To compute cross products both tensors must be vectors",
        ):
            cross_prod(A, u)

        n_samples = u.values.shape[0]
        domain_dims = list(u.values.shape[-u.domain:])
        large_codomain = backend.zeros(
            [n_samples, 4, *domain_dims],
            dtype=u.values.dtype,
            device=u.values.device,
        )
        vector4 = TensOps(
            large_codomain,
            domain_dim=u.domain,
            codomain_dim=[4],
            order=[1, 0],
            contravariant_indices=[0],
            covariant_indices=None,
        )
        with self.assertRaisesRegex(
            ValueError,
            "To compute cross product vectors codomain must be 2 or 3.",
        ):
            cross_prod(u, vector4)

    def test_levi_civita_tensor_covariant(self) -> None:
        """A covariant metric yields the fully covariant tensor, scaled by sqrt(det g)."""

        n_samples, discretization = 2, [2, 2, 2]
        metric = _diagonal_metric([1.0, 1.0, 1.0], n_samples, discretization)
        symbol = levi_civita_symbol(n_samples, 3, discretization)

        E = levi_civita_tensor(
            symbol,
            domain_dim=3,
            codomain_dim=[3, 3, 3],
            order=[0, 3],
            metric=metric,
        )

        self.assertEqual(E.order, [0, 3])
        self.assertIsNone(E.contravariant)
        self.assertEqual(E.covariant, [0, 1, 2])
        self.assertEqual(E.codomain, [3, 3, 3])

        # sqrt(det(I)) = 1, so the tensor equals the raw symbol here.
        self.assertTrue(backend.allclose(E.values, symbol))

    def test_levi_civita_tensor_scales_by_the_volume_form(self) -> None:
        """A non-identity metric scales the symbol by sqrt(det(g))."""

        n_samples, discretization = 2, [2, 2, 2]
        # det(diag(4, 1, 1)) = 4 -> sqrt(det g) = 2.
        metric = _diagonal_metric([4.0, 1.0, 1.0], n_samples, discretization)
        symbol = levi_civita_symbol(n_samples, 3, discretization)

        E = levi_civita_tensor(
            symbol,
            domain_dim=3,
            codomain_dim=[3, 3, 3],
            order=[0, 3],
            metric=metric,
        )

        self.assertTrue(backend.allclose(E.values, 2.0 * symbol))

    def test_levi_civita_tensor_contravariant(self) -> None:
        """An inverse metric yields the fully contravariant tensor."""

        n_samples, discretization = 2, [2, 2, 2]
        metric = _diagonal_metric([1.0, 1.0, 1.0], n_samples, discretization)
        inv_metric = inverse(metric)
        symbol = levi_civita_symbol(n_samples, 3, discretization)

        E = levi_civita_tensor(
            symbol,
            domain_dim=3,
            codomain_dim=[3, 3, 3],
            order=[3, 0],
            inv_metric=inv_metric,
        )

        self.assertEqual(E.order, [3, 0])
        self.assertEqual(E.contravariant, [0, 1, 2])
        self.assertIsNone(E.covariant)

    def test_levi_civita_tensor_requires_exactly_one_metric(self) -> None:
        """Neither both metrics nor none of them is accepted."""

        n_samples, discretization = 2, [2, 2, 2]
        metric = _diagonal_metric([1.0, 1.0, 1.0], n_samples, discretization)
        inv_metric = inverse(metric)
        symbol = levi_civita_symbol(n_samples, 3, discretization)

        with self.assertRaisesRegex(ValueError, "exactly one of"):
            levi_civita_tensor(
                symbol, 3, [3, 3, 3], [0, 3],
                metric=metric, inv_metric=inv_metric,
            )

        with self.assertRaisesRegex(ValueError, "exactly one of"):
            levi_civita_tensor(symbol, 3, [3, 3, 3], [0, 3])

    def test_kronecker_delta_is_the_identity(self) -> None:
        """delta^i_j = g^{ik} g_{kj} is the identity for any metric."""

        n_samples, discretization = 2, [2, 2]
        metric = _diagonal_metric([2.0, 5.0], n_samples, discretization)

        delta = kronecker_delta(metric)

        self.assertEqual(delta.order, [1, 1])
        self.assertEqual(delta.contravariant, [0])
        self.assertEqual(delta.covariant, [1])
        self.assertEqual(delta.codomain, [2, 2])

        # Identity at every point of the grid, despite the non-identity metric.
        expected = backend.eye(2).reshape(1, 2, 2, 1, 1).expand(
            n_samples, 2, 2, *discretization)
        self.assertTrue(backend.allclose(delta.values, expected, atol=1e-6))

    def test_kronecker_delta_rejects_contravariant_metric(self) -> None:
        """kronecker_delta() expects the covariant metric, order [0, 2]."""

        n_samples, discretization = 2, [2, 2]
        metric = _diagonal_metric([2.0, 5.0], n_samples, discretization)
        inv_metric = inverse(metric)

        with self.assertRaisesRegex(ValueError, r"order \[0, 2\]"):
            kronecker_delta(inv_metric)

    def test_lower_index_contracts_against_the_metric(self) -> None:
        """Lowering contracts g_ij against the vector: T_i = g_ij u^j."""

        n_samples, discretization = 2, [2, 2, 2]
        metric = _diagonal_metric([2.0, 3.0, 4.0], n_samples, discretization)
        vector = self._grid_vector([1.0, 1.0, 1.0], n_samples, discretization)

        lowered = lower_index(metric, vector)

        self.assertEqual(lowered.order, [0, 1])
        self.assertIsNone(lowered.contravariant)
        self.assertEqual(lowered.covariant, [0])

        # A diagonal metric scales each component by its diagonal entry: this
        # is what distinguishes a real contraction from a variance relabel.
        expected = self._grid_vector(
            [2.0, 3.0, 4.0], n_samples, discretization).values
        self.assertTrue(backend.allclose(lowered.values, expected))

    def test_lower_index_euclidean_metric_preserves_values(self) -> None:
        """With g = I, lowering changes the variance but not the values."""

        lowered = lower_index(EUCLIDEAN_METRIC, u)

        self.assertEqual(lowered.order, [0, 1])
        self.assertIsNone(lowered.contravariant)
        self.assertEqual(lowered.covariant, [0])
        self.assertTrue(backend.allclose(lowered.values, u.values))

    def test_raise_index_inverts_lower_index(self) -> None:
        """Raising with g^{ij} undoes lowering with g_{ij}."""

        n_samples, discretization = 2, [2, 2, 2]
        metric = _diagonal_metric([2.0, 3.0, 4.0], n_samples, discretization)
        inv_metric = inverse(metric)
        vector = self._grid_vector([1.0, 2.0, 3.0], n_samples, discretization)

        lowered = lower_index(metric, vector)
        raised = raise_index(inv_metric, lowered)

        self.assertEqual(raised.order, [1, 0])
        self.assertEqual(raised.contravariant, [0])
        self.assertIsNone(raised.covariant)
        self.assertTrue(backend.allclose(
            raised.values, vector.values, atol=1e-6))

    def test_lower_index_requires_a_covariant_metric(self) -> None:
        """lower_index() rejects the inverse (contravariant) metric."""

        with self.assertRaisesRegex(ValueError, r"order \[0, 2\]"):
            lower_index(EUCLIDEAN_INV_METRIC, u)

    def test_raise_index_requires_a_contravariant_metric(self) -> None:
        """raise_index() rejects the covariant metric."""

        u_covariant = lower_index(EUCLIDEAN_METRIC, u)

        with self.assertRaisesRegex(ValueError, r"order \[2, 0\]"):
            raise_index(EUCLIDEAN_METRIC, u_covariant)

    def test_contraction_covariant_codomain_mismatch(self) -> None:
        """The covariant-against-contravariant pairing checks sizes too.

        The mirror of the contravariant check: here T's covariant index is
        contracted against S's contravariant one, and their sizes must agree.
        """
        covector = self._make_vector(2, 3, contravariant=False)
        vector = self._make_vector(2, 2, contravariant=True)

        with self.assertRaisesRegex(
            ValueError, "Contracted codomain dimensions must match"
        ):
            contraction(covector, vector, [[None], [0]], [[0], [None]])

    def test_contraction_domain_mismatch_raises(self) -> None:
        """Contracting tensors defined on different domains is rejected."""

        covector_2d = TensOps(
            backend.zeros((10, 3, 4, 4), dtype=u.values.dtype),
            domain_dim=2,
            codomain_dim=[3],
            order=[0, 1],
            contravariant_indices=None,
            covariant_indices=[0],
        )

        with self.assertRaisesRegex(ValueError, "Space dimension not supported"):
            contraction(u, covector_2d, [[0], [None]], [[None], [0]])

    def test_self_contraction_duplicate_covariant_indices(self) -> None:
        """A covariant index cannot be contracted twice either."""

        tensor = TensOps(
            backend.zeros((1, 2, 2, 2, 2, 4), dtype=backend.float32),
            domain_dim=1,
            codomain_dim=[2, 2, 2, 2],
            order=[2, 2],
            contravariant_indices=[0, 1],
            covariant_indices=[2, 3],
        )

        with self.assertRaisesRegex(
            ValueError, "Duplicate entries found in covariant_index"
        ):
            self_contraction(tensor, [0, 1], [0, 0])

    def test_self_contraction_leaving_free_indices(self) -> None:
        """A partial self-contraction keeps the indices it does not contract."""

        tensor = TensOps(
            backend.ones((1, 2, 2, 2, 4), dtype=backend.float32),
            domain_dim=1,
            codomain_dim=[2, 2, 2],
            order=[2, 1],
            contravariant_indices=[0, 1],
            covariant_indices=[2],
        )

        result = self_contraction(tensor, [1], [0])

        # One contravariant index survives; the result is not a scalar.
        self.assertEqual(result.order, [1, 0])
        self.assertEqual(result.codomain, [2])
        self.assertEqual(result.contravariant, [0])
        self.assertIsNone(result.covariant)

    def test_tensor_product_unsupported_domain_raises(self) -> None:
        """tensor_product() rejects a domain outside {1, 2, 3} with a ValueError.

        Distinct from the domain-mismatch guard above: here both operands agree
        on a domain that simply isn't supported.
        """
        vector = TensOps(
            backend.zeros((1, 2, 4), dtype=backend.float32),
            domain_dim=1,
            codomain_dim=[2],
            order=[1, 0],
            contravariant_indices=[0],
            covariant_indices=None,
        )
        object.__setattr__(vector, "_TensOps__domain", 4)

        with self.assertRaisesRegex(ValueError, "Unsupported space dimension"):
            tensor_product(vector, vector)

    def test_self_contraction_unsupported_domain_raises(self) -> None:
        """self_contraction() rejects a domain outside {1, 2, 3} with a ValueError.

        The einsum discretization labels are only looked up once the domain is
        known to be supported, so an unsupported one reaches the guard instead
        of dying earlier on a raw KeyError.
        """
        tensor = TensOps(
            backend.zeros((1, 2, 2, 4), dtype=backend.float32),
            domain_dim=1,
            codomain_dim=[2, 2],
            order=[1, 1],
            contravariant_indices=[0],
            covariant_indices=[1],
        )
        object.__setattr__(tensor, "_TensOps__domain", 4)

        with self.assertRaisesRegex(ValueError, "Unsupported space dimension"):
            self_contraction(tensor, [0], [0])

    def test_self_contraction_without_indices_is_a_no_op(self) -> None:
        """Empty index lists contract nothing and leave the tensor untouched."""

        result = self_contraction(A, [], [])

        self.assertEqual(result.order, A.order)
        self.assertEqual(result.codomain, A.codomain)
        self.assertTrue(backend.allclose(result.values, A.values))

    def test_symmetrization_requires_exactly_two_indices(self) -> None:
        """Symmetrization is defined over an index pair, not an arbitrary set."""

        data = self._sample_rank_three_tensor()
        covariant_pair = TensOps(
            data,
            domain_dim=1,
            codomain_dim=[2, 2, 2],
            order=[1, 2],
            contravariant_indices=[0],
            covariant_indices=[1, 2],
        )
        contravariant_pair = TensOps(
            data,
            domain_dim=1,
            codomain_dim=[2, 2, 2],
            order=[2, 1],
            contravariant_indices=[0, 1],
            covariant_indices=[2],
        )

        with self.assertRaisesRegex(ValueError, "exactly 2 indices"):
            symmetrization_covariant(covariant_pair, sym_indices=[0, 1, 2])

        with self.assertRaisesRegex(ValueError, "exactly 2 indices"):
            symmetrization_contravariant(
                contravariant_pair, sym_indices=[0, 1, 2])

        with self.assertRaisesRegex(ValueError, "exactly 2 indices"):
            antisymmetrization_covariant(covariant_pair, sym_indices=[0, 1, 2])

        with self.assertRaisesRegex(ValueError, "exactly 2 indices"):
            antisymmetrization_contravariant(
                contravariant_pair, sym_indices=[0, 1, 2])

    def test_symmetrize_contravariant_branch(self) -> None:
        """symmetrize() also dispatches to the contravariant implementation."""

        data = self._sample_rank_three_tensor()
        tensor = TensOps(
            data,
            domain_dim=1,
            codomain_dim=[2, 2, 2],
            order=[2, 1],
            contravariant_indices=[0, 1],
            covariant_indices=[2],
        )

        result = symmetrize(tensor, [0, 1], "contravariant")
        expected = symmetrization_contravariant(tensor, sym_indices=[0, 1])

        self.assertTrue(backend.allclose(result.values, expected.values))

    def test_antisymmetrize_covariant_branch(self) -> None:
        """antisymmetrize() also dispatches to the covariant implementation."""

        data = self._sample_rank_three_tensor()
        tensor = TensOps(
            data,
            domain_dim=1,
            codomain_dim=[2, 2, 2],
            order=[1, 2],
            contravariant_indices=[0],
            covariant_indices=[1, 2],
        )

        result = antisymmetrize(tensor, [0, 1], "covariant")
        expected = antisymmetrization_covariant(tensor, sym_indices=[0, 1])

        self.assertTrue(backend.allclose(result.values, expected.values))

    def test_levi_civita_tensor_rejects_contravariant_as_metric(self) -> None:
        """The 'metric' argument must be the covariant metric, order [0, 2]."""

        n_samples, discretization = 2, [2, 2, 2]
        metric = _diagonal_metric([1.0, 1.0, 1.0], n_samples, discretization)
        inv_metric = inverse(metric)
        symbol = levi_civita_symbol(n_samples, 3, discretization)

        with self.assertRaisesRegex(ValueError, r"'metric'.*order \[0, 2\]"):
            levi_civita_tensor(
                symbol, 3, [3, 3, 3], [0, 3], metric=inv_metric)

    def test_levi_civita_tensor_rejects_covariant_as_inv_metric(self) -> None:
        """The 'inv_metric' argument must be the inverse metric, order [2, 0]."""

        n_samples, discretization = 2, [2, 2, 2]
        metric = _diagonal_metric([1.0, 1.0, 1.0], n_samples, discretization)
        symbol = levi_civita_symbol(n_samples, 3, discretization)

        with self.assertRaisesRegex(ValueError, r"'inv_metric'.*\[2, 0\]"):
            levi_civita_tensor(
                symbol, 3, [3, 3, 3], [3, 0], inv_metric=metric)

    def test_lower_index_invalid_index_raises(self) -> None:
        """Test that lowering without contravariant indices raises an error."""
        with self.assertRaisesRegex(
            ValueError,
            "Index can't be lowered because all indices are already covariant.",
        ):
            lower_index(EUCLIDEAN_METRIC, phi)

    def test_raise_index_invalid_index_raises(self) -> None:
        """Test that raising without covariant indices raises an error."""
        with self.assertRaisesRegex(
            ValueError,
            "Index can't be raised because all indices are already contravariant.",
        ):
            raise_index(EUCLIDEAN_INV_METRIC, u)


if __name__ == "__main__":
    unittest.main()
