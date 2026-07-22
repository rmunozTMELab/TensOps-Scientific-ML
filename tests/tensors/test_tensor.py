"""Tests for the TensOps tensor class.

This module contains comprehensive unit tests for the TensOps class, which
wraps backend tensors with mathematical context for tensor calculus operations.
Tests cover initialization, validation, properties, dimension handling, index
variance, and consistency checks using fixtures from conftest.py.

Classes:
    TestTensOpsInitialization: Tests for TensOps initialization.
    TestTensOpsScalarFields: Tests for scalar field tensors.
    TestTensOpsVectorFields: Tests for vector field tensors.
    TestTensOpsMatrixFields: Tests for second-order tensor fields.
    TestTensOpsValidation: Tests for parameter and shape validation.
    TestTensOpsProperties: Tests for TensOps properties and consistency.
"""

import unittest

import numpy as np
import torch

from tests.conftest import phi, psi, u, v, w, x, A, B, C, backend
from tensopsciml.tensors import TensOps


class TestTensOpsInitialization(unittest.TestCase):
    """Test suite for TensOps tensor initialization and basic properties."""

    def setUp(self) -> None:
        """Set up test fixtures before each test method."""
        self.backend = backend

        self.mock_scalar_1d = torch.randn(20, 1, 10)
        self.mock_scalar_2d = torch.randn(20, 1, 10, 10)
        self.mock_scalar_3d = torch.randn(20, 1, 10, 10, 10)

        self.mock_vector_1d = torch.randn(20, 3, 10)
        self.mock_vector_2d = torch.randn(20, 3, 10, 10)
        self.mock_vector_3d = torch.randn(20, 3, 10, 10, 10)

        self.mock_matrix_1d = torch.randn(20, 3, 3, 10)
        self.mock_matrix_2d = torch.randn(20, 3, 3, 10, 10)
        self.mock_matrix_3d = torch.randn(20, 3, 3, 10, 10, 10)

    def test_tensops_mock_scalar_1d_initialization(self) -> None:
        """Test initialization for a scalar field in one dimension."""
        mock_tensops = TensOps(
            self.mock_scalar_1d,
            domain_dim=1,
            codomain_dim=[1],
            order=[0, 0],
            contravariant_indices=None,
            covariant_indices=None
        )
        
        self.assertEqual(mock_tensops.domain, 1)
        self.assertEqual(mock_tensops.codomain, [1])
        self.assertEqual(mock_tensops.order, [0, 0])
        self.assertIsNone(mock_tensops.contravariant)
        self.assertIsNone(mock_tensops.covariant)

    def test_tensops_mock_scalar_2d_initialization(self) -> None:
        """Test initialization for a scalar field in two dimensions."""
        mock_tensops = TensOps(
            self.mock_scalar_2d,
            domain_dim=2,
            codomain_dim=[1],
            order=[0, 0],
            contravariant_indices=None,
            covariant_indices=None
        )
        
        self.assertEqual(mock_tensops.domain, 2)
        self.assertEqual(mock_tensops.codomain, [1])
        self.assertEqual(mock_tensops.order, [0, 0])
        self.assertIsNone(mock_tensops.contravariant)
        self.assertIsNone(mock_tensops.covariant)

    def test_tensops_mock_scalar_3d_initialization(self) -> None:
        """Test initialization for a scalar field in three dimensions."""
        mock_tensops = TensOps(
            self.mock_scalar_3d,
            domain_dim=3,
            codomain_dim=[1],
            order=[0, 0],
            contravariant_indices=None,
            covariant_indices=None
        )
        
        self.assertEqual(mock_tensops.domain, 3)
        self.assertEqual(mock_tensops.codomain, [1])
        self.assertEqual(mock_tensops.order, [0, 0])
        self.assertIsNone(mock_tensops.contravariant)
        self.assertIsNone(mock_tensops.covariant)

    def test_tensops_mock_vector_1d_initialization(self) -> None:
        """Test initialization for a vector field in one dimension."""
        mock_tensops = TensOps(
            self.mock_vector_1d,
            domain_dim=1,
            codomain_dim=[3],
            order=[1, 0],
            contravariant_indices=[0],
            covariant_indices=None
        )
        
        self.assertEqual(mock_tensops.domain, 1)
        self.assertEqual(mock_tensops.codomain, [3])
        self.assertEqual(mock_tensops.order, [1, 0])
        self.assertEqual(mock_tensops.contravariant, [0])
        self.assertIsNone(mock_tensops.covariant)

    def test_tensops_mock_vector_2d_initialization(self) -> None:
        """Test initialization for a vector field in two dimensions."""
        mock_tensops = TensOps(
            self.mock_vector_2d,
            domain_dim=2,
            codomain_dim=[3],
            order=[1, 0],
            contravariant_indices=[0],
            covariant_indices=None
        )
        
        self.assertEqual(mock_tensops.domain, 2)
        self.assertEqual(mock_tensops.codomain, [3])
        self.assertEqual(mock_tensops.order, [1, 0])
        self.assertEqual(mock_tensops.contravariant, [0])
        self.assertIsNone(mock_tensops.covariant)

    def test_tensops_mock_vector_3d_initialization(self) -> None:
        """Test initialization for a vector field in three dimensions."""
        mock_tensops = TensOps(
            self.mock_vector_3d,
            domain_dim=3,
            codomain_dim=[3],
            order=[1, 0],
            contravariant_indices=[0],
            covariant_indices=None
        )
        
        self.assertEqual(mock_tensops.domain, 3)
        self.assertEqual(mock_tensops.codomain, [3])
        self.assertEqual(mock_tensops.order, [1, 0])
        self.assertEqual(mock_tensops.contravariant, [0])
        self.assertIsNone(mock_tensops.covariant)

    def test_tensops_mock_covector_1d_initialization(self) -> None:
        """Test initialization for a covector field in one dimension."""
        mock_tensops = TensOps(
            self.mock_vector_1d,
            domain_dim=1,
            codomain_dim=[3],
            order=[0, 1],
            contravariant_indices=None,
            covariant_indices=[0]
        )
        
        self.assertEqual(mock_tensops.domain, 1)
        self.assertEqual(mock_tensops.codomain, [3])
        self.assertEqual(mock_tensops.order, [0, 1])
        self.assertIsNone(mock_tensops.contravariant)
        self.assertEqual(mock_tensops.covariant, [0])

    def test_tensops_mock_covector_2d_initialization(self) -> None:
        """Test initialization for a covector field in two dimensions."""
        mock_tensops = TensOps(
            self.mock_vector_2d,
            domain_dim=2,
            codomain_dim=[3],
            order=[0, 1],
            contravariant_indices=None,
            covariant_indices=[0]
        )
        
        self.assertEqual(mock_tensops.domain, 2)
        self.assertEqual(mock_tensops.codomain, [3])
        self.assertEqual(mock_tensops.order, [0, 1])
        self.assertIsNone(mock_tensops.contravariant)
        self.assertEqual(mock_tensops.covariant, [0])

    def test_tensops_mock_covector_3d_initialization(self) -> None:
        """Test initialization for a covector field in three dimensions."""
        mock_tensops = TensOps(
            self.mock_vector_3d,
            domain_dim=3,
            codomain_dim=[3],
            order=[0, 1],
            contravariant_indices=None,
            covariant_indices=[0]
        )
        
        self.assertEqual(mock_tensops.domain, 3)
        self.assertEqual(mock_tensops.codomain, [3])
        self.assertEqual(mock_tensops.order, [0, 1])
        self.assertIsNone(mock_tensops.contravariant)
        self.assertEqual(mock_tensops.covariant, [0])

    def test_tensops_mock_matrix_1d_initialization(self) -> None:
        """Test initialization for a matrix field in one dimension."""
        mock_tensops = TensOps(
            self.mock_matrix_1d,
            domain_dim=1,
            codomain_dim=[3, 3],
            order=[1, 1],
            contravariant_indices=[0],
            covariant_indices=[1]
        )
        
        self.assertEqual(mock_tensops.domain, 1)
        self.assertEqual(mock_tensops.codomain, [3, 3])
        self.assertEqual(mock_tensops.order, [1, 1])
        self.assertEqual(mock_tensops.contravariant, [0])
        self.assertEqual(mock_tensops.covariant, [1])

    def test_tensops_mock_matrix_2d_initialization(self) -> None:
        """Test initialization for a matrix field in two dimensions."""
        mock_tensops = TensOps(
            self.mock_matrix_2d,
            domain_dim=2,
            codomain_dim=[3, 3],
            order=[1, 1],
            contravariant_indices=[0],
            covariant_indices=[1]
        )
        
        self.assertEqual(mock_tensops.domain, 2)
        self.assertEqual(mock_tensops.codomain, [3, 3])
        self.assertEqual(mock_tensops.order, [1, 1])
        self.assertEqual(mock_tensops.contravariant, [0])
        self.assertEqual(mock_tensops.covariant, [1])

    def test_tensops_mock_matrix_3d_initialization(self) -> None:
        """Test initialization for a matrix field in three dimensions."""
        mock_tensops = TensOps(
            self.mock_matrix_3d,
            domain_dim=3,
            codomain_dim=[3, 3],
            order=[1, 1],
            contravariant_indices=[0],
            covariant_indices=[1]
        )
        
        self.assertEqual(mock_tensops.domain, 3)
        self.assertEqual(mock_tensops.codomain, [3, 3])
        self.assertEqual(mock_tensops.order, [1, 1])
        self.assertEqual(mock_tensops.contravariant, [0])
        self.assertEqual(mock_tensops.covariant, [1])

    def test_tensops_values_access(self) -> None:
        """Test accessing the underlying tensor values."""
        mock_tensops = TensOps(
            self.mock_scalar_1d,
            domain_dim=1,
            codomain_dim=[1],
            order=[0, 0],
            contravariant_indices=None,
            covariant_indices=None
        )
        
        self.assertIsNotNone(mock_tensops.values)
        self.assertEqual(mock_tensops.values.shape, self.mock_scalar_1d.shape)
        self.assertTrue(self.backend.equal(mock_tensops.values, self.mock_scalar_1d))

    def test_tensops_base_property(self) -> None:
        """Test the base property for representative tensor types."""
        scalar = TensOps(
            self.mock_scalar_1d,
            domain_dim=1,
            codomain_dim=[1],
            order=[0, 0],
            contravariant_indices=None,
            covariant_indices=None
        )
        self.assertIsNone(scalar.base)
        
        vector = TensOps(
            self.mock_vector_1d,
            domain_dim=1,
            codomain_dim=[3],
            order=[1, 0],
            contravariant_indices=[0],
            covariant_indices=None
        )
        self.assertEqual(vector.base, "V")
        
        matrix = TensOps(
            self.mock_matrix_1d,
            domain_dim=1,
            codomain_dim=[3, 3],
            order=[1, 1],
            contravariant_indices=[0],
            covariant_indices=[1]
        )
        self.assertEqual(matrix.base, "V x V*")

        matrix = TensOps(
            self.mock_matrix_1d,
            domain_dim=1,
            codomain_dim=[3, 3],
            order=[1, 1],
            contravariant_indices=[1],
            covariant_indices=[0]
        )
        self.assertEqual(matrix.base, "V* x V")

    def test_tensops_dtype_preservation(self) -> None:
        """Test that TensOps preserves the dtype of the input data."""
        data_float64 = self.mock_scalar_1d.double()
        tensops = TensOps(
            data_float64,
            domain_dim=1,
            codomain_dim=[1],
            order=[0, 0],
            contravariant_indices=None,
            covariant_indices=None
        )
        self.assertEqual(tensops.values.dtype, data_float64.dtype)

    def test_tensops_device_preservation(self) -> None:
        """Test that TensOps preserves the device of the input data."""
        tensops = TensOps(
            self.mock_scalar_1d,
            domain_dim=1,
            codomain_dim=[1],
            order=[0, 0],
            contravariant_indices=None,
            covariant_indices=None
        )
        self.assertEqual(tensops.values.device, self.mock_scalar_1d.device)


class TestTensOpsScalarFields(unittest.TestCase):
    """Test suite for scalar field tensors using conftest fixtures."""

    def setUp(self) -> None:
        """Set up test fixtures before each test method."""
        pass

    def test_phi_scalar_properties(self) -> None:
        """Test scalar field properties for phi."""
        self.assertEqual(phi.domain, 3)
        self.assertEqual(phi.codomain, [1])
        self.assertEqual(phi.order, [0, 0])
        self.assertIsNone(phi.contravariant)
        self.assertIsNone(phi.covariant)

    def test_psi_scalar_properties(self) -> None:
        """Test scalar field properties for psi."""
        self.assertEqual(psi.domain, 3)
        self.assertEqual(psi.codomain, [1])
        self.assertEqual(psi.order, [0, 0])
        self.assertIsNone(psi.contravariant)
        self.assertIsNone(psi.covariant)

    def test_phi_base_none(self) -> None:
        """Test that scalar fields have no base representation."""
        self.assertIsNone(phi.base)

    def test_psi_base_none(self) -> None:
        """Test that scalar fields have no base representation."""
        self.assertIsNone(psi.base)

    def test_phi_values_batch_size(self) -> None:
        """Test the batch size of phi values."""
        self.assertEqual(phi.values.shape[0], 10)

    def test_psi_values_batch_size(self) -> None:
        """Test the batch size of psi values."""
        self.assertEqual(psi.values.shape[0], 10)


class TestTensOpsVectorFields(unittest.TestCase):
    """Test suite for vector field tensors using conftest fixtures."""

    def setUp(self) -> None:
        """Set up test fixtures before each test method."""
        pass

    def test_vector_field_order(self) -> None:
        """Test tensorial order for vector fields."""
        for vector in [u, v, w, x]:
            self.assertEqual(vector.order, [1, 0])
            self.assertEqual(vector.contravariant, [0])
            self.assertIsNone(vector.covariant)

    def test_vector_field_codomain(self) -> None:
        """Test codomain dimensions for vector fields."""
        for vector in [u, v, w, x]:
            self.assertEqual(vector.codomain, [3])

    def test_vector_field_domain(self) -> None:
        """Test domain dimensions for vector fields."""
        for vector in [u, v, w, x]:
            self.assertEqual(vector.domain, 3)

    def test_vector_base_v(self) -> None:
        """Test that vector fields have base 'V'."""
        for vector in [u, v, w, x]:
            self.assertEqual(vector.base, "V")


class TestTensOpsMatrixFields(unittest.TestCase):
    """Test suite for second-order tensor fields using conftest fixtures."""

    def setUp(self) -> None:
        """Set up test fixtures before each test method."""
        pass

    def test_matrix_field_order(self) -> None:
        """Test tensorial order for second-order tensor fields."""
        for matrix in [A, B, C]:
            self.assertEqual(matrix.order, [1, 1])
            self.assertEqual(matrix.contravariant, [0])
            self.assertEqual(matrix.covariant, [1])

    def test_matrix_field_codomain(self) -> None:
        """Test codomain dimensions for matrix fields."""
        for matrix in [A, B, C]:
            self.assertEqual(matrix.codomain, [3, 3])

    def test_matrix_field_domain(self) -> None:
        """Test domain dimensions for matrix fields."""
        for matrix in [A, B, C]:
            self.assertEqual(matrix.domain, 3)

    def test_matrix_base_v_vstar(self) -> None:
        """Test that matrix fields have base 'V x V*'."""
        for matrix in [A, B, C]:
            self.assertEqual(matrix.base, "V x V*")


class TestTensOpsValidation(unittest.TestCase):
    """Test suite for TensOps validation and error handling."""

    def setUp(self) -> None:
        """Set up test fixtures before each test method."""
        
        self.mock_scalar_3d = torch.randn(10, 1, 10, 10, 10)
        self.mock_matrix_3d = torch.randn(10, 3, 3, 10, 10, 10)
        self.mock_tensor_3d = torch.randn(10, 3, 3, 3, 10, 10, 10)

    def test_invalid_domain_dimension(self) -> None:
        """Test that invalid domain dimensions raise ValueError."""
        with self.assertRaises(ValueError):
            TensOps(
                self.mock_scalar_3d,
                domain_dim=4,
                codomain_dim=[3, 3],
                order=[1, 1],
                contravariant_indices=[0],
                covariant_indices=[1]
            )

    def test_invalid_domain_zero(self) -> None:
        """Test that domain dimension 0 raises ValueError."""
        with self.assertRaises(ValueError):
            TensOps(
                self.mock_scalar_3d,
                domain_dim=0,
                codomain_dim=[1],
                order=[0, 0],
                contravariant_indices=None,
                covariant_indices=None
            )

    def test_invalid_empty_codomain(self) -> None:
        """Test that an empty codomain raises ValueError."""
        with self.assertRaises(ValueError):
            TensOps(
                torch.randn(10, 10, 10, 10),
                domain_dim=3,
                codomain_dim=[],
                order=[1, 0],
                contravariant_indices=[0],
                covariant_indices=None
            )

    def test_mismatched_codomain_and_order(self) -> None:
        """Test that codomain dimensions must match tensor order."""
        with self.assertRaises(ValueError):
            TensOps(
                self.mock_scalar_3d,
                domain_dim=3,
                codomain_dim=[3, 3, 3],
                order=[1, 1],
                contravariant_indices=[0],
                covariant_indices=[1]
            )

    def test_mismatched_indices_and_order(self) -> None:
        """Test that indices must match tensor order."""
        with self.assertRaises(ValueError):
            TensOps(
                self.mock_scalar_3d,
                domain_dim=3,
                codomain_dim=[3],
                order=[1, 0],
                contravariant_indices=None,
                covariant_indices=None
            )

    def test_repeated_indices(self) -> None:
        """Test that contravariant and covariant indices cannot repeat."""
        with self.assertRaises(ValueError):
            TensOps(
                self.mock_tensor_3d,
                domain_dim=3,
                codomain_dim=[3, 3, 3],
                order=[2, 1],
                contravariant_indices=[0, 1],
                covariant_indices=[0]
            )

    def test_invalid_index_position(self) -> None:
        """Test that index positions must be within tensor order."""
        with self.assertRaises(ValueError):
            TensOps(
                self.mock_matrix_3d,
                domain_dim=3,
                codomain_dim=[3, 3],
                order=[1, 1],
                contravariant_indices=[2],
                covariant_indices=[1]
            )

    def test_invalid_shape_for_data(self) -> None:
        """Test that tensor shape must match declared dimensions."""
        with self.assertRaises(ValueError):
            TensOps(
                self.mock_matrix_3d,
                domain_dim=3,
                codomain_dim=[2, 3],
                order=[1, 1],
                contravariant_indices=[0],
                covariant_indices=[1]
            )

    def test_invalid_parameter_type(self) -> None:
        """Test that invalid parameter types raise ValueError."""
        with self.assertRaises(ValueError):
            TensOps(
                self.mock_matrix_3d,
                domain_dim="three",
                codomain_dim=[3, 3],
                order=[1, 1],
                contravariant_indices=[0],
                covariant_indices=[1]
            )

    def test_non_tensor_data(self) -> None:
        """Test that data must be convertible to a backend tensor."""
        with self.assertRaises(ValueError):
            TensOps(
                "not a tensor",
                domain_dim=3,
                codomain_dim=[3, 3],
                order=[1, 1],
                contravariant_indices=[0],
                covariant_indices=[1]
            )

    def test_unit_codomain_allows_nonzero_order(self) -> None:
        """A codomain of [1] with nonzero order is a 1D vector/covector."""
        vector_1d = TensOps(
            torch.randn(10, 1, 10),
            domain_dim=1,
            codomain_dim=[1],
            order=[1, 0],
            contravariant_indices=[0],
            covariant_indices=None,
        )
        self.assertEqual(vector_1d.order, [1, 0])
        self.assertFalse(vector_1d.is_scalar)

        covector_1d = TensOps(
            torch.randn(10, 1, 10),
            domain_dim=1,
            codomain_dim=[1],
            order=[0, 1],
            contravariant_indices=None,
            covariant_indices=[0],
        )
        self.assertEqual(covector_1d.order, [0, 1])
        self.assertFalse(covector_1d.is_scalar)

    def test_zero_order_is_scalar(self) -> None:
        """A zero-order field is scalar even though it carries a [1] axis."""
        scalar = TensOps(
            torch.randn(10, 1, 10),
            domain_dim=1,
            codomain_dim=[1],
            order=[0, 0],
            contravariant_indices=None,
            covariant_indices=None,
        )
        self.assertTrue(scalar.is_scalar)

    def test_non_integer_list_elements_raise(self) -> None:
        """List parameters must hold integers, not floats or strings."""
        with self.assertRaisesRegex(ValueError, "must be integers"):
            TensOps(
                self.mock_matrix_3d,
                domain_dim=3,
                codomain_dim=[3, 3.0],
                order=[1, 1],
                contravariant_indices=[0],
                covariant_indices=[1],
            )

    def test_index_count_must_match_declared_order(self) -> None:
        """Declaring order [1, 0] with no indices is inconsistent.

        Distinct from a shape mismatch: here the shape agrees with the codomain,
        so the failure is purely the index/order bookkeeping.
        """
        with self.assertRaisesRegex(ValueError, "must match the declared tensorial order"):
            TensOps(
                torch.randn(10, 3, 10, 10, 10),
                domain_dim=3,
                codomain_dim=[3],
                order=[1, 0],
                contravariant_indices=None,
                covariant_indices=None,
            )

    def test_invalid_covariant_index_position(self) -> None:
        """Covariant index positions must fall within the tensorial order."""
        with self.assertRaisesRegex(ValueError, "Covariant index positions"):
            TensOps(
                torch.randn(10, 3, 10, 10, 10),
                domain_dim=3,
                codomain_dim=[3],
                order=[0, 1],
                contravariant_indices=None,
                covariant_indices=[5],
            )

    def test_zero_order_requires_unit_codomain(self) -> None:
        """A scalar field (order 0) cannot declare a non-unit codomain."""
        with self.assertRaises(ValueError):
            TensOps(
                self.mock_matrix_3d[:, 0],
                domain_dim=3,
                codomain_dim=[3],
                order=[0, 0],
                contravariant_indices=None,
                covariant_indices=None,
            )


class TestTensOpsArithmetic(unittest.TestCase):
    """Test suite covering arithmetic, comparisons, and utility helpers."""

    def setUp(self) -> None:
        self.backend = backend
        self.dtype = backend.float32
        self.scalar_values = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]], dtype=self.dtype)
        self.vector_values = torch.tensor(
            [[[1.0, 0.0, -1.0, -2.0], [2.0, 2.0, 2.0, 2.0]]], dtype=self.dtype
        )

    @staticmethod
    def _domain_dim(values: torch.Tensor) -> int:
        return values.dim() - 2

    def _scalar(self, values: torch.Tensor) -> TensOps:
        return TensOps(
            values.clone(),
            domain_dim=self._domain_dim(values),
            codomain_dim=[1],
            order=[0, 0],
            contravariant_indices=None,
            covariant_indices=None,
        )

    def _vector(self, values: torch.Tensor, *, covariant: bool = False) -> TensOps:
        order = [1, 0] if not covariant else [0, 1]
        contra = [0] if not covariant else None
        cova = None if not covariant else [0]
        return TensOps(
            values.clone(),
            domain_dim=self._domain_dim(values),
            codomain_dim=[values.shape[1]],
            order=order,
            contravariant_indices=contra,
            covariant_indices=cova,
        )

    def test_addition_success(self) -> None:
        lhs = self._vector(self.vector_values)
        rhs = self._vector(self.vector_values * 2)
        result = lhs + rhs
        expected = lhs.values + rhs.values

        self.assertTrue(self.backend.allclose(result.values, expected))
        self.assertEqual(result.contravariant, [0])

    def test_addition_type_error(self) -> None:
        tensor = self._vector(self.vector_values)
        with self.assertRaises(TypeError):
            _ = tensor + 3

    def test_addition_shape_mismatch(self) -> None:
        lhs = self._vector(self.vector_values)
        mismatch = self._vector(torch.ones(1, 2, 5, dtype=self.dtype))
        with self.assertRaises(ValueError):
            _ = lhs + mismatch

    def test_subtraction_success(self) -> None:
        lhs = self._vector(self.vector_values)
        rhs = self._vector(self.vector_values)
        result = lhs - rhs
        zeros = torch.zeros_like(lhs.values)

        self.assertTrue(self.backend.allclose(result.values, zeros))

    def test_subtraction_type_error(self) -> None:
        tensor = self._vector(self.vector_values)
        with self.assertRaises(TypeError):
            _ = tensor - 3

    def test_multiplication_batch_size_mismatch(self) -> None:
        """Two tensors must share the same number of samples to combine."""
        vector = self._vector(self.vector_values)
        other_batch = self._scalar(
            torch.ones(2, 1, 4, dtype=self.dtype))

        with self.assertRaisesRegex(ValueError, "same number of samples"):
            _ = vector * other_batch

    def test_subtraction_base_mismatch(self) -> None:
        lhs = self._vector(self.vector_values)
        covector = self._vector(self.vector_values, covariant=True)
        with self.assertRaises(ValueError):
            _ = lhs - covector

    def test_multiplication_with_number(self) -> None:
        tensor = self._vector(self.vector_values)
        result = tensor * 2.5
        expected = tensor.values * 2.5

        self.assertTrue(self.backend.allclose(result.values, expected))

    def test_multiplication_with_numpy_scalar(self) -> None:
        tensor = self._vector(self.vector_values)
        result = tensor * np.float32(2.5)
        expected = tensor.values * 2.5

        self.assertTrue(self.backend.allclose(result.values, expected))

    def test_multiplication_with_zero_dim_tensor(self) -> None:
        tensor = self._vector(self.vector_values)
        result = tensor * torch.tensor(2.5, dtype=self.dtype)
        expected = tensor.values * 2.5

        self.assertTrue(self.backend.allclose(result.values, expected))

    def test_multiplication_scalar_right(self) -> None:
        vector = self._vector(self.vector_values)
        scalar = self._scalar(self.scalar_values)
        result = vector * scalar
        expanded_scalar = scalar.values.repeat(1, vector.values.shape[1], 1)
        expected = vector.values * expanded_scalar

        self.assertTrue(self.backend.allclose(result.values, expected))
        self.assertEqual(result.order, [1, 0])

    def test_multiplication_scalar_left(self) -> None:
        scalar = self._scalar(self.scalar_values)
        vector = self._vector(self.vector_values)
        result = scalar * vector
        expanded_scalar = scalar.values.repeat(1, vector.values.shape[1], 1)
        expected = expanded_scalar * vector.values

        self.assertTrue(self.backend.allclose(result.values, expected))
        self.assertEqual(result.order, [1, 0])

    def test_multiplication_scalar_scalar(self) -> None:
        tensor_a = self._scalar(self.scalar_values)
        tensor_b = self._scalar(self.scalar_values + 1)
        result = tensor_a * tensor_b
        expected = tensor_a.values * tensor_b.values

        self.assertTrue(self.backend.allclose(result.values, expected))
        self.assertEqual(result.codomain, [1])

    def test_multiplication_invalid_operand(self) -> None:
        tensor = self._vector(self.vector_values)
        with self.assertRaises(ValueError):
            _ = tensor * "nan"

    def test_multiplication_domain_mismatch(self) -> None:
        vector = self._vector(self.vector_values)
        plane = self._scalar(torch.ones(1, 1, 4, 4, dtype=self.dtype))
        with self.assertRaises(ValueError):
            _ = vector * plane

    def test_multiplication_non_scalar_tensor(self) -> None:
        lhs = self._vector(self.vector_values)
        rhs = self._vector(self.vector_values * -1)
        with self.assertRaises(ValueError):
            _ = lhs * rhs

    def test_right_multiplication_numeric(self) -> None:
        tensor = self._vector(self.vector_values)
        result = 3 * tensor
        expected = tensor.values * 3

        self.assertTrue(self.backend.allclose(result.values, expected))

    def test_division_numeric(self) -> None:
        tensor = self._vector(self.vector_values)
        result = tensor / 2
        expected = tensor.values / 2

        self.assertTrue(self.backend.allclose(result.values, expected))

    def test_division_invalid_type(self) -> None:
        tensor = self._vector(self.vector_values)
        with self.assertRaises(TypeError):
            _ = tensor / tensor

    def test_division_by_zero(self) -> None:
        tensor = self._vector(self.vector_values)
        with self.assertRaises(ZeroDivisionError):
            _ = tensor / 0

    def test_division_by_numpy_scalar(self) -> None:
        tensor = self._vector(self.vector_values)
        result = tensor / np.float32(2.5)
        expected = tensor.values / 2.5

        self.assertTrue(self.backend.allclose(result.values, expected))

    def test_division_by_zero_dim_tensor(self) -> None:
        tensor = self._vector(self.vector_values)
        result = tensor / torch.tensor(2.5, dtype=self.dtype)
        expected = tensor.values / 2.5

        self.assertTrue(self.backend.allclose(result.values, expected))

    def test_division_by_scalar_field(self) -> None:
        vector = self._vector(self.vector_values)
        scalar = self._scalar(self.scalar_values)
        result = vector / scalar
        expanded_scalar = scalar.values.repeat(1, vector.values.shape[1], 1)
        expected = vector.values / expanded_scalar

        self.assertTrue(self.backend.allclose(result.values, expected))
        self.assertEqual(result.order, [1, 0])

    def test_division_scalar_by_scalar_field(self) -> None:
        tensor_a = self._scalar(self.scalar_values)
        tensor_b = self._scalar(self.scalar_values + 1)
        result = tensor_a / tensor_b
        expected = tensor_a.values / tensor_b.values

        self.assertTrue(self.backend.allclose(result.values, expected))
        self.assertEqual(result.codomain, [1])

    def test_negation(self) -> None:
        tensor = self._vector(self.vector_values)
        result = -tensor
        expected = -tensor.values

        self.assertTrue(self.backend.allclose(result.values, expected))

    def test_unary_plus(self) -> None:
        tensor = self._vector(self.vector_values)
        result = +tensor

        self.assertTrue(self.backend.allclose(result.values, tensor.values))

    def test_equality_true(self) -> None:
        reference = self._vector(self.vector_values)
        duplicate = self._vector(self.vector_values)
        self.assertTrue(reference == duplicate)

    def test_equality_false(self) -> None:
        reference = self._vector(self.vector_values)
        different = self._vector(self.vector_values + 1)
        self.assertFalse(reference == different)

    def test_equality_with_non_tensops_returns_false(self) -> None:
        """Comparing with non-TensOps returns False, not TypeError."""
        tensor = self._vector(self.vector_values)

        # Comparison with None should return False
        self.assertFalse(tensor == None)

        # Comparison with other types should return False
        self.assertFalse(tensor == 1)
        self.assertFalse(tensor == "string")
        self.assertFalse(tensor == [1, 2, 3])
        self.assertFalse(tensor == {"key": "value"})

    def test_hash_makes_tensops_hashable(self) -> None:
        """TensOps instances are hashable and can be used in sets and dicts."""
        tensor1 = self._vector(self.vector_values)
        tensor2 = self._vector(self.vector_values)

        # TensOps should be hashable
        hash1 = hash(tensor1)
        hash2 = hash(tensor2)
        self.assertIsInstance(hash1, int)
        self.assertIsInstance(hash2, int)

        # Each instance has a different hash (identity-based)
        self.assertNotEqual(hash1, hash2)

        # Can be used in sets
        tensor_set = {tensor1, tensor2}
        self.assertEqual(len(tensor_set), 2)

        # Can be used as dict keys
        tensor_dict = {tensor1: "value1", tensor2: "value2"}
        self.assertEqual(tensor_dict[tensor1], "value1")
        self.assertEqual(tensor_dict[tensor2], "value2")

    def test_equality_true_has_same_hash_by_identity(self) -> None:
        """Equal tensors have the same hash only if they're the same object."""
        tensor1 = self._vector(self.vector_values)
        tensor2 = tensor1  # Same object

        # Same object should have same hash
        self.assertEqual(hash(tensor1), hash(tensor2))

    def test_keys(self) -> None:
        tensor = self._vector(self.vector_values)
        self.assertEqual(tensor.keys(), ["values", "domain", "codomain", "contravariant", "covariant", "order"])

    def test_getitem_completes_mapping_protocol(self) -> None:
        """dict(T) works because keys() is backed by __getitem__."""
        tensor = self._vector(self.vector_values)
        as_dict = dict(tensor)

        self.assertEqual(list(as_dict.keys()), tensor.keys())
        self.assertTrue(self.backend.equal(as_dict["values"], tensor.values))
        self.assertEqual(as_dict["domain"], tensor.domain)
        self.assertEqual(as_dict["codomain"], tensor.codomain)
        self.assertEqual(as_dict["order"], tensor.order)
        self.assertEqual(as_dict["contravariant"], tensor.contravariant)
        self.assertEqual(as_dict["covariant"], tensor.covariant)

        with self.assertRaises(KeyError):
            _ = tensor["base"]

    def test_string_representation(self) -> None:
        tensor = self._vector(self.vector_values)
        description = str(tensor)

        self.assertIn("Shape", description)
        self.assertIn("Order", description)

    def test_requires_grad_sets_flag(self) -> None:
        tensor = self._vector(self.vector_values)
        tensor.requires_grad_()
        self.assertTrue(tensor.values.requires_grad)

    def test_requires_grad_on_non_leaf_values(self) -> None:
        """requires_grad_ tolerates non-leaf values (cloned from tracked data)."""
        tracked = self.vector_values.clone().requires_grad_(True)
        tensor = self._vector(tracked)  # __init__ clones -> values is non-leaf

        tensor.requires_grad_(True)
        self.assertTrue(tensor.values.requires_grad)

        tensor.requires_grad_(False)
        self.assertFalse(tensor.values.requires_grad)

    def test_to_dtype(self) -> None:
        tensor = self._vector(self.vector_values)
        converted = tensor.to(dtype=backend.float64)

        self.assertEqual(converted.values.dtype, backend.float64)
        self.assertTrue(self.backend.allclose(converted.values.to(self.dtype), tensor.values))


if __name__ == "__main__":
    unittest.main()