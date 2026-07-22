"""Tests for the Derivative kernel generator."""

import unittest

from tensopsciml.backends import get_backend
from tensopsciml.kernels.derivative import Derivative


class TestDerivative(unittest.TestCase):
    """Test suite for Derivative kernel computation."""

    def setUp(self) -> None:
        """Set up backend and default spacing."""
        self.backend = get_backend()
        self.d1 = 0.1
        self.d2 = 0.1
        self.d3 = 0.1

    def test_derivative_kernel_initialization(self) -> None:
        """Test that Derivative stores initialization parameters."""
        deriv_1d = Derivative(self.d1)
        self.assertEqual(deriv_1d.d1, self.d1)
        self.assertEqual(deriv_1d.order, "1")
        self.assertEqual(deriv_1d.indexing, "ij")

        deriv_2d = Derivative(self.d1, self.d2)
        self.assertEqual(deriv_2d.d1, self.d1)
        self.assertEqual(deriv_2d.d2, self.d2)

        deriv_3d = Derivative(self.d1, self.d2, self.d3)
        self.assertEqual(deriv_3d.d1, self.d1)
        self.assertEqual(deriv_3d.d2, self.d2)
        self.assertEqual(deriv_3d.d3, self.d3)

    def test_derivative_kernel_invalid_order(self) -> None:
        """Test that invalid derivative order raises ValueError."""
        with self.assertRaises(ValueError):
            Derivative(self.d1, order="2")

    def test_derivative_kernel_invalid_indexing(self) -> None:
        """Test that invalid indexing raises ValueError."""
        with self.assertRaises(ValueError):
            Derivative(self.d1, indexing="xyz")

    def test_derivative_kernel_first_order_1d(self) -> None:
        """Test the shape of the 1D derivative kernel."""
        deriv = Derivative(self.d1, order="1")
        kernel = deriv.one_dimension()

        self.assertEqual(kernel.shape, (1, 1, 1, 2))

    def test_derivative_kernel_first_order_2d(self) -> None:
        """Test the shape of the 2D derivative kernels."""
        deriv = Derivative(self.d1, self.d2, order="1", indexing="ij")
        kernels = deriv.two_dimensions()

        self.assertEqual(kernels.shape, (2, 1, 1, 2, 2))

    def test_derivative_kernel_first_order_3d(self) -> None:
        """Test the shape of the 3D derivative kernels."""
        deriv = Derivative(self.d1, self.d2, self.d3, order="1", indexing="ij")
        kernels = deriv.three_dimensions()

        self.assertEqual(kernels.shape, (3, 1, 1, 2, 2, 2))

    def test_derivative_kernel_2d_missing_d2(self) -> None:
        """Test that two_dimensions requires the second spacing."""
        deriv = Derivative(self.d1)
        with self.assertRaises(ValueError):
            deriv.two_dimensions()

    def test_derivative_kernel_3d_missing_d3(self) -> None:
        """Test that three_dimensions requires the third spacing."""
        deriv = Derivative(self.d1, self.d2)
        with self.assertRaises(ValueError):
            deriv.three_dimensions()

    def test_derivative_kernel_indexing_xy(self) -> None:
        """Test derivative shapes across indexing conventions."""
        deriv_ij = Derivative(self.d1, self.d2, indexing="ij")
        deriv_xy = Derivative(self.d1, self.d2, indexing="xy")

        kernels_ij = deriv_ij.two_dimensions()
        kernels_xy = deriv_xy.two_dimensions()

        self.assertEqual(kernels_ij.shape, kernels_xy.shape)

    def test_derivative_kernel_anisotropic_ij(self) -> None:
        """Test that 'ij' kernels use d1 on axis 0 and d2 on axis 1."""
        d1, d2 = 2.0, 5.0
        deriv = Derivative(d1, d2, indexing="ij")
        D = deriv.two_dimensions()

        field = self.backend.zeros((1, 1, 4, 4))
        for i in range(4):
            field[0, 0, i, :] = i

        conv2d = self.backend.conv2d
        d_axis0 = conv2d(field, D[0])
        d_axis1 = conv2d(field, D[1])

        self.assertTrue(self.backend.allclose(
            d_axis0, self.backend.zeros_like(d_axis0) + 1 / d1, atol=1e-6))
        self.assertTrue(self.backend.allclose(
            d_axis1, self.backend.zeros_like(d_axis1), atol=1e-6))

    def test_derivative_kernel_anisotropic_xy(self) -> None:
        """Test that 'xy' kernels use d1 on axis 1 and d2 on axis 0."""
        d1, d2 = 2.0, 5.0
        deriv = Derivative(d1, d2, indexing="xy")
        D = deriv.two_dimensions()

        field = self.backend.zeros((1, 1, 4, 4))
        for j in range(4):
            field[0, 0, :, j] = j

        conv2d = self.backend.conv2d
        d_axis0 = conv2d(field, D[0])
        d_axis1 = conv2d(field, D[1])

        self.assertTrue(self.backend.allclose(
            d_axis0, self.backend.zeros_like(d_axis0) + 1 / d1, atol=1e-6))
        self.assertTrue(self.backend.allclose(
            d_axis1, self.backend.zeros_like(d_axis1), atol=1e-6))

    def test_derivative_kernel_anisotropic_3d_ij(self) -> None:
        """Test that 'ij' 3D kernels use d1/d2/d3 on axes 0/1/2 respectively."""
        d1, d2, d3 = 2.0, 5.0, 9.0
        deriv = Derivative(d1, d2, d3, indexing="ij")
        D = deriv.three_dimensions()
        conv3d = self.backend.conv3d

        field_0 = self.backend.zeros((1, 1, 4, 4, 4))
        for i in range(4):
            field_0[0, 0, i, :, :] = i

        d0_on_field0 = conv3d(field_0, D[0])
        d1_on_field0 = conv3d(field_0, D[1])
        d2_on_field0 = conv3d(field_0, D[2])
        self.assertTrue(self.backend.allclose(
            d0_on_field0, self.backend.zeros_like(d0_on_field0) + 1 / d1, atol=1e-6))
        self.assertTrue(self.backend.allclose(
            d1_on_field0, self.backend.zeros_like(d1_on_field0), atol=1e-6))
        self.assertTrue(self.backend.allclose(
            d2_on_field0, self.backend.zeros_like(d2_on_field0), atol=1e-6))

        field_1 = self.backend.zeros((1, 1, 4, 4, 4))
        for j in range(4):
            field_1[0, 0, :, j, :] = j

        d1_on_field1 = conv3d(field_1, D[1])
        d0_on_field1 = conv3d(field_1, D[0])
        self.assertTrue(self.backend.allclose(
            d1_on_field1, self.backend.zeros_like(d1_on_field1) + 1 / d2, atol=1e-6))
        self.assertTrue(self.backend.allclose(
            d0_on_field1, self.backend.zeros_like(d0_on_field1), atol=1e-6))

        field_2 = self.backend.zeros((1, 1, 4, 4, 4))
        for k in range(4):
            field_2[0, 0, :, :, k] = k

        d2_on_field2 = conv3d(field_2, D[2])
        self.assertTrue(self.backend.allclose(
            d2_on_field2, self.backend.zeros_like(d2_on_field2) + 1 / d3, atol=1e-6))

    def test_derivative_kernel_anisotropic_3d_xy(self) -> None:
        """Test that 'xy' 3D kernels swap d1/d2 onto axes 1/0, leaving axis 2 (d3) unaffected."""
        d1, d2, d3 = 2.0, 5.0, 9.0
        deriv = Derivative(d1, d2, d3, indexing="xy")
        D = deriv.three_dimensions()
        conv3d = self.backend.conv3d

        field_1 = self.backend.zeros((1, 1, 4, 4, 4))
        for j in range(4):
            field_1[0, 0, :, j, :] = j

        d0_on_field1 = conv3d(field_1, D[0])
        d1_on_field1 = conv3d(field_1, D[1])
        self.assertTrue(self.backend.allclose(
            d0_on_field1, self.backend.zeros_like(d0_on_field1) + 1 / d1, atol=1e-6))
        self.assertTrue(self.backend.allclose(
            d1_on_field1, self.backend.zeros_like(d1_on_field1), atol=1e-6))

        field_0 = self.backend.zeros((1, 1, 4, 4, 4))
        for i in range(4):
            field_0[0, 0, i, :, :] = i

        d1_on_field0 = conv3d(field_0, D[1])
        d0_on_field0 = conv3d(field_0, D[0])
        self.assertTrue(self.backend.allclose(
            d1_on_field0, self.backend.zeros_like(d1_on_field0) + 1 / d2, atol=1e-6))
        self.assertTrue(self.backend.allclose(
            d0_on_field0, self.backend.zeros_like(d0_on_field0), atol=1e-6))

        field_2 = self.backend.zeros((1, 1, 4, 4, 4))
        for k in range(4):
            field_2[0, 0, :, :, k] = k

        d2_on_field2 = conv3d(field_2, D[2])
        self.assertTrue(self.backend.allclose(
            d2_on_field2, self.backend.zeros_like(d2_on_field2) + 1 / d3, atol=1e-6))

    def test_derivative_kernel_indexing_tampered_after_init_raises(self) -> None:
        """The per-method guard catches an indexing mutated after construction.

        The constructor validates `indexing`, but it stays a public attribute,
        so the kernel builders re-check it rather than silently emitting a
        stencil for the wrong axis.
        """
        deriv = Derivative(self.d1, self.d2, self.d3, indexing="ij")
        deriv.indexing = "zw"

        with self.assertRaisesRegex(ValueError, "indexing must be 'ij' or 'xy'"):
            deriv.two_dimensions()
        with self.assertRaisesRegex(ValueError, "indexing must be 'ij' or 'xy'"):
            deriv.three_dimensions()

    def test_derivative_kernel_explicit_backend_override(self) -> None:
        """Test that an explicitly passed backend instance is used as-is."""
        explicit_backend = get_backend()
        deriv = Derivative(self.d1, backend=explicit_backend)
        self.assertIs(deriv.backend, explicit_backend)

    def test_derivative_kernel_dtype_consistency(self) -> None:
        """Test that derivative kernels preserve requested dtype."""
        deriv_f32 = Derivative(self.d1, dtype=self.backend.float32)
        kernel_f32 = deriv_f32.one_dimension()
        self.assertEqual(kernel_f32.dtype, self.backend.float32)

        deriv_f64 = Derivative(self.d1, dtype=self.backend.float64)
        kernel_f64 = deriv_f64.one_dimension()
        self.assertEqual(kernel_f64.dtype, self.backend.float64)

    def test_derivative_kernel_device(self) -> None:
        """Test that derivative kernels use the requested device."""
        deriv = Derivative(self.d1, device="cpu")
        kernel = deriv.one_dimension()
        self.assertEqual(str(kernel.device), "cpu")

    def test_derivative_kernel_spacing_normalization(self) -> None:
        """Test normalization of derivative kernels by spacing."""
        dn = 0.1
        deriv = Derivative(dn, dn, dn)

        kernel_1d = deriv.one_dimension()
        expected_1d = self.backend.tensor([
            [[[-10.0, 10.0]]]
            ])
        self.assertTrue(self.backend.allclose(kernel_1d, expected_1d, atol=1e-3))

        kernel_2d = deriv.two_dimensions()
        expected_2d = self.backend.tensor([
            [[[[-5., -5.], [5., 5.]]]], [[[[-5., 5.], [-5., 5.]]]]
            ])
        self.assertTrue(self.backend.allclose(kernel_2d, expected_2d, atol=1e-3))
        
        kernel_3d = deriv.three_dimensions()
        expected_3d = self.backend.tensor([
            [[[[[-2.5, -2.5], [-2.5, -2.5]], [[ 2.5,  2.5], [ 2.5,  2.5]]]]],
            [[[[[-2.5, -2.5], [ 2.5,  2.5]], [[-2.5, -2.5], [ 2.5,  2.5]]]]],
            [[[[[-2.5,  2.5], [-2.5,  2.5]], [[-2.5,  2.5], [-2.5,  2.5]]]]]
            ])
        self.assertTrue(self.backend.allclose(kernel_3d, expected_3d, atol=1e-3))


if __name__ == "__main__":
    unittest.main()