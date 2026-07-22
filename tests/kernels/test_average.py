"""Tests for the Average kernel generator."""

import unittest

from tensopsciml.backends import get_backend
from tensopsciml.kernels.average import Average


class TestAverage(unittest.TestCase):
    """Test suite for Average kernel computation."""

    def setUp(self) -> None:
        """Set up shared backend fixture."""
        self.backend = get_backend()

    def test_average_kernel_initialization(self) -> None:
        """Test that Average initializes with default attributes."""
        avg = Average()
        self.assertEqual(avg.order, "1")
        self.assertEqual(avg.indexing, "ij")
        self.assertEqual(avg.device, "cpu")

    def test_average_kernel_invalid_order(self) -> None:
        """Test that invalid order raises ValueError."""
        with self.assertRaises(ValueError):
            Average(order="2")

    def test_average_kernel_invalid_indexing(self) -> None:
        """Test that invalid indexing raises ValueError."""
        with self.assertRaises(ValueError):
            Average(indexing="xyz")

    def test_average_kernel_first_order_1d(self) -> None:
        """Test the shape of the 1D averaging kernel."""
        avg = Average(order="1")
        kernel = avg.one_dimension()

        self.assertEqual(kernel.shape, (1, 1, 1, 2))

    def test_average_kernel_first_order_2d(self) -> None:
        """Test the shapes of 2D averaging kernels."""
        avg = Average(order="1", indexing="ij")
        kernels = avg.two_dimensions()

        self.assertEqual(len(kernels), 2)
        self.assertEqual(kernels[0].shape, (1, 1, 2, 1))
        self.assertEqual(kernels[1].shape, (1, 1, 1, 2))

    def test_average_kernel_first_order_3d(self) -> None:
        """Test the shapes of 3D averaging kernels."""
        avg = Average(order="1", indexing="ij")
        kernels = avg.three_dimensions()

        self.assertEqual(len(kernels), 3)
        self.assertEqual(kernels[0].shape, (1, 1, 2, 1, 1))
        self.assertEqual(kernels[1].shape, (1, 1, 1, 2, 1))
        self.assertEqual(kernels[2].shape, (1, 1, 1, 1, 2))

    def test_average_kernel_indexing_xy(self) -> None:
        """Test kernel counts for different indexing conventions."""
        avg_ij = Average(indexing="ij")
        avg_xy = Average(indexing="xy")

        kernels_ij = avg_ij.two_dimensions()
        kernels_xy = avg_xy.two_dimensions()

        self.assertEqual(len(kernels_ij), 2)
        self.assertEqual(len(kernels_xy), 2)

    def test_average_kernel_indexing_xy_shapes_are_swapped(self) -> None:
        """Test that 'xy' 2D kernels are the 'ij' ones in reversed order."""
        kernels_ij = Average(indexing="ij").two_dimensions()
        kernels_xy = Average(indexing="xy").two_dimensions()

        self.assertEqual(kernels_xy[0].shape, kernels_ij[1].shape)
        self.assertEqual(kernels_xy[1].shape, kernels_ij[0].shape)

    def test_average_kernel_first_order_3d_xy(self) -> None:
        """Test the shapes of 3D averaging kernels under 'xy' indexing."""
        avg = Average(order="1", indexing="xy")
        kernels = avg.three_dimensions()

        self.assertEqual(len(kernels), 3)
        self.assertEqual(kernels[0].shape, (1, 1, 1, 2, 1))
        self.assertEqual(kernels[1].shape, (1, 1, 2, 1, 1))
        self.assertEqual(kernels[2].shape, (1, 1, 1, 1, 2))

    def test_average_kernel_explicit_backend_override(self) -> None:
        """Test that an explicitly passed backend instance is used as-is."""
        explicit_backend = get_backend()
        avg = Average(backend=explicit_backend)
        self.assertIs(avg.backend, explicit_backend)

    def test_average_kernel_dtype_consistency(self) -> None:
        """Test that averaging kernels preserve requested dtype."""
        avg_f32 = Average(dtype=self.backend.float32)
        kernel_f32 = avg_f32.one_dimension()
        self.assertEqual(kernel_f32.dtype, self.backend.float32)

        avg_f64 = Average(dtype=self.backend.float64)
        kernel_f64 = avg_f64.one_dimension()
        self.assertEqual(kernel_f64.dtype, self.backend.float64)

    def test_average_kernel_device(self) -> None:
        """Test that averaging kernels use the requested device."""
        avg = Average(device="cpu")
        kernel = avg.one_dimension()
        self.assertEqual(str(kernel.device), "cpu")

    def test_average_kernel_values_1d(self) -> None:
        """Test normalized values for the 1D averaging kernel."""
        avg = Average()
        kernel = avg.one_dimension()
        expected = self.backend.tensor([[[[ 0.5, 0.5]]]], dtype=kernel.dtype)

        self.assertTrue(self.backend.allclose(kernel, expected, atol=1e-5))

    def test_average_kernel_values_2d(self) -> None:
        """Test normalized values for the 2D averaging kernels."""
        avg = Average()
        kernels = avg.two_dimensions()

        kernel_m2 = kernels[0]
        expected_m2 = self.backend.tensor([[[[0.5, 0.5]]]], dtype=kernel_m2.dtype)
        self.assertTrue(self.backend.allclose(kernel_m2, expected_m2, atol=1e-5))

        kernel_m1 = kernels[1]
        expected_m1 = self.backend.tensor([[[[0.5], [0.5]]]], dtype=kernel_m1.dtype)
        self.assertTrue(self.backend.allclose(kernel_m1, expected_m1, atol=1e-5))

    def test_average_kernel_values_3d(self) -> None:
        """Test normalized values for the 3D averaging kernels."""
        avg = Average()
        kernels = avg.three_dimensions()

        self.assertEqual(len(kernels), 3)

        expected_k1 = self.backend.tensor([[[[[0.5, 0.5]]]]], dtype=kernels[0].dtype)
        self.assertTrue(self.backend.allclose(kernels[0], expected_k1, atol=1e-5))

        expected_k2 = self.backend.tensor([[[[[0.5], [0.5]]]]], dtype=kernels[1].dtype)
        self.assertTrue(self.backend.allclose(kernels[1], expected_k2, atol=1e-5))

        expected_k3 = self.backend.tensor([[[[[0.5]], [[0.5]]]]], dtype=kernels[2].dtype)
        self.assertTrue(self.backend.allclose(kernels[2], expected_k3, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
