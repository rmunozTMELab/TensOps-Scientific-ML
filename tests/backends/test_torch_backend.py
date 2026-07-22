"""Tests for the PyTorch backend implementation.

This module verifies that TorchBackend implements tensor operations, linear algebra,
and autograd features defined by the backend interface.
"""

import unittest

import numpy as np
import torch

from tensopsciml.backends.torch_backend import TorchBackend


class TestTorchBackend(unittest.TestCase):
    """Test suite for the PyTorch backend wrapper."""

    def setUp(self) -> None:
        """Initialize the backend instance before each test."""
        self.backend = TorchBackend()

    def test_initialization(self) -> None:
        """Verify that the backend initializes correctly."""
        self.assertIsNotNone(self.backend)

    def test_tensor_creation_from_list(self) -> None:
        """Verify tensor creation from a standard Python list."""
        data = [1.0, 2.0, 3.0]
        tensor = self.backend.tensor(data)

        self.assertIsInstance(tensor, torch.Tensor)
        self.assertEqual(tensor.shape, (3,))

    def test_tensor_creation_from_numpy(self) -> None:
        """Verify tensor creation from a NumPy array."""
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        tensor = self.backend.tensor(data)

        self.assertIsInstance(tensor, torch.Tensor)
        self.assertEqual(tensor.shape, (2, 2))

    def test_tensor_creation_with_dtype(self) -> None:
        """Verify tensor creation enforces the specified data type."""
        data = [1, 2, 3]
        tensor = self.backend.tensor(data, dtype=torch.float32)

        self.assertEqual(tensor.dtype, torch.float32)

    def test_tensor_creation_requires_grad(self) -> None:
        """Tensor factory enables gradient tracking when requested."""
        data = [1.0, 2.0]
        tensor = self.backend.tensor(data, requires_grad=True)

        self.assertTrue(tensor.requires_grad)

    def test_tensor_creation_on_device(self) -> None:
        """Verify tensor creation on a specific device."""
        data = [1.0, 2.0]
        tensor = self.backend.tensor(data, device="cpu")

        self.assertEqual(str(tensor.device), "cpu")

    def test_requires_grad_on_non_leaf_enable_is_noop(self) -> None:
        """Enabling tracking on a tracked non-leaf tensor returns it untouched."""
        leaf = torch.tensor([1.0, 2.0], requires_grad=True)
        non_leaf = leaf.clone()

        result = self.backend.requires_grad_(non_leaf, True)

        self.assertIs(result, non_leaf)
        self.assertTrue(result.requires_grad)

    def test_requires_grad_on_non_leaf_disable_detaches(self) -> None:
        """Disabling tracking on a non-leaf tensor detaches instead of raising."""
        leaf = torch.tensor([1.0, 2.0], requires_grad=True)
        non_leaf = leaf.clone()

        result = self.backend.requires_grad_(non_leaf, False)

        self.assertFalse(result.requires_grad)
        self.assertTrue(torch.equal(result, non_leaf))

    def test_clone_preserves_gradients(self) -> None:
        """Verify that cloning preserves data and gradient requirements."""
        tensor = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        cloned = self.backend.clone(tensor)

        self.assertTrue(torch.equal(tensor, cloned))
        self.assertTrue(cloned.requires_grad)

    def test_equal_comparison(self) -> None:
        """Verify equality checks for identical and different tensors."""
        # True case.
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([1.0, 2.0, 3.0])
        self.assertTrue(self.backend.equal(a, b))

        # False case.
        c = torch.tensor([1.0, 2.0, 4.0])
        self.assertFalse(self.backend.equal(a, c))

    def test_dtype_attributes(self) -> None:
        """The backend exposes the dtypes callers reference as backend.floatXX."""
        self.assertIs(self.backend.float32, torch.float32)
        self.assertIs(self.backend.float64, torch.float64)

    def test_determinant_2x2(self) -> None:
        """Verify determinant computation for a 2x2 matrix."""
        matrix = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        det = self.backend.det(matrix)

        # Determinant equals 1*4 - 2*3 = -2.
        self.assertAlmostEqual(det.item(), -2.0, places=5)

    def test_determinant_batched(self) -> None:
        """Determinant acts on the trailing d x d block of a stack of matrices.

        This is how the library always calls it: never on a bare matrix, but on
        a [..., d, d] stack produced by permuting a tensor field's codomain axes
        to the end (see the _matrix_view helper in tensors/metrics.py).
        """
        matrices = torch.tensor([
            [[1.0, 2.0], [3.0, 4.0]],   # det = -2
            [[2.0, 0.0], [0.0, 3.0]],   # det = 6
        ])

        determinants = self.backend.det(matrices)

        self.assertEqual(determinants.shape, (2,))
        self.assertTrue(torch.allclose(
            determinants, torch.tensor([-2.0, 6.0]), atol=1e-5))

    def test_matrix_inverse(self) -> None:
        """Verify that matrix inversion yields the identity product."""
        matrix = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        inv = self.backend.inv(matrix)

        product = torch.mm(matrix, inv)
        identity = torch.eye(2)

        self.assertTrue(torch.allclose(product, identity, atol=1e-5))

    def test_matrix_inverse_batched(self) -> None:
        """Inversion acts on the trailing d x d block of a stack of matrices."""
        matrices = torch.tensor([
            [[1.0, 2.0], [3.0, 4.0]],
            [[2.0, 0.0], [0.0, 4.0]],
        ])

        inverses = self.backend.inv(matrices)

        products = torch.matmul(matrices, inverses)
        identities = torch.eye(2).expand(2, 2, 2)

        self.assertEqual(inverses.shape, (2, 2, 2))
        self.assertTrue(torch.allclose(products, identities, atol=1e-5))

    def test_transpose_swaps_the_requested_dimensions(self) -> None:
        """Verify transpose swaps exactly the two dimensions it is given."""
        tensor = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)

        transposed = self.backend.transpose(tensor, (0, 1))

        self.assertEqual(transposed.shape, (3, 2, 4))
        self.assertTrue(torch.equal(transposed, tensor.permute(1, 0, 2)))

    def test_transpose_on_codomain_axes(self) -> None:
        """Transposing axes (1, 2) swaps a rank-2 field's two codomain indices.

        This is the call MetrOps' symmetry check and matrix_algebra.transpose
        rely on: values shaped [n_samples, d, d, discretization...].
        """
        values = torch.tensor([[[[1.0], [2.0]], [[3.0], [4.0]]]])  # [1, 2, 2, 1]

        transposed = self.backend.transpose(values, (1, 2))

        self.assertEqual(transposed.shape, values.shape)
        self.assertAlmostEqual(transposed[0, 0, 1, 0].item(), 3.0)
        self.assertAlmostEqual(transposed[0, 1, 0, 0].item(), 2.0)

    def test_eye_square(self) -> None:
        """Verify eye builds a square identity matrix."""
        identity = self.backend.eye(3)

        self.assertEqual(identity.shape, (3, 3))
        self.assertTrue(torch.equal(identity, torch.eye(3)))

    def test_eye_rectangular(self) -> None:
        """Verify eye builds a rectangular diagonal matrix when given m."""
        diagonal = self.backend.eye(2, 3)

        self.assertEqual(diagonal.shape, (2, 3))
        self.assertTrue(torch.equal(diagonal, torch.eye(2, 3)))

    def test_eye_honours_dtype_and_requires_grad(self) -> None:
        """Verify eye forwards dtype and gradient tracking to the backend."""
        identity = self.backend.eye(
            2, dtype=torch.float64, requires_grad=True)

        self.assertEqual(identity.dtype, torch.float64)
        self.assertTrue(identity.requires_grad)

    def test_ones_creation(self) -> None:
        """Verify creation of one-filled tensors."""
        ones = self.backend.ones((2, 3))

        self.assertEqual(ones.shape, (2, 3))
        self.assertTrue(torch.all(ones == 1))

    def test_ones_honours_dtype_and_requires_grad(self) -> None:
        """Verify ones forwards dtype and gradient tracking to the backend."""
        ones = self.backend.ones(
            (2,), dtype=torch.float64, requires_grad=True)

        self.assertEqual(ones.dtype, torch.float64)
        self.assertTrue(ones.requires_grad)

    def test_einsum_operations(self) -> None:
        """Verify einsum for matrix multiplication and trace."""
        # Matrix multiplication.
        a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        b = torch.tensor([[5.0, 6.0], [7.0, 8.0]])

        result_mm = self.backend.einsum("ij,jk->ik", a, b)
        expected_mm = torch.mm(a, b)
        self.assertTrue(torch.allclose(result_mm, expected_mm))

        # Trace computation.
        result_trace = self.backend.einsum("ii->", a)
        # Trace equals 1.0 + 4.0 = 5.0.
        self.assertAlmostEqual(result_trace.item(), 5.0, places=5)

    def test_permute_dimensions(self) -> None:
        """Verify tensor dimension permutation."""
        # Original shape (2, 3, 4).
        tensor = torch.arange(24).reshape(2, 3, 4)

        # Permute to shape (4, 2, 3) using indices [2, 0, 1].
        permuted = self.backend.permute(tensor, [2, 0, 1])
        self.assertEqual(permuted.shape, (4, 2, 3))

    def test_shape_manipulation_squeeze_unsqueeze(self) -> None:
        """Verify squeeze and unsqueeze operations."""
        # Unsqueeze along dimension 0.
        tensor = torch.tensor([1.0, 2.0, 3.0])
        unsqueezed = self.backend.unsqueeze(tensor, dim=0)
        self.assertEqual(unsqueezed.shape, (1, 3))

        # Squeeze all singleton dimensions.
        tensor_sq = torch.tensor([[[1.0, 2.0, 3.0]]])
        squeezed = self.backend.squeeze(tensor_sq)
        self.assertEqual(squeezed.shape, (3,))

        # Squeeze along dimension 2.
        tensor_sq_dim = torch.tensor([[[1.0], [2.0], [3.0]]])
        squeezed_dim = self.backend.squeeze(tensor_sq_dim, dim=2)
        self.assertEqual(squeezed_dim.shape, (1, 3))

    def test_stack_tensors(self) -> None:
        """Verify stacking tensors along a new dimension."""
        tensors = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]
        stacked = self.backend.stack(tensors, dim=0)

        self.assertEqual(stacked.shape, (2, 2))

    def test_zeros_creation(self) -> None:
        """Verify creation of zero-filled tensors."""
        # Create zeros via explicit shape.
        zeros = self.backend.zeros((2, 3))
        self.assertEqual(zeros.shape, (2, 3))
        self.assertTrue(torch.all(zeros == 0))

        # Create zeros using zeros_like.
        ref_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        zeros_like = self.backend.zeros_like(ref_tensor)
        self.assertEqual(zeros_like.shape, ref_tensor.shape)
        self.assertTrue(torch.all(zeros_like == 0))

    def test_zeros_honours_dtype_and_requires_grad(self) -> None:
        """Verify zeros forwards dtype and gradient tracking to the backend."""
        zeros = self.backend.zeros(
            (2,), dtype=torch.float64, requires_grad=True)

        self.assertEqual(zeros.dtype, torch.float64)
        self.assertTrue(zeros.requires_grad)

    def test_zeros_like_honours_dtype_override(self) -> None:
        """zeros_like takes its shape from the reference but can override dtype."""
        reference = torch.ones((2, 3), dtype=torch.float32)

        zeros = self.backend.zeros_like(reference, dtype=torch.float64)

        self.assertEqual(zeros.shape, reference.shape)
        self.assertEqual(zeros.dtype, torch.float64)
        self.assertTrue(torch.all(zeros == 0))

    def test_gradient_computation(self) -> None:
        """Verify automatic differentiation via the backend wrapper."""
        x = torch.tensor([2.0, 3.0], requires_grad=True)
        y = (x ** 2) + 3 * x

        # Analytical gradient: dy/dx = 2x + 3.
        # For x=2, gradient equals 7.
        # For x=3, gradient equals 9.

        grads = self.backend.autograd(y, x)

        expected_grads = torch.tensor([7.0, 9.0])
        self.assertTrue(torch.allclose(grads, expected_grads))

    def test_autograd_explicit_grad_outputs_selects_component(self) -> None:
        """An explicit grad_outputs mask isolates a single output component."""
        x = torch.tensor([2.0], requires_grad=True)
        y = torch.cat([x, x ** 2, x ** 3])

        grad_component1 = self.backend.autograd(
            y, x, grad_outputs=torch.tensor([0.0, 1.0, 0.0]), retain_graph=True)

        # d(x^2)/dx = 2x = 4.0 at x=2, isolated from the other components.
        self.assertTrue(torch.allclose(grad_component1, torch.tensor([4.0])))

    def test_autograd_allows_unused_inputs(self) -> None:
        """Autograd returns None for unused inputs when allowed."""
        used = torch.tensor([1.0], requires_grad=True)
        unused = torch.tensor([2.0], requires_grad=True)
        y = used ** 2

        gradient = self.backend.autograd(y, unused, allow_unused=True)

        self.assertIsNone(gradient)

    def test_autograd_create_graph_enables_second_derivatives(self) -> None:
        """create_graph keeps the gradient differentiable, enabling higher orders."""
        x = torch.tensor([2.0], requires_grad=True)
        y = x ** 3

        # d(x^3)/dx = 3x^2 = 12.0 at x=2, still attached to the graph.
        first = self.backend.autograd(y, x, create_graph=True)
        self.assertTrue(torch.allclose(first, torch.tensor([12.0])))
        self.assertTrue(first.requires_grad)

        # d(3x^2)/dx = 6x = 12.0 at x=2.
        second = self.backend.autograd(first, x)
        self.assertTrue(torch.allclose(second, torch.tensor([12.0])))

    def test_requires_grad_toggle(self) -> None:
        """requires_grad_ toggles gradient tracking in place."""
        tensor = torch.tensor([1.0], requires_grad=True)

        updated = self.backend.requires_grad_(tensor, requires_grad=False)

        self.assertIs(updated, tensor)
        self.assertFalse(updated.requires_grad)

    def test_convolution_1d(self) -> None:
        """Verify one-dimensional convolution."""
        # Input shape (batch=1, channels=1, length=5).
        input_tensor = torch.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0]]])

        # Kernel shape (out_channels=1, in_channels=1, kernel_size=3).
        # Kernel [1, 0, -1] computes left minus right.
        weight = torch.tensor([[[1.0, 0.0, -1.0]]])

        result = self.backend.conv1d(input_tensor, weight)

        expected = torch.tensor([[[-2.0, -2.0, -2.0]]])
        self.assertEqual(result.shape, (1, 1, 3))
        self.assertTrue(torch.allclose(result, expected))

    def test_convolution_2d(self) -> None:
        """Verify two-dimensional convolution."""
        # Input shape (batch=1, channels=1, height=3, width=3) filled with ones.
        input_tensor = torch.ones((1, 1, 3, 3))

        # Kernel shape (out_channels=1, in_channels=1, height=2, width=2) of ones.
        weight = torch.ones((1, 1, 2, 2))

        # Sum of four pixels equals 4.
        result = self.backend.conv2d(input_tensor, weight)

        expected = torch.full((1, 1, 2, 2), 4.0)
        self.assertEqual(result.shape, (1, 1, 2, 2))
        self.assertTrue(torch.allclose(result, expected))

    def test_convolution_2d_same_padding(self) -> None:
        """Convolution accepts the 'same' padding alias."""
        input_tensor = torch.arange(9, dtype=torch.float32).reshape(1, 1, 3, 3)
        weight = torch.ones((1, 1, 3, 3))

        result = self.backend.conv2d(input_tensor, weight, padding="same")

        self.assertEqual(result.shape, (1, 1, 3, 3))

    def test_convolution_3d(self) -> None:
        """Verify three-dimensional convolution."""
        # Input shape (batch=1, channels=1, depth=3, height=3, width=3).
        input_tensor = torch.ones((1, 1, 3, 3, 3))

        # Kernel shape (out_channels=1, in_channels=1, depth=2, height=2, width=2).
        weight = torch.ones((1, 1, 2, 2, 2))

        # Sum of eight voxels equals 8.
        result = self.backend.conv3d(input_tensor, weight)

        self.assertEqual(result.shape, (1, 1, 2, 2, 2))
        self.assertTrue(torch.all(result == 8.0))

    def test_convolution_rejects_unknown_padding(self) -> None:
        """Convolution helpers normalize only supported padding modes."""
        input_tensor = torch.ones((1, 1, 3))
        weight = torch.ones((1, 1, 1))

        with self.assertRaisesRegex(ValueError, "Unsupported padding"):
            self.backend.conv1d(input_tensor, weight, padding="wide")

    def test_convolution_valid_padding_shrinks_the_output(self) -> None:
        """'valid' padding (the default every operator relies on) adds no padding."""
        input_tensor = torch.ones((1, 1, 5))
        weight = torch.ones((1, 1, 2))

        result = self.backend.conv1d(input_tensor, weight, padding="valid")

        # A width-2 kernel over 5 nodes leaves 4 without padding.
        self.assertEqual(result.shape, (1, 1, 4))

    def test_convolution_padding_is_case_insensitive(self) -> None:
        """Padding modes are normalized regardless of casing."""
        input_tensor = torch.ones((1, 1, 5))
        weight = torch.ones((1, 1, 3))

        valid = self.backend.conv1d(input_tensor, weight, padding="VALID")
        same = self.backend.conv1d(input_tensor, weight, padding="Same")

        self.assertEqual(valid.shape, (1, 1, 3))
        self.assertEqual(same.shape, (1, 1, 5))

    def test_convolution_numeric_padding_passes_through(self) -> None:
        """A non-string padding is forwarded to the backend untouched."""

        input_tensor = torch.ones((1, 1, 3))
        weight = torch.ones((1, 1, 2))

        result = self.backend.conv1d(input_tensor, weight, padding=1)

        # One zero column on each side: 3 + 2 - 2 + 1 = 4 output positions.
        self.assertEqual(result.shape, (1, 1, 4))
        self.assertTrue(torch.allclose(
            result, torch.tensor([[[1.0, 2.0, 2.0, 1.0]]])))

    def test_convolution_with_bias(self) -> None:
        """Verify the bias term is added to the convolution result."""
        input_tensor = torch.ones((1, 1, 3))
        weight = torch.ones((1, 1, 2))
        bias = torch.tensor([5.0])

        result = self.backend.conv1d(input_tensor, weight, bias=bias)

        # Each window sums two ones, plus the bias: 2 + 5 = 7.
        self.assertTrue(torch.allclose(result, torch.full((1, 1, 2), 7.0)))

    def test_convolution_with_stride(self) -> None:
        """Verify stride skips windows."""
        input_tensor = torch.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0]]])
        weight = torch.ones((1, 1, 2))

        result = self.backend.conv1d(input_tensor, weight, stride=2)

        # Windows at offsets 0 and 2: (1+2)=3 and (3+4)=7.
        self.assertTrue(torch.allclose(result, torch.tensor([[[3.0, 7.0]]])))

    def test_convolution_with_dilation(self) -> None:
        """Verify dilation spreads the kernel taps apart."""
        input_tensor = torch.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0]]])
        weight = torch.ones((1, 1, 2))

        result = self.backend.conv1d(input_tensor, weight, dilation=2)

        # Taps at i and i+2: (1+3), (2+4), (3+5).
        self.assertTrue(torch.allclose(
            result, torch.tensor([[[4.0, 6.0, 8.0]]])))

    def test_convolution_with_groups(self) -> None:
        """Verify grouped convolution keeps channels independent."""
        channel_0 = torch.ones((1, 1, 3))
        channel_1 = torch.full((1, 1, 3), 2.0)
        input_tensor = torch.cat([channel_0, channel_1], dim=1)
        weight = torch.ones((2, 1, 2))

        result = self.backend.conv1d(input_tensor, weight, groups=2)

        # Channel 0 sums two ones (2.0); channel 1 sums two twos (4.0).
        expected = torch.tensor([[[2.0, 2.0], [4.0, 4.0]]])
        self.assertEqual(result.shape, (1, 2, 2))
        self.assertTrue(torch.allclose(result, expected))

    def test_dtype_conversion(self) -> None:
        """Verify casting tensors to different data types."""
        tensor = torch.tensor([1, 2, 3], dtype=torch.int32)
        converted = self.backend.to(tensor, dtype=torch.float32)

        self.assertEqual(converted.dtype, torch.float32)

    def test_device_conversion(self) -> None:
        """Verify moving tensors to different devices."""
        tensor = torch.tensor([1.0, 2.0, 3.0])
        converted = self.backend.to(tensor, device="cpu")

        self.assertEqual(str(converted.device), "cpu")

    def test_allclose_with_custom_tolerances(self) -> None:
        """allclose honors absolute and relative tolerances."""
        a = torch.tensor([1.0])
        b = torch.tensor([1.0 + 5e-4])

        self.assertTrue(self.backend.allclose(a, b, rtol=1e-3, atol=1e-3))
        self.assertFalse(self.backend.allclose(a, b, rtol=1e-6, atol=1e-6))


if __name__ == "__main__":
    unittest.main()