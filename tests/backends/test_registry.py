"""Tests for the backend registry module.

This module verifies that the backend registry can dynamically switch between
different computation backends without tight coupling.
"""

import unittest
from typing import Any, List, Union
from tensopsciml.backends.registry import get_backend, set_backend, use_backend
from tensopsciml.backends.torch_backend import TorchBackend

# Mock backends (TF and JAX).

class MockTensorFlowBackend:
    """Simulate a TensorFlow backend for testing purposes."""

    def __init__(self) -> None:
        self.name = "tensorflow"

    def tensor(self, data: Any) -> str:
        """Create a dummy tensor representation.

        Args:
            data: Input data to convert.

        Returns:
            str: Mock TensorFlow tensor string.
        """
        return "tf_tensor"


class MockJaxBackend:
    """Simulate a JAX backend for testing purposes."""

    def __init__(self) -> None:
        self.name = "jax"

    def tensor(self, data: Any) -> str:
        """Create a dummy tensor representation.

        Args:
            data: Input data to convert.

        Returns:
            str: Mock JAX array string.
        """
        return "jax_array"


# Test suite.

BackendVariant = Union[TorchBackend, MockTensorFlowBackend, MockJaxBackend]

class TestRegistry(unittest.TestCase):
    """Test suite for the backend registry."""

    def setUp(self) -> None:
        """Prepare the test fixture before each test method.

        The fixture saves the current backend state and initializes a list of mixed
        backends (real and mock) to verify polymorphism.
        """
        self.original_backend = get_backend()

        # We test the registry against the real implementation and upcoming mocks.
        # This ensures the registry is not coupled to Torch-specific logic.
        self.backends_to_test: List[BackendVariant] = [
            TorchBackend(),
            MockTensorFlowBackend(),
            MockJaxBackend(),
        ]

    def tearDown(self) -> None:
        """Restore the environment after each test method.

        The teardown resets the global backend state to prevent side effects in other
        test suites.
        """
        set_backend(self.original_backend)

    def test_initial_state_is_valid(self) -> None:
        """Verify that the library initializes with a functional backend.

        The default backend must not be None and must satisfy the basic backend
        interface, including exposing a tensor method.
        """
        backend = get_backend()
        self.assertIsNotNone(backend, "Default backend should not be None.")
        self.assertTrue(
            hasattr(backend, "tensor"),
            "Default backend must implement a 'tensor' method."
        )

    def test_set_backend_enforces_switch(self) -> None:
        """Verify that set_backend permanently changes the global backend.

        The test iterates through multiple backend types to confirm that the registry
        accepts any valid object.
        """
        for backend in self.backends_to_test:
            # Safely retrieve name for test reporting.
            backend_name = getattr(backend, "name", "torch")

            with self.subTest(backend=backend_name):
                # Act.
                set_backend(backend)
                current = get_backend()

                # Assert.
                self.assertIs(
                    current,
                    backend,
                    f"Failed to switch global backend to {backend_name}."
                )

    def test_use_backend_context_manager(self) -> None:
        """Verify that use_backend temporarily overrides the backend.

        The backend should switch inside the context manager and restore its previous
        state afterwards.
        """
        original = get_backend()

        for temp_backend in self.backends_to_test:
            backend_name = temp_backend.__class__.__name__

            with self.subTest(backend=backend_name):
                # Act and assert inside the context.
                with use_backend(temp_backend):
                    self.assertIs(
                        get_backend(),
                        temp_backend,
                        "Context manager did not activate the temporary backend."
                    )

                # Assert restoration.
                self.assertIs(
                    get_backend(),
                    original,
                    "Context manager failed to restore the original backend."
                )

    def test_use_backend_restores_on_exception(self) -> None:
        """Verify backend restoration even when an exception occurs.

        The registry should recover the original backend after an exception inside the
        context manager.
        """
        original = get_backend()
        target_backend = MockJaxBackend()

        # Act.
        try:
            with use_backend(target_backend):
                # Verify switch happened before crash.
                self.assertIs(get_backend(), target_backend)
                raise RuntimeError("Simulated crash in user code")
        except RuntimeError:
            pass  # Expected exception.

        # Assert.
        current = get_backend()
        self.assertIs(
            current,
            original,
            "Backend was not restored after an exception occurred."
        )
        self.assertNotIsInstance(
            current,
            MockJaxBackend,
            "System remains stuck on the temporary backend."
        )


if __name__ == "__main__":
    unittest.main()