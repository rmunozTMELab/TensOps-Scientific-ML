"""Tests for validation utilities."""

import unittest
from unittest.mock import MagicMock

from tensopsciml.backends import get_backend
from tensopsciml.calculus.connections import ConnectOps
from tensopsciml.tensors import MetrOps, TensOps, VolFormOps
from tensopsciml.utils.validation import (
    validate_codomain_matches_connection,
    validate_connectops,
    validate_kernels,
    validate_metrops,
    validate_tensops,
    validate_volformops,
)

backend = get_backend()


def _identity_metric(domain_dim: int, dimension: int, grid_size: int = 5) -> MetrOps:
    """Build an identity metric on a `domain_dim`-dimensional grid."""
    shape = [2, dimension, dimension] + [grid_size] * domain_dim
    values = backend.eye(dimension).reshape(
        1, dimension, dimension, *([1] * domain_dim)
    ).expand(*shape).clone()
    return MetrOps(values, domain_dim=domain_dim, codomain_dim=[dimension, dimension])


class TestValidateTensops(unittest.TestCase):
    """Test suite covering validate_tensops."""

    def setUp(self) -> None:
        """Prepare a reusable valid TensOps object."""
        self.tensor = TensOps(
            data=backend.zeros((1, 3, 10, 10, 10)),
            domain_dim=3,
            codomain_dim=[3],
            order=[1, 0],
            contravariant_indices=[0],
            covariant_indices=None,
        )

    def test_validate_tensops_valid(self) -> None:
        """Ensure validate_tensops accepts valid TensOps objects."""
        try:
            validate_tensops(self.tensor)
            validate_tensops(self.tensor, self.tensor)
        except ValueError:
            self.fail("validate_tensops() raised ValueError.")

    def test_validate_tensops_invalid_type(self) -> None:
        """Ensure validate_tensops rejects non-TensOps inputs."""
        invalid_inputs = [
            "not a tensor",
            123,
            backend.tensor([1.0]),
            [self.tensor],  # List containing tensor, not tensor itself
        ]
        for invalid in invalid_inputs:
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    validate_tensops(invalid)
                with self.assertRaises(ValueError):
                    validate_tensops(self.tensor, invalid)


class TestValidateMetrops(unittest.TestCase):
    """Test suite covering validate_metrops."""

    def test_validate_metrops_valid(self) -> None:
        """Ensure validate_metrops accepts MetrOps instances."""
        metric = _identity_metric(domain_dim=2, dimension=2)
        try:
            validate_metrops(metric)
            validate_metrops(metric, metric)
        except ValueError:
            self.fail("validate_metrops() raised ValueError.")

    def test_validate_metrops_rejects_plain_tensops(self) -> None:
        """Ensure a plain TensOps is not accepted as a metric."""
        plain = TensOps(
            backend.zeros((2, 2, 2, 5, 5)),
            domain_dim=2,
            codomain_dim=[2, 2],
            order=[0, 2],
            contravariant_indices=None,
            covariant_indices=[0, 1],
        )
        with self.assertRaisesRegex(ValueError, "Expected MetrOps"):
            validate_metrops(plain)

    def test_validate_metrops_rejects_non_tensor(self) -> None:
        """Ensure validate_metrops rejects arbitrary objects."""
        with self.assertRaises(ValueError):
            validate_metrops("not a metric")


class TestValidateVolformops(unittest.TestCase):
    """Test suite covering validate_volformops."""

    def test_validate_volformops_valid(self) -> None:
        """Ensure validate_volformops accepts VolFormOps instances."""
        volume_form = VolFormOps(_identity_metric(domain_dim=2, dimension=2))
        try:
            validate_volformops(volume_form)
            validate_volformops(volume_form, volume_form)
        except ValueError:
            self.fail("validate_volformops() raised ValueError.")

    def test_validate_volformops_rejects_metrops(self) -> None:
        """Ensure a MetrOps is not accepted as a volume form."""
        metric = _identity_metric(domain_dim=2, dimension=2)
        with self.assertRaisesRegex(ValueError, "Expected VolFormOps"):
            validate_volformops(metric)

    def test_validate_volformops_rejects_plain_tensops(self) -> None:
        """Ensure a plain scalar TensOps is not accepted as a volume form."""
        scalar = TensOps(
            backend.ones((2, 1, 5, 5)),
            domain_dim=2,
            codomain_dim=[1],
            order=[0, 0],
            contravariant_indices=None,
            covariant_indices=None,
        )
        with self.assertRaises(ValueError):
            validate_volformops(scalar)


class TestValidateConnectops(unittest.TestCase):
    """Test suite covering validate_connectops."""

    def setUp(self) -> None:
        """Build a connection from raw (zero) Christoffel coefficients."""
        self.connection = ConnectOps(
            backend.zeros((2, 2, 2, 2, 5, 5)),
            domain_dim=2,
            codomain_dim=[2, 2, 2],
        )

    def test_validate_connectops_valid(self) -> None:
        """Ensure validate_connectops accepts ConnectOps instances."""
        try:
            validate_connectops(self.connection)
            validate_connectops(self.connection, self.connection)
        except ValueError:
            self.fail("validate_connectops() raised ValueError.")

    def test_validate_connectops_rejects_tensops(self) -> None:
        """Ensure a TensOps is not accepted as a connection."""
        metric = _identity_metric(domain_dim=2, dimension=2)
        with self.assertRaisesRegex(ValueError, "Expected ConnectOps"):
            validate_connectops(metric)

    def test_validate_connectops_rejects_non_tensor(self) -> None:
        """Ensure validate_connectops rejects arbitrary objects."""
        with self.assertRaises(ValueError):
            validate_connectops("not a connection")


class TestValidateCodomainMatchesConnection(unittest.TestCase):
    """Test suite covering validate_codomain_matches_connection."""

    def setUp(self) -> None:
        """Build a 2D connection whose indices range over dimension 2."""
        self.connection = ConnectOps(
            backend.zeros((2, 2, 2, 2, 5, 5)),
            domain_dim=2,
            codomain_dim=[2, 2, 2],
        )

    def test_matching_codomain_accepted(self) -> None:
        """A field whose codomain dims equal the connection's is accepted."""
        field = TensOps(
            backend.zeros((2, 2, 2, 5, 5)),
            domain_dim=2,
            codomain_dim=[2, 2],
            order=[1, 1],
            contravariant_indices=[0],
            covariant_indices=[1],
        )
        try:
            validate_codomain_matches_connection(field, self.connection)
        except ValueError:
            self.fail(
                "validate_codomain_matches_connection() raised ValueError.")

    def test_mismatched_codomain_rejected(self) -> None:
        """A field whose codomain dims differ from the connection's is rejected."""
        field = TensOps(
            backend.zeros((2, 3, 5, 5)),
            domain_dim=2,
            codomain_dim=[3],
            order=[1, 0],
            contravariant_indices=[0],
            covariant_indices=None,
        )
        with self.assertRaisesRegex(ValueError, "codomain dimensions must match"):
            validate_codomain_matches_connection(field, self.connection)

    def test_scalar_field_is_exempt(self) -> None:
        """A scalar field has no free indices, so any codomain is accepted."""
        scalar = TensOps(
            backend.zeros((2, 1, 5, 5)),
            domain_dim=2,
            codomain_dim=[1],
            order=[0, 0],
            contravariant_indices=None,
            covariant_indices=None,
        )
        try:
            validate_codomain_matches_connection(scalar, self.connection)
        except ValueError:
            self.fail(
                "validate_codomain_matches_connection() rejected a scalar field.")


class TestValidateKernels(unittest.TestCase):
    """Test suite covering validate_kernels."""

    def setUp(self) -> None:
        """Prepare a kernel bank matching a three-dimensional domain."""
        self.kernel = MagicMock()
        self.kernel.shape = [3, 1, 1, 2, 1, 1]  # [domain_dim, ...]

    def test_validate_kernels_valid(self) -> None:
        """Ensure validate_kernels accepts a kernel bank matching the domain."""
        field = TensOps(
            data=backend.zeros((10, 1, 5)),
            domain_dim=1,
            codomain_dim=[1],
            order=[0, 0],
            contravariant_indices=None,
            covariant_indices=None,
        )
        kernel = MagicMock()
        kernel.shape = [1, 1, 1, 2]  # Matches domain_dim=1

        try:
            validate_kernels(field, kernel)
        except ValueError:
            self.fail("validate_kernels() raised ValueError.")

    def test_validate_kernels_accepts_non_scalar_field(self) -> None:
        """Variance is the caller's concern: a vector field is accepted."""
        vector = TensOps(
            data=backend.zeros((1, 3, 10, 10, 10)),
            domain_dim=3,
            codomain_dim=[3],
            order=[1, 0],
            contravariant_indices=[0],
            covariant_indices=None,
        )
        try:
            validate_kernels(vector, self.kernel)
        except ValueError:
            self.fail("validate_kernels() rejected a non-scalar field.")

    def test_validate_kernels_rejects_non_tensops(self) -> None:
        """Ensure validate_kernels rejects a field that is not a TensOps."""
        with self.assertRaises(ValueError):
            validate_kernels("not a tensor", self.kernel)

    def test_validate_kernels_rejects_unsupported_domain(self) -> None:
        """Ensure validate_kernels rejects a domain outside {1, 2, 3}."""
        field = TensOps(
            data=backend.zeros((1, 1, 10)),
            domain_dim=1,
            codomain_dim=[1],
            order=[0, 0],
            contravariant_indices=None,
            covariant_indices=None,
        )
        object.__setattr__(field, "_TensOps__domain", 4)
        with self.assertRaisesRegex(ValueError, "Unsupported domain"):
            validate_kernels(field, self.kernel)

    def test_validate_kernels_rejects_kernel_count_mismatch(self) -> None:
        """Ensure the kernel count must equal the field's domain dimension."""
        field = TensOps(
            data=backend.zeros((1, 1, 10)),
            domain_dim=1,
            codomain_dim=[1],
            order=[0, 0],
            contravariant_indices=None,
            covariant_indices=None,
        )
        mismatch_kernel = MagicMock()
        mismatch_kernel.shape = [2, 1, 1, 2, 2]  # Field domain is 1

        with self.assertRaisesRegex(ValueError, "number of derivative kernels"):
            validate_kernels(field, mismatch_kernel)


if __name__ == "__main__":
    unittest.main()
