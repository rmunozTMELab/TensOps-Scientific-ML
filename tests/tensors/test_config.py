"""Tests for the TensOps configuration class."""

import unittest
from dataclasses import asdict
from tensopsciml.tensors.config import TensorConfig


class TestConfig(unittest.TestCase):
    """Test suite for TensOps configuration."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.config_data = {
            "domain_dim": 3,
            "codomain_dim": [3, 3],
            "order": [1, 1],
            "contravariant_indices": [0],
            "covariant_indices": [1],
        }

    def test_config_initialization(self) -> None:
        """Test that TensorConfig initializes correctly with all parameters."""
        config = TensorConfig(**self.config_data)
        self.assertEqual(config.domain_dim, self.config_data["domain_dim"])
        self.assertEqual(config.codomain_dim, self.config_data["codomain_dim"])
        self.assertEqual(config.order, self.config_data["order"])
        self.assertEqual(
            config.contravariant_indices,
            self.config_data["contravariant_indices"],
        )
        self.assertEqual(
            config.covariant_indices,
            self.config_data["covariant_indices"],
        )

    def test_config_default_values(self) -> None:
        """Test that optional fields default to None when omitted."""
        config_data = {
            "domain_dim": 3,
            "codomain_dim": [3],
            "order": [1, 0],
        }
        config = TensorConfig(**config_data)
        self.assertIsNone(config.contravariant_indices)
        self.assertIsNone(config.covariant_indices)

    def test_config_attribute_error_on_new_field(self) -> None:
        """Test that assigning a new attribute raises AttributeError."""
        config = TensorConfig(**self.config_data)
        with self.assertRaises(AttributeError):
            config.new_field = "test"

    def test_config_from_dict_unpacking(self) -> None:
        """Test creating a TensorConfig instance via dictionary unpacking."""
        config = TensorConfig(**self.config_data)
        self.assertEqual(config.domain_dim, 3)
        self.assertEqual(config.codomain_dim, [3, 3])

    def test_config_to_dict(self) -> None:
        """Test converting a TensorConfig instance to a dictionary."""
        config = TensorConfig(**self.config_data)
        config_dict = asdict(config)
        self.assertEqual(config_dict, self.config_data)


if __name__ == "__main__":
    unittest.main()
