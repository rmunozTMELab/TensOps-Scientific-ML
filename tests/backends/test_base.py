"""Tests for the Backend interface definition.

This module verifies that the Backend protocol defines the required API contract
that concrete implementations must satisfy, and that the shipped backend
actually satisfies it.
"""

import unittest
from typing import List

from tensopsciml.backends.base import Backend
from tensopsciml.backends.torch_backend import TorchBackend


def _protocol_members() -> List[str]:
    """Return every public member the Backend protocol declares.

    Derived from the protocol itself (methods plus annotated attributes) rather
    than hard-coded, so a member added to Backend is automatically required of
    every implementation instead of silently going unchecked.
    """
    methods = [
        name for name in vars(Backend)
        if not name.startswith("_") and callable(getattr(Backend, name))
    ]
    attributes = [
        name for name in getattr(Backend, "__annotations__", {})
        if not name.startswith("_")
    ]
    return sorted(set(methods) | set(attributes))


class TestBaseBackendInterface(unittest.TestCase):
    """Test suite for the Backend interface contract."""

    def test_backend_is_abstract_or_protocol(self) -> None:
        """Verify that Backend is defined as an interface.

        The interface should not be instantiated directly and should expose
        attributes that signal protocol or abstract base class semantics.
        """
        # Check whether the class exposes protocol or abstract base class metadata.
        is_protocol = hasattr(Backend, "_is_protocol") or hasattr(
            Backend, "__protocol_attrs__")
        is_abc = hasattr(Backend, "__abstractmethods__")

        self.assertTrue(
            is_protocol or is_abc,
            "Backend class should be a Protocol or Abstract Base Class."
        )

    def test_backend_defines_required_methods(self) -> None:
        """Verify that the Backend interface exposes all necessary methods.

        Any concrete backend implementation should implement these methods to
        satisfy static type checking and interface conformance.
        """
        required_methods: List[str] = [
            "allclose",
            "autograd",
            "clone",
            "conv1d",
            "conv2d",
            "conv3d",
            "det",
            "einsum",
            "equal",
            "eye",
            "inv",
            "ones",
            "permute",
            "requires_grad_",
            "squeeze",
            "stack",
            "tensor",
            "to",
            "transpose",
            "unsqueeze",
            "zeros",
            "zeros_like",
        ]

        missing_methods = [
            method for method in required_methods if not hasattr(Backend, method)
        ]

        self.assertFalse(
            missing_methods,
            f"The Backend interface is missing definitions for: {missing_methods}"
        )

    def test_backend_declares_dtype_attributes(self) -> None:
        """Verify that the interface declares the dtype attributes callers rely on."""
        annotations = getattr(Backend, "__annotations__", {})

        self.assertIn("float32", annotations)
        self.assertIn("float64", annotations)

    def test_torch_backend_implements_every_protocol_member(self) -> None:
        """Verify TorchBackend satisfies the whole protocol, member by member.

        Derived from the protocol at runtime, so a member added to Backend but
        never implemented in TorchBackend fails here instead of at the call site.
        """
        backend = TorchBackend()

        missing = [
            member for member in _protocol_members()
            if not hasattr(backend, member)
        ]

        self.assertFalse(
            missing,
            f"TorchBackend does not implement: {missing}"
        )

    def test_protocol_members_are_covered_by_the_required_list(self) -> None:
        """Verify the hard-coded required list stays in sync with the protocol.

        Guards against the failure mode this file previously had: the protocol
        grew (ones, eye, requires_grad_, allclose) while the list did not, so a
        backend missing those members still passed.
        """
        required = set(self.__class__._required_methods())
        declared = set(_protocol_members())

        self.assertSetEqual(
            declared,
            required,
            "The Backend protocol and the required-member list have diverged."
        )

    @staticmethod
    def _required_methods() -> List[str]:
        """Return the contract this suite asserts, including dtype attributes."""
        return [
            "allclose",
            "autograd",
            "clone",
            "conv1d",
            "conv2d",
            "conv3d",
            "det",
            "einsum",
            "equal",
            "eye",
            "float32",
            "float64",
            "inv",
            "ones",
            "permute",
            "requires_grad_",
            "squeeze",
            "stack",
            "tensor",
            "to",
            "transpose",
            "unsqueeze",
            "zeros",
            "zeros_like",
        ]


if __name__ == "__main__":
    unittest.main()
