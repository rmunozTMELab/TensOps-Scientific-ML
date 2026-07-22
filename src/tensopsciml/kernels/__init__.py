"""Kernel-based operators (e.g., averaging, derivative)."""

from .average import Average
from .derivative import Derivative

__all__ = [
    "Average",
    "Derivative",
]
