"""Global backend registry: hold and swap the active backend implementation."""

from contextlib import contextmanager
from typing import Iterator

from .base import Backend
from .torch_backend import TorchBackend


_backend: Backend = TorchBackend()


def get_backend() -> Backend:
    """Return the active backend instance.

    Returns:
        Backend: Currently configured backend implementation.
    """

    return _backend


def set_backend(backend: Backend) -> None:
    """Set the global backend instance.

    Args:
        backend (Backend): Backend implementation to register as the current
            backend.
    """

    global _backend
    _backend = backend


@contextmanager
def use_backend(backend: Backend) -> Iterator[None]:
    """Temporarily swap the active backend within a context.

    Args:
        backend (Backend): Backend to activate for the duration of the
            context.

    Yields:
        Iterator[None]: Control to the caller while the backend is active.
    """

    previous_backend = get_backend()
    try:
        set_backend(backend)
        yield
    finally:
        set_backend(previous_backend)
