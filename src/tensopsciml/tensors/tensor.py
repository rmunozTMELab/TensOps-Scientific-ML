"""Core tensor class enriched with backend-agnostic tensor metadata."""

from __future__ import annotations

import numbers
from typing import Any

from ..backends import get_backend


def _is_scalar_value(value: Any) -> bool:
    """Returns True for Python/numpy numbers and 0-dim array-like scalars."""

    return isinstance(value, numbers.Real) or getattr(value, "ndim", None) == 0


class TensOps:
    """Backend tensor enriched with tensor calculus metadata.

    The class wraps a backend-provided tensor and encodes its mathematical
    structure, including domain and codomain dimensions, tensorial order, and
    index positions. It enables rigorous tensor calculus operations, such as
    index manipulation, validation, and arithmetic, following multilinear
    algebra conventions.

    The tensor data follows the shape convention
    [n_samples, codomain..., discretization...], where n_samples is the number
    of data samples, codomain represents the tensor rank dimensions, and
    discretization corresponds to spatial or domain dimensions.

    Attributes: 
        values (Any): Underlying backend tensor data.
        domain (int): Number of domain dimensions (spatial discretization).
        codomain (list[int]): Dimensions of the codomain (tensor rank).
        order (list[int]): Two-element list ``[contravariant_order,
            covariant_order]``.
        contravariant (list[int] | None): Positions of contravariant (upper)
            indices.
        covariant (list[int] | None): Positions of covariant (lower) indices.
        base (str | None): String representation of the tensor base describing
            the composition of vector spaces (V) and their duals (V*).

    Examples:
        >>> import torch
        >>> from tensopsciml.tensors.tensor import TensOps
        >>> data = torch.rand((100, 3, 3, 10, 10, 10))
        >>> TensOps(
        ...     data,
        ...     domain_dim=3,
        ...     codomain_dim=[3, 3],
        ...     order=[1, 1],
        ...     contravariant_indices=[0],
        ...     covariant_indices=[1],
        ... )
    """

    def __init__(
        self,
        data: Any,
        domain_dim: int,
        codomain_dim: list[int],
        order: list[int],
        contravariant_indices: list[int] | None,
        covariant_indices: list[int] | None,
    ):
        """Initialize a TensOps object.

        Args:
            data (Any): Tensor data with shape
                ``[n_samples, codomain..., discretization...]``.
            domain_dim (int): Number of domain dimensions (spatial
                discretization).
            codomain_dim (list[int]): Dimensions of the codomain defining the
                tensor rank.
            order (list[int]): Two-element list ``[contravariant_order,
                covariant_order]``.
            contravariant_indices (list[int] | None): Positions for the
                contravariant (upper) indices.
            covariant_indices (list[int] | None): Positions for the covariant
                (lower) indices.

        Raises:
            ValueError: If any parameter is invalid or the tensor metadata is
                mathematically inconsistent (for example, mismatched
                dimensions or incorrect index positions).
        """
        backend = get_backend()

        self.__validate_parameters(
            data,
            domain_dim,
            codomain_dim,
            order,
            contravariant_indices,
            covariant_indices,
        )

        self.__values = backend.clone(data)
        self.__domain = domain_dim
        self.__codomain = codomain_dim
        self.__order = order
        self.__contravariant_indices = contravariant_indices
        self.__covariant_indices = covariant_indices

        self.__verify_mathematical_consistency()

    def __validate_parameters(
        self,
        data: Any,
        domain_dim: int,
        codomain_dim: list[int],
        order: list[int],
        contravariant_indices: list[int] | None,
        covariant_indices: list[int] | None,
    ) -> None:
        """Validate parameter types for TensOps initialization.

        Args:
            data (Any): Tensor data to wrap.
            domain_dim (int): Number of domain dimensions.
            codomain_dim (list[int]): Dimensions of the codomain.
            order (list[int]): Tensorial order ``[contravariant, covariant]``.
            contravariant_indices (list[int] | None): Positions of
                contravariant indices.
            covariant_indices (list[int] | None): Positions of covariant
                indices.

        Raises:
            ValueError: If any parameter has an invalid type.
        """
        params: dict[str, tuple[Any, tuple[type, ...]]] = {
            "contravariant_indices": (contravariant_indices, (list, type(None))),
            "covariant_indices": (covariant_indices, (list, type(None))),
            "domain_dim": (domain_dim, (int,)),
            "codomain_dim": (codomain_dim, (list,)),
            "order": (order, (list,)),
        }

        for param_name, (param_value, expected_types) in params.items():
            if not isinstance(param_value, expected_types):
                raise ValueError(
                    f"Expected {expected_types} for '{param_name}', received {type(param_value)}"
                )

        list_params = {
            "contravariant_indices": contravariant_indices,
            "covariant_indices": covariant_indices,
            "codomain_dim": codomain_dim,
        }
        for list_name, list_value in list_params.items():
            if list_value is None:
                continue
            if not all(isinstance(index, int) for index in list_value):
                raise ValueError(
                    f"All elements of '{list_name}' must be integers, received {list_value}"
                )

        backend = get_backend()
        try:
            backend.tensor(data)
        except Exception as exc:  # pylint: disable=broad-except
            raise ValueError(
                "Data cannot be converted into a backend tensor.") from exc

    def __validate_dimensions(self) -> None:
        """Validate that the domain dimensionality is supported.

        Raises:
            ValueError: If the domain dimensionality falls outside {1, 2, 3}.
        """

        allowed_dimensions = {1, 2, 3}
        if self.domain not in allowed_dimensions:
            raise ValueError(
                f"Domain dimension {self.domain} must be one of {sorted(allowed_dimensions)}."
            )

    def __validate_shape(self) -> None:
        """Ensure the tensor shape matches its mathematical structure.

        Raises:
            ValueError: If the tensor shape is incompatible with domain or
                codomain metadata.
        """

        actual_shape = list(self.__values.shape)
        expected_shape = [actual_shape[0]] + \
            self.codomain + actual_shape[-self.domain:]

        if actual_shape != expected_shape:
            raise ValueError(
                "Inconsistent tensor shape: "
                f"expected {expected_shape}, received {actual_shape}."
            )

    def __validate_tensorial_order(self) -> None:
        """Validate the tensorial order and index consistency.

        The method checks that a scalar field (tensorial order 0) has codomain
        ``[1]``, that non-scalar fields declare an order matching the number of
        codomain dimensions, contravariant and covariant indices do not overlap,
        the number of indices matches the declared order, and each index
        position is valid.

        Raises:
            ValueError: If the tensorial order conflicts with the codomain
                dimensions or the indices are improperly defined.
        """
        codomain_dims = self.codomain
        total_order = sum(self.order)

        # The tensorial order, not the codomain, is what defines a scalar field:
        # a scalar has order [0, 0], while a 1D vector/covector has order
        # [1, 0]/[0, 1]. All three carry codomain [1], so discriminating by the
        # codomain would conflate them.
        if total_order == 0:
            if codomain_dims != [1]:
                raise ValueError(
                    "A scalar field (tensorial order 0) must have codomain [1]."
                )
        elif total_order != len(codomain_dims):
            raise ValueError(
                "Tensorial order cannot differ from the number of codomain dimensions."
            )

        contra = len(self.contravariant or [])
        cov = len(self.covariant or [])
        indices_order = [contra, cov]

        repeated = set(self.contravariant or []) & set(self.covariant or [])
        if repeated:
            raise ValueError(f"Repeated indices are not allowed: {repeated}.")

        if self.order != indices_order:
            raise ValueError(
                "Contravariant and covariant indices must match the declared tensorial order."
            )

        if self.contravariant is not None:
            if not all(0 <= index < total_order for index in self.contravariant):
                raise ValueError(
                    "Contravariant index positions must fall within the tensorial order."
                )

        if self.covariant is not None:
            if not all(0 <= index < total_order for index in self.covariant):
                raise ValueError(
                    "Covariant index positions must fall within the tensorial order."
                )

    def __verify_mathematical_consistency(self) -> None:
        """Verify the overall mathematical consistency of the tensor.

        This method runs a sequence of validation checks to ensure that the
        tensor's properties (dimensions, shape, and order) are coherent.
        """
        self.__validate_dimensions()
        self.__validate_shape()
        self.__validate_tensorial_order()

    def __get_base(self) -> list[str] | None:
        """Define the tensor base as a composition of vector spaces.

        The base identifies whether each index belongs to a vector space (V) or
        its dual (V*). V corresponds to a contravariant index, and V* (denoted
        internally as D) corresponds to a covariant index.

        Returns:
            Optional[list[str]]: Sequence of "V" or "D" tokens describing the
                tensor base, or None for a scalar.
        """
        if sum(self.order) == 0:
            return None

        covariant_set = set(self.covariant or [])
        contravariant_count = len(self.contravariant or [])
        covariant_count = len(self.covariant or [])
        total_indices = contravariant_count + covariant_count
        return ["D" if index in covariant_set else "V" for index in range(total_indices)]

    @property
    def values(self) -> Any:
        """Return the underlying tensor data.

        Returns:
            Any: Backend tensor holding the numerical values.
        """
        return self.__values

    @property
    def domain(self) -> int:
        """Return the number of domain dimensions (spatial discretization).

        Returns:
            int: Number of dimensions in the tensor's domain.
        """
        return self.__domain

    @property
    def codomain(self) -> list[int]:
        """Return the dimensions of the codomain (tensor rank).

        Returns:
            list[int]: Integers representing the codomain dimensions.
        """
        return self.__codomain

    @property
    def order(self) -> list[int]:
        """Return the tensorial order [contravariant, covariant].

        Returns:
            list[int]: Number of contravariant and covariant indices.
        """
        return self.__order

    @property
    def contravariant(self) -> list[int] | None:
        """Return the positions of the contravariant (upper) indices.

        Returns:
            Optional[list[int]]: Integer positions for contravariant indices.
        """
        return self.__contravariant_indices

    @property
    def covariant(self) -> list[int] | None:
        """Return the positions of the covariant (lower) indices.

        Returns:
            Optional[list[int]]: Integer positions for covariant indices.
        """
        return self.__covariant_indices

    @property
    def base(self) -> str | None:
        """Return a string representation of the tensor's base.

        The base is formed by the tensor product of vector spaces (V) and their
        duals (V*). For example, a tensor of order (1, 1) has the base "V x V*".

        Returns:
            Optional[str]: Base description or None for a scalar field.
        """

        base_list = self.__get_base()

        if base_list is None:
            return None

        symbol_map = {"V": "V", "D": "V*"}
        return " x ".join(symbol_map.get(val, "") for val in base_list)

    @property
    def is_scalar(self) -> bool:
        """Return whether the tensor represents a scalar field.

        A scalar field is identified by a zero tensorial order
        (``order == [0, 0]``). This is what distinguishes it from a 1D
        vector/covector, since both share codomain ``[1]``.
        """

        return sum(self.order) == 0

    def __ensure_batch_and_domain_match(self, other: TensOps) -> None:
        """Validate batch size and domain alignment between tensors.

        Args:
            other (TensOps): Tensor to compare against.

        Raises:
            ValueError: If the tensors differ in sample count or domain dimensionality.
        """

        if self.values.shape[0] != other.values.shape[0]:
            raise ValueError("Tensors must share the same number of samples.")
        if self.domain != other.domain:
            raise ValueError(
                "Tensor domains must match to perform this operation.")

    def __expand_scalar_like(self, target: TensOps) -> Any:
        """Broadcast a scalar tensor to match the codomain rank of target.

        Args:
            target (TensOps): Tensor whose codomain rank should be matched.

        Returns:
            Any: Broadcasted scalar tensor.
        """

        backend = get_backend()
        tensor = self.values
        additional_dims = max(len(target.codomain) - len(self.codomain), 0)
        for _ in range(additional_dims):
            tensor = backend.unsqueeze(tensor, 1)
        return tensor

    def __add__(self, other: TensOps) -> TensOps:
        """Perform element-wise addition of two tensors.

        Tensors must have the same shape and base to be added.

        Args:
            other (TensOps): Tensor to add to the current object.

        Returns:
            TensOps: New tensor representing the sum.

        Raises:
            TypeError: If other is not a TensOps instance.
            ValueError: If the tensors have mismatched shapes or bases.

        Examples:
            >>> T3 = T1 + T2
        """
        if not isinstance(other, TensOps):
            raise TypeError(
                f"Addition is only supported with another TensOps, not {type(other).__name__}"
            )

        if self.values.shape != other.values.shape or self.base != other.base:
            raise ValueError(
                "Cannot add tensors with mismatched properties.\n"
                f" Left shape: {self.values.shape}\n"
                f" Right shape: {other.values.shape}\n"
                f" Base: {self.base} vs {other.base}"
            )
        return TensOps(
            self.values + other.values,
            self.domain,
            self.codomain,
            self.order,
            self.contravariant,
            self.covariant,
        )

    def __sub__(self, other: TensOps) -> TensOps:
        """Perform element-wise subtraction of two tensors.

        Tensors must have the same shape and base for subtraction.

        Args:
            other (TensOps): Tensor to subtract from the current object.

        Returns:
            TensOps: New tensor representing the difference.

        Raises:
            TypeError: If other is not a TensOps instance.
            ValueError: If the tensors have mismatched shapes or bases.

        Examples:
            >>> T_diff = T1 - T2
        """
        if not isinstance(other, TensOps):
            raise TypeError(
                f"Subtraction is only supported with another TensOps, not {type(other).__name__}"
            )

        if self.values.shape != other.values.shape or self.base != other.base:
            raise ValueError(
                "Cannot subtract tensors with mismatched properties.\n"
                f" Left shape: {self.values.shape}\n"
                f" Right shape: {other.values.shape}\n"
                f" Base: {self.base} vs {other.base}"
            )
        return TensOps(
            self.values - other.values,
            self.domain,
            self.codomain,
            self.order,
            self.contravariant,
            self.covariant,
        )

    def __mul__(self, scalar: int | float | TensOps) -> TensOps:
        """Perform element-wise multiplication with a scalar or tensor.

        This operation supports multiplication by Python integers or floats,
        or by a TensOps object representing a scalar field.

        Args:
            scalar (int | float | TensOps): Value to multiply by. When a
                ``TensOps`` instance, it must represent a scalar field
                (codomain ``[1]``).

        Returns:
            TensOps: New tensor representing the product.

        Raises:
            ValueError: If the operation is invalid due to shape, domain, or
                type constraints.

        Examples:
            >>> T_doubled = T * 2
            >>> # S is a TensOps scalar field.
            >>> T_scaled = T * S
        """
        if _is_scalar_value(scalar):
            return TensOps(
                self.values * scalar,
                self.domain,
                self.codomain,
                self.order,
                self.contravariant,
                self.covariant,
            )

        if not isinstance(scalar, TensOps):
            raise ValueError(
                "Multiplication only supports int, float, or TensOps scalar fields.")

        self.__ensure_batch_and_domain_match(scalar)

        if self.is_scalar and not scalar.is_scalar:
            expanded_self = self.__expand_scalar_like(scalar)
            return TensOps(
                expanded_self * scalar.values,
                scalar.domain,
                scalar.codomain,
                scalar.order,
                scalar.contravariant,
                scalar.covariant,
            )

        if not self.is_scalar and scalar.is_scalar:
            expanded_scalar = scalar.__expand_scalar_like(self)
            return TensOps(
                self.values * expanded_scalar,
                self.domain,
                self.codomain,
                self.order,
                self.contravariant,
                self.covariant,
            )

        if self.is_scalar and scalar.is_scalar:
            # Both operands are scalars, so both necessarily carry order [0, 0]
            # and codomain [1] (enforced at construction): their metadata always
            # agrees and needs no further check.
            return TensOps(
                self.values * scalar.values,
                max(self.domain, scalar.domain),
                self.codomain,
                self.order,
                self.contravariant or scalar.contravariant,
                self.covariant or scalar.covariant,
            )

        raise ValueError(
            "Multiplication only supports int, float, or TensOps scalar fields."
        )

    def __rmul__(self, scalar: int | float) -> TensOps:
        """Perform element-wise right multiplication by a scalar.

        This method allows a TensOps object to be multiplied by a scalar from
        the left (for example, 2 * T).

        Args:
            scalar (int | float): Scalar value to multiply by.

        Returns:
            TensOps: New tensor representing the product.

        Examples:
            >>> T_doubled = 2 * T
        """
        return self.__mul__(scalar)

    def __truediv__(self, scalar: int | float | TensOps) -> TensOps:
        """Perform element-wise division by a scalar or scalar field.

        This operation supports division by Python/numpy numbers, 0-dim
        array-like scalars, or a TensOps object representing a scalar field.

        Args:
            scalar (int | float | TensOps): Value to divide by. When a
                ``TensOps`` instance, it must represent a scalar field
                (codomain ``[1]``).

        Returns:
            TensOps: New tensor representing the result of the division.

        Raises:
            TypeError: If scalar is not a number or a TensOps scalar field.
            ValueError: If a TensOps divisor's metadata is incompatible.
            ZeroDivisionError: If scalar is zero.

        Examples:
            >>> T_half = T / 2.0
            >>> # S is a TensOps scalar field.
            >>> T_scaled = T / S
        """
        if _is_scalar_value(scalar):
            if scalar == 0:
                raise ZeroDivisionError("Cannot divide by zero.")

            return TensOps(
                self.values / scalar,
                self.domain,
                self.codomain,
                self.order,
                self.contravariant,
                self.covariant,
            )

        if not isinstance(scalar, TensOps) or not scalar.is_scalar:
            raise TypeError(
                f"Scalar must be an int, float, or TensOps scalar field, not {type(scalar).__name__}"
            )

        self.__ensure_batch_and_domain_match(scalar)

        if self.is_scalar:
            # The divisor is already known to be a scalar, so both operands
            # carry order [0, 0] and codomain [1] (enforced at construction):
            # their metadata always agrees and needs no further check.
            return TensOps(
                self.values / scalar.values,
                max(self.domain, scalar.domain),
                self.codomain,
                self.order,
                self.contravariant or scalar.contravariant,
                self.covariant or scalar.covariant,
            )

        expanded_scalar = scalar.__expand_scalar_like(self)
        return TensOps(
            self.values / expanded_scalar,
            self.domain,
            self.codomain,
            self.order,
            self.contravariant,
            self.covariant,
        )

    def __neg__(self) -> TensOps:
        """Perform element-wise negation of the tensor.

        Returns:
            TensOps: New tensor with all values negated.

        Examples:
            >>> T_neg = -T
        """
        return TensOps(
            -self.values,
            self.domain,
            self.codomain,
            self.order,
            self.contravariant,
            self.covariant,
        )

    def __pos__(self) -> TensOps:
        """Return the tensor unchanged (unary plus).

        Returns:
            TensOps: New tensor with the same values.

        Examples:
            >>> T_pos = +T
        """
        return TensOps(
            +self.values,
            self.domain,
            self.codomain,
            self.order,
            self.contravariant,
            self.covariant,
        )

    def __eq__(self, other: object) -> bool:
        """Check if two TensOps objects are equal.

        Equality requires identical shapes, values, and mathematical metadata
        (domain, codomain, and index types).

        Args:
            other (object): Tensor object to compare against.

        Returns:
            bool: True if the tensors are equal, False otherwise.

        Examples:
            >>> T1 == T2
        """
        if not isinstance(other, TensOps):
            return NotImplemented

        backend = get_backend()

        return (
            self.values.shape == other.values.shape and
            backend.equal(self.values, other.values) and
            self.domain == other.domain and
            self.codomain == other.codomain and
            self.contravariant == other.contravariant and
            self.covariant == other.covariant)

    def __hash__(self) -> int:
        """Hash a TensOps object by identity.

        Since TensOps contains mutable torch tensors, hashing by value is
        unsafe: the hash would change if the underlying data changed. Instead,
        hash by object identity using id(). This allows TensOps to be used in
        sets and as dict keys, but with the understanding that equality and
        hashing are decoupled (two numerically-equal tensors will have different
        hashes).

        Returns:
            int: Hash based on object identity.
        """
        return hash(id(self))

    def keys(self) -> list[str]:
        """Return the tensor's logical attribute names.

        Together with ``__getitem__`` this implements the mapping protocol,
        so ``dict(T)`` yields a metadata dictionary of the tensor.

        Returns:
            list[str]: Key attribute names of the tensor object.
        """
        return ["values", "domain", "codomain", "contravariant", "covariant", "order"]

    def __getitem__(self, key: str) -> Any:
        """Return the attribute named by ``key``.

        Completes the mapping protocol announced by ``keys()`` (enables
        ``dict(T)``).

        Args:
            key (str): One of the names returned by ``keys()``.

        Returns:
            Any: Value of the requested attribute.

        Raises:
            KeyError: If ``key`` is not one of the names in ``keys()``.
        """
        if key not in self.keys():
            raise KeyError(key)
        return getattr(self, key)

    def __str__(self) -> str:
        """Return a user-friendly string representation of the tensor.

        This representation provides a summary of the tensor's key properties,
        including its shape, order, dimensions, and base.

        Returns:
            str: A formatted string describing the tensor.
        """
        return (
            f"<class '{self.__class__.__name__}'>\n"
            f" Shape: {list(self.values.shape)}\n"
            "\n"
            f" Order: {self.order}\n"
            f" Domain dim: {self.domain}\n"
            f" Codomain dim: {self.codomain}\n"
            "\n"
            f" Base: {self.base}\n"
            f" Contravariant indices: {self.contravariant}\n"
            f" Covariant indices: {self.covariant}")

    def __repr__(self) -> str:
        """Return a detailed string representation for debugging.

        This representation is identical to __str__ and is useful for
        inspecting the object in interactive sessions.

        Returns:
            str: A formatted string describing the tensor.
        """
        return self.__str__()

    def requires_grad_(self, requires_grad: bool = True) -> TensOps:
        """Set the requires_grad attribute of the backend tensor in place.

        The method toggles gradient tracking on the internal tensor and returns
        the TensOps instance to support method chaining.

        Args:
            requires_grad (bool): Whether to enable gradient tracking.
                Defaults to ``True``.

        Returns:
            TensOps: Tensor with the updated gradient tracking flag.

        Examples:
            >>> T = TensOps(torch.rand(2, 2), ...)
            >>> T.requires_grad_(True)
            >>> T.values.requires_grad
            True
        """
        backend = get_backend()
        self.__values = backend.requires_grad_(self.values, requires_grad)
        return self

    def to(self, device: Any = None, *, dtype: Any | None = None) -> TensOps:
        """Move the tensor data to the specified device.

        The method returns a new TensOps object with tensor data transferred to
        the provided device, leaving the original object unchanged.

        Args:
            device (Any | None): Target device for the tensor data (for
                example ``"cpu"`` or ``"cuda"``).
            dtype (Any | None): Optional target dtype supplied to the backend.

        Returns:
            TensOps: New tensor with data located on the specified device.

        Examples:
            >>> # Assuming T is on CPU
            >>> T_gpu = T.to("cuda")
        """
        backend = get_backend()
        new_values = backend.to(self.values, dtype=dtype, device=device)

        return TensOps(
            new_values,
            self.domain,
            self.codomain,
            self.order,
            self.contravariant,
            self.covariant,
        )
