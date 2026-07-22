"""Tensor algebra utilities for the TensOps computational framework."""

import math
from itertools import permutations
from typing import Any

from ..backends import get_backend
from ..tensors import MetrOps, TensOps, VolFormOps
from ..utils.indices import (
    check_contraction_order,
    check_index_letter_capacity,
    check_indices_list_double,
    check_not_repeat_indices,
    combine_characters_delete_repeated,
    permutation_sign,
    reindex_by_sorted_order,
)
from ..utils.validation import validate_metrops, validate_tensops
from .matrix_algebra import inverse


def contraction(
    T: TensOps,
    S: TensOps,
    T_contracted_indices: list[list[int]],
    S_contracted_indices: list[list[int]],
) -> TensOps:
    """Contract contravariant and covariant indices across two tensors.

    Args:
        T (TensOps): First tensor to be contracted.
        S (TensOps): Second tensor to be contracted.
        T_contracted_indices (list[list[int]]): Pair of lists with the
            contravariant and covariant indices from ``T`` to contract. Use
            ``[None]`` when the corresponding variance has no contracted
            indices.
        S_contracted_indices (list[list[int]]): Pair of lists with the
            contravariant and covariant indices from ``S`` to contract.

    Returns:
        TensOps: Tensor with contracted indices.

    Raises:
        ValueError: If the tensors, indices, or space dimensions are invalid.

    Examples:
        >>> import torch
        >>> from tensopsciml import TensOps
        >>> from tensopsciml.algebra import contraction
        >>> a_torch = torch.rand((100, 2, 10, 10, 10))
        >>> b_torch = torch.rand((100, 2, 2, 10, 10, 10))
        >>> a = TensOps(
        ...     a_torch,
        ...     domain_dim=3,
        ...     codomain_dim=[2],
        ...     order=[1, 0],
        ...     contravariant_indices=[0],
        ...     covariant_indices=None,
        ... )
        >>> b = TensOps(
        ...     b_torch,
        ...     domain_dim=3,
        ...     codomain_dim=[2, 2],
        ...     order=[1, 1],
        ...     contravariant_indices=[0],
        ...     covariant_indices=[1],
        ... )
        >>> contraction(a, b, [[0], [None]], [[None], [0]]).order
        [1, 0]
        >>> contraction(a, b, [[0], [None]], [[None], [0]]).values.shape
        torch.Size([100, 2, 10, 10, 10])
    """
    indices_characters = 'ijklmpqrstuvwa'
    discretization = {1: 'x', 2: 'xy', 3: 'xyz'}[T.domain]
    backend = get_backend()

    # Verify if inputs are valid for contraction operation
    validate_tensops(T)
    validate_tensops(S)
    check_index_letter_capacity(
        indices_characters, sum(T.order) + sum(S.order), "contraction"
    )

    if sum(T.order) == 0 or sum(S.order) == 0:
        raise ValueError(
            "contraction() doesn't support scalar operands (order [0, 0]); "
            "use tensor_product for scalar multiplication instead."
        )

    check_indices_list_double(T_contracted_indices)
    check_indices_list_double(S_contracted_indices)

    check_not_repeat_indices(T_contracted_indices)
    check_not_repeat_indices(S_contracted_indices)

    check_contraction_order(T_contracted_indices, S_contracted_indices)

    if T.values.shape[0] != S.values.shape[0]:
        raise ValueError(
            "contraction() requires matching batch sizes: "
            f"T has {T.values.shape[0]}, S has {S.values.shape[0]}."
        )

    if T_contracted_indices[0][0] is not None:
        for a_slot, b_slot in zip(T_contracted_indices[0], S_contracted_indices[1]):
            a_dim = T.codomain[T.contravariant[a_slot]]
            b_dim = S.codomain[S.covariant[b_slot]]
            if a_dim != b_dim:
                raise ValueError(
                    "Contracted codomain dimensions must match: "
                    f"T's contravariant index {a_slot} has size {a_dim}, "
                    f"S's covariant index {b_slot} has size {b_dim}."
                )

    if T_contracted_indices[1][0] is not None:
        for a_slot, b_slot in zip(T_contracted_indices[1], S_contracted_indices[0]):
            a_dim = T.codomain[T.covariant[a_slot]]
            b_dim = S.codomain[S.contravariant[b_slot]]
            if a_dim != b_dim:
                raise ValueError(
                    "Contracted codomain dimensions must match: "
                    f"T's covariant index {a_slot} has size {a_dim}, "
                    f"S's contravariant index {b_slot} has size {b_dim}."
                )

    # Map tensor index positions to einsum character labels for both T and S tensors.
    # Each index in the contravariant/covariant list is mapped to a unique character.
    T_indices_char = [
        ''.join([indices_characters[i] for i in (T.contravariant or [])]),
        ''.join([indices_characters[i] for i in (T.covariant or [])])
    ]
    S_indices_char = [
        ''.join([indices_characters[sum(T.order) + i]
                for i in (S.contravariant or [])]),
        ''.join([indices_characters[sum(T.order) + i]
                for i in (S.covariant or [])])
    ]

    # Identify the einsum characters corresponding to the contracted indices for
    # both tensors. For each contracted index, get the character label from the
    # appropriate position.
    T_contracted_indices_char = [
        [T_indices_char[0][i] for i in T_contracted_indices[0]] if (
            T_contracted_indices[0][0] is not None) else [],
        [T_indices_char[1][i] for i in T_contracted_indices[1]] if (
            T_contracted_indices[1][0] is not None) else []
    ]
    S_contracted_indices_char = [
        [S_indices_char[0][i] for i in S_contracted_indices[0]] if (
            S_contracted_indices[0][0] is not None) else [],
        [S_indices_char[1][i] for i in S_contracted_indices[1]] if (
            S_contracted_indices[1][0] is not None) else []
    ]

    # Sort the einsum character labels alphabetically for consistent ordering.
    # This ensures the mapping between tensor axes and einsum indices is correct.
    T_indices_char = ''.join(sorted(''.join(T_indices_char)))
    S_indices_char = ''.join(sorted(''.join(S_indices_char)))

    # Replace the contracted characters in T's indices with those from S,
    # so that einsum recognizes the contraction between the correct axes.
    if T_contracted_indices[0][0] is not None:
        for i in range(len(T_contracted_indices[0])):
            T_indices_char = T_indices_char.replace(
                T_contracted_indices_char[0][i],
                S_contracted_indices_char[1][i])

    if T_contracted_indices[1][0] is not None:
        for i in range(len(T_contracted_indices[1])):
            T_indices_char = T_indices_char.replace(
                T_contracted_indices_char[1][i],
                S_contracted_indices_char[0][i])

    # Combine the remaining (non-contracted) indices from both tensors,
    # removing any repeated characters, to form the output einsum indices.
    combined_indices_char = combine_characters_delete_repeated(
        T_indices_char, S_indices_char
    )

    # Build the einsum formula for the contraction based on the domain dimension.
    if T.domain == S.domain and T.domain in {1, 2, 3}:
        T_formula = 'n' + T_indices_char + discretization
        S_formula = 'n' + S_indices_char + discretization
        contraction_formula = 'n' + combined_indices_char + discretization
    else:
        raise ValueError("Space dimension not supported.")

    # Perform the contraction using Einstein summation convention.
    einsum_formula = T_formula + ',' + S_formula + '->' + contraction_formula
    contraction_result = backend.einsum(einsum_formula, T.values, S.values)

    # Determine the new positions of the remaining (not contracted) indices for
    # both tensors.
    T_contracted = [
        [T.contravariant[i] for i in T_contracted_indices[0]
         ] if T_contracted_indices[0][0] is not None else [],
        [T.covariant[i] for i in T_contracted_indices[1]
         ] if T_contracted_indices[1][0] is not None else []]

    T_not_contracted = [
        [index for i, index in enumerate(
            T.contravariant) if index not in T_contracted[0]] if T.contravariant
        is not None else [],
        [index for i, index in enumerate(
            T.covariant) if index not in T_contracted[1]] if T.covariant is not
        None else []]

    S_contracted = [
        [S.contravariant[i] for i in S_contracted_indices[0]
         ] if S_contracted_indices[0][0] is not None else [],
        [S.covariant[i] for i in S_contracted_indices[1]
         ] if S_contracted_indices[1][0] is not None else []]

    S_not_contracted = [
        [index for i, index in enumerate(
            S.contravariant) if index not in S_contracted[0]] if S.contravariant
        is not None else [],
        [index for i, index in enumerate(
            S.covariant) if index not in S_contracted[1]] if S.covariant is not
        None else []]

    # Reindex the remaining indices to ensure correct ordering in the output tensor.
    T_not_contracted_sort = reindex_by_sorted_order(T_not_contracted)
    S_not_contracted_sort = reindex_by_sorted_order(S_not_contracted)

    # Offset S's indices so they do not overlap with T's in the output.
    shift = max(T_not_contracted_sort[0] +
                T_not_contracted_sort[1], default=-1) + 1
    new_tensorial_order = [T_not_contracted_sort[0] +
                           [x + shift for x in S_not_contracted_sort[0]],
                           T_not_contracted_sort[1] +
                           [x + shift for x in S_not_contracted_sort[1]]]

    # Parameters of the resultant tensor.
    new_order = [len(new_tensorial_order[0]), len(new_tensorial_order[1])]
    new_contravariance = None if new_tensorial_order[0] == [
    ] else new_tensorial_order[0]
    new_covariance = None if new_tensorial_order[1] == [
    ] else new_tensorial_order[1]

    if new_order == [0, 0]:
        contraction_result = backend.unsqueeze(contraction_result, 1)

    codomain_result = list(contraction_result.shape[1:-T.domain])
    new_codomain = ([codomain_result[0]] if len(
        codomain_result) == 1 else codomain_result)

    # Return the contracted tensor as a new TensOps object.
    return TensOps(
        contraction_result,
        domain_dim=T.domain,
        codomain_dim=new_codomain,
        order=new_order,
        contravariant_indices=new_contravariance,
        covariant_indices=new_covariance,
    )


def self_contraction(
    T: TensOps, contravariant_index: list[int], covariant_index: list[int]
) -> TensOps:
    """Contract a tensor with itself along paired indices.

    Args:
        T (TensOps): Tensor to be self-contracted.
        contravariant_index (list[int]): Contravariant index positions to
            contract.
        covariant_index (list[int]): Covariant index positions to contract.

    Returns:
        TensOps: Tensor with the specified index pairs contracted.

    Raises:
        TypeError: If either index collection is not provided as a list.
        ValueError: If the tensor lacks the required variances, if indices are
            out of range, or if the lists contain duplicates.
    """
    indices_characters = 'ijklmpqrstuvwa'
    backend = get_backend()

    # Validate that T is a TensOps instance
    validate_tensops(T)
    check_index_letter_capacity(
        indices_characters, sum(T.order), "self_contraction"
    )

    # Ensure the tensor has at least one contravariant and one covariant index
    if T.contravariant is None or T.covariant is None:
        raise ValueError(
            "Both contravariant and covariant orders of T must be non-zero for self-contraction.")

    # Check that both index lists are provided as lists
    if not isinstance(contravariant_index, list) or not isinstance(covariant_index, list):
        raise TypeError(
            "Both contravariant_index and covariant_index must be provided as lists.")

    # Ensure the lists of indices to contract have the same length
    if len(contravariant_index) != len(covariant_index):
        raise ValueError(
            "The lengths of contravariant_index and covariant_index must match for self-contraction.")

    # Validate that all indices are within the valid range for the tensor's order
    if any(idx >= T.order[0] or idx < 0 for idx in contravariant_index):
        raise ValueError(
            "All indices in contravariant_index must be within the range [0, T.order[0]).")
    if any(idx >= T.order[1] or idx < 0 for idx in covariant_index):
        raise ValueError(
            "All indices in covariant_index must be within the range [0, T.order[1]).")

    # Ensure there are no duplicate indices in either list
    if len(contravariant_index) != len(set(contravariant_index)):
        raise ValueError("Duplicate entries found in contravariant_index.")
    if len(covariant_index) != len(set(covariant_index)):
        raise ValueError("Duplicate entries found in covariant_index.")

    # Map indices to character labels for einsum notation
    T_indices_char = [
        ''.join([indices_characters[i] for i in (T.contravariant or [])]),
        ''.join([indices_characters[i] for i in (T.covariant or [])])]

    # Get characters to be replaced for contraction
    T_contracted_indices_char = [
        [T_indices_char[0][i]
            for i in contravariant_index] if contravariant_index else [],
        [T_indices_char[1][i]
            for i in covariant_index] if covariant_index else []
    ]

    T_indices_char = ''.join(sorted(''.join(T_indices_char)))

    # Substitute the characters for the contracted indices to be the same for einsum
    if contravariant_index:
        for i in range(len(contravariant_index)):
            T_indices_char = T_indices_char.replace(
                T_contracted_indices_char[0][i],
                T_contracted_indices_char[1][i])

    # Combine indices and filter out duplicates to form the resulting tensor's indices
    combined = ''.join(T_indices_char)
    freq = {char: combined.count(char) for char in set(combined)}
    combined_indices = ''.join(char for char in combined if freq[char] == 1)

    if T.domain in {1, 2, 3}:
        discretization = {1: 'x', 2: 'xy', 3: 'xyz'}[T.domain]
        T_formula = 'n' + T_indices_char + discretization
        contraction_formula = 'n' + combined_indices + discretization
    else:
        raise ValueError(f"Unsupported space dimension: {T.domain}.")

    # Complete formula for Einstein sum and perform the contraction
    einsum_formula = T_formula + '->' + contraction_formula
    contraction_result = backend.einsum(einsum_formula, T.values)

    # Identify which indices were contracted and which remain
    T_contracted = [
        [T.contravariant[i]
            for i in contravariant_index] if contravariant_index and contravariant_index[0]
        is not None else [],
        [T.covariant[i]
            for i in covariant_index] if covariant_index and covariant_index[0]
        is not None else []]

    T_not_contracted = [
        [index for i, index in enumerate(
            T.contravariant) if index not in T_contracted[0]] if T.contravariant
        is not None else [],
        [index for i, index in enumerate(
            T.covariant) if index not in T_contracted[1]] if T.covariant is not
        None else []]

    T_not_contracted_sort = reindex_by_sorted_order(T_not_contracted)

    # Parameters of the resultant tensor
    new_order = [len(T_not_contracted[0]), len(T_not_contracted[1])]
    new_contravariance = None if T_not_contracted_sort[0] == [
    ] else T_not_contracted_sort[0]
    new_covariance = None if T_not_contracted_sort[1] == [
    ] else T_not_contracted_sort[1]

    if new_order == [0, 0]:
        contraction_result = backend.unsqueeze(contraction_result, 1)

    codomain_result = list(contraction_result.shape[1:-T.domain])
    new_codomain = ([codomain_result[0]] if len(
        codomain_result) == 1 else codomain_result)

    # Return the contracted tensor as a TensOps object
    return TensOps(
        contraction_result,
        domain_dim=T.domain,
        codomain_dim=new_codomain,
        order=new_order,
        contravariant_indices=new_contravariance,
        covariant_indices=new_covariance,
    )


def tensor_product(T: TensOps, S: TensOps) -> TensOps:
    """Compute the tensor product of two tensors.

    Args:
        T (TensOps): First tensor operand.
        S (TensOps): Second tensor operand.

    Returns:
        TensOps: Tensor containing the outer product of ``T`` and ``S``.

    Raises:
        ValueError: If the tensors are invalid or their domains are
            incompatible.

    Examples:
        >>> import torch
        >>> from tensopsciml import TensOps
        >>> from tensopsciml.algebra import tensor_product
        >>> a = TensOps(
        ...     torch.rand((100, 2, 10, 10, 10)),
        ...     domain_dim=3,
        ...     codomain_dim=[2],
        ...     order=[1, 0],
        ...     contravariant_indices=[0],
        ... )
        >>> b = TensOps(
        ...     torch.rand((100, 2, 2, 10, 10, 10)),
        ...     domain_dim=3,
        ...     codomain_dim=[2, 2],
        ...     order=[1, 1],
        ...     contravariant_indices=[1],
        ...     covariant_indices=[0],
        ... )
        >>> tensor_product(a, b).order
        [2, 1]
    """
    indices_characters = 'ijklmpqrstuvwabdc'
    backend = get_backend()

    # Validate that both inputs are TensOps instances
    validate_tensops(T)
    validate_tensops(S)

    if T.domain != S.domain:
        raise ValueError("Not compatible dimensions between tensors.")

    # Handle scalar fields: scalar ⊗ T = scalar * T (pointwise multiplication).
    # The scalar has shape (N, 1, *spatial) while T has (N, *codomain, *spatial):
    # its single channel axis must be expanded to one singleton axis per
    # codomain axis of T, otherwise broadcasting right-aligns the scalar's
    # batch axis with a codomain axis of T for rank >= 2.
    if sum(T.order) == 0:
        scalar_values = T.values
        for _ in range(len(S.codomain) - 1):
            scalar_values = backend.unsqueeze(scalar_values, 1)
        return TensOps(
            S.values * scalar_values,
            domain_dim=S.domain,
            codomain_dim=S.codomain,
            order=S.order,
            contravariant_indices=S.contravariant,
            covariant_indices=S.covariant,
        )
    if sum(S.order) == 0:
        scalar_values = S.values
        for _ in range(len(T.codomain) - 1):
            scalar_values = backend.unsqueeze(scalar_values, 1)
        return TensOps(
            T.values * scalar_values,
            domain_dim=T.domain,
            codomain_dim=T.codomain,
            order=T.order,
            contravariant_indices=T.contravariant,
            covariant_indices=T.covariant,
        )
    # Determine the index ranges for contravariant and covariant indices in both
    # tensors
    a_contra_end = T.order[0]
    a_cov_end = a_contra_end + T.order[1]
    b_contra_end = a_cov_end + S.order[0]
    b_cov_end = b_contra_end + S.order[1]

    # Assign einsum characters to each index position for both tensors
    T_indices = [indices_characters[:a_contra_end],
                 indices_characters[a_contra_end:a_cov_end]]
    S_indices = [indices_characters[a_cov_end:b_contra_end],
                 indices_characters[b_contra_end:b_cov_end]]

    # Concatenate all index characters for the output tensor
    combined_indices = ''.join([T_indices[0] + T_indices[1],
                                S_indices[0] + S_indices[1]])

    # Build the einsum formula for the tensor product based on the domain dimension
    if T.domain in {1, 2, 3}:
        discretization = {1: 'x', 2: 'xy', 3: 'xyz'}[T.domain]
        T_formula = 'n' + ''.join(T_indices) + discretization
        S_formula = 'n' + ''.join(S_indices) + discretization
        contraction_formula = 'n' + ''.join(combined_indices) + discretization
    else:
        raise ValueError(f"Unsupported space dimension: {T.domain}.")

    # Perform the tensor product using Einstein summation convention
    einsum_formula = T_formula + ',' + S_formula + '->' + contraction_formula
    tensor_prod = backend.einsum(einsum_formula, T.values, S.values)

    # Compute the resulting contravariant and covariant order. Both operands
    # carry at least one index here: a scalar operand already returned above,
    # so the product can never collapse to a scalar field.
    contravariance_result = T.order[0] + S.order[0]
    covariance_result = T.order[1] + S.order[1]

    codomain_result = list(tensor_prod.shape[1:-T.domain])

    # Compute new contravariant and covariant index lists for the output tensor
    T_contravariant = T.contravariant if T.contravariant is not None else []
    T_covariant = T.covariant if T.covariant is not None else []
    S_contravariant = S.contravariant if S.contravariant is not None else []
    S_covariant = S.covariant if S.covariant is not None else []

    new_contravariant = T_contravariant + \
        [x + sum(T.order) for x in S_contravariant]
    new_covariant = T_covariant + [x + sum(T.order) for x in S_covariant]

    # Return the resulting tensor as a new TensOps object
    return TensOps(
        tensor_prod,
        domain_dim=T.domain,
        codomain_dim=(codomain_result[0] if len(codomain_result) == 1
                      else codomain_result),
        order=[contravariance_result, covariance_result],
        contravariant_indices=None if new_contravariant == [] else new_contravariant,
        covariant_indices=None if new_covariant == [] else new_covariant)


def raise_index(
    inv_metric: MetrOps,
    T: TensOps,
    inv_metric_index: int = -1,
    covariant_index: int = -1,
) -> TensOps:
    """Raise a covariant index of ``T`` to contravariant.

    Contracts one contravariant index of ``inv_metric`` (the inverse metric
    tensor, ``g^{ij}``) with one covariant index of ``T``
    (``g^{ij} T_j = T^i``), turning that index from covariant to
    contravariant.

    Args:
        inv_metric (MetrOps): Inverse (contravariant) metric tensor, with
            order ``[2, 0]``. Obtain it from a covariant :class:`MetrOps`
            instance via ``inverse(metric)``.
        T (TensOps): Tensor on which the index is raised. Must have at least
            one covariant index.
        inv_metric_index (int): Position of the contravariant index of
            ``inv_metric`` to contract. Defaults to ``-1``.
        covariant_index (int): Position of the covariant index of ``T`` to
            raise. Defaults to ``-1`` to raise the last covariant index.

    Returns:
        TensOps: Tensor with one additional contravariant index.

    Raises:
        ValueError: If ``inv_metric`` doesn't have order ``[2, 0]``, or if
            ``T`` lacks covariant indices.

    Examples:
        >>> import torch
        >>> from tensopsciml import TensOps
        >>> from tensopsciml.algebra import raise_index, inverse
        >>> inv_metric = inverse(metric)  # metric: MetrOps, order [0, 2]
        >>> tensor = TensOps(
        ...     torch.rand((100, 2, 2, 10, 10, 10)),
        ...     domain_dim=3,
        ...     codomain_dim=[2, 2],
        ...     order=[0, 2],
        ...     covariant_indices=[0, 1],
        ... )
        >>> raise_index(inv_metric, tensor, covariant_index=0).order
        [1, 1]
    """
    validate_tensops(T)
    validate_metrops(inv_metric)

    if inv_metric.order != [2, 0]:
        raise ValueError(
            "raise_index() expects the inverse (contravariant) metric tensor, "
            f"with order [2, 0]; received order {inv_metric.order}. "
            "Obtain it via inverse(metric)."
        )

    if T.order[1] == 0:
        raise ValueError(
            "Index can't be raised because all indices are already contravariant."
        )

    return contraction(
        inv_metric, T, [[inv_metric_index], [None]], [[None], [covariant_index]]
    )


def lower_index(
    metric: MetrOps,
    T: TensOps,
    metric_index: int = -1,
    contravariant_index: int = -1,
) -> TensOps:
    """Lower a contravariant index of ``T`` to covariant.

    Contracts one covariant index of ``metric`` (the metric tensor,
    ``g_{ij}``) with one contravariant index of ``T``
    (``g_{ij} T^j = T_i``), turning that index from contravariant to
    covariant.

    Args:
        metric (MetrOps): Covariant metric tensor, with order ``[0, 2]``.
        T (TensOps): Tensor on which the index is lowered. Must have at
            least one contravariant index.
        metric_index (int): Position of the covariant index of ``metric``
            to contract. Defaults to ``-1``.
        contravariant_index (int): Position of the contravariant index of
            ``T`` to lower. Defaults to ``-1`` to lower the last
            contravariant index.

    Returns:
        TensOps: Tensor with one additional covariant index.

    Raises:
        ValueError: If ``metric`` doesn't have order ``[0, 2]``, or if
            ``T`` lacks contravariant indices.

    Examples:
        >>> import torch
        >>> from tensopsciml import TensOps
        >>> from tensopsciml.algebra import lower_index
        >>> tensor = TensOps(
        ...     torch.rand((100, 2, 2, 10, 10, 10)),
        ...     domain_dim=3,
        ...     codomain_dim=[2, 2],
        ...     order=[2, 0],
        ...     contravariant_indices=[0, 1],
        ... )
        >>> lower_index(metric, tensor, contravariant_index=0).order
        [1, 1]
    """
    validate_tensops(T)
    validate_metrops(metric)

    if metric.order != [0, 2]:
        raise ValueError(
            "lower_index() expects the covariant metric tensor, with order "
            f"[0, 2]; received order {metric.order}."
        )

    if T.order[0] == 0:
        raise ValueError(
            "Index can't be lowered because all indices are already covariant."
        )

    return contraction(
        metric, T, [[None], [metric_index]], [[contravariant_index], [None]]
    )


def kronecker_delta(metric: MetrOps) -> TensOps:
    """Build the mixed Kronecker delta ``delta^i{}_j`` from a metric tensor.

    Computes the delta as ``delta^i{}_j = g^{ik} g_{kj}`` by inverting the
    metric and contracting the inverse against it. The identity is therefore
    *derived* from ``metric`` rather than fabricated: it automatically
    inherits ``metric``'s grid, dtype and device, and because ``g^{ik}
    g_{kj}`` is the identity for any metric it is exactly the mixed Kronecker
    delta in every coordinate system.

    Args:
        metric (MetrOps): Covariant metric tensor ``g_{ij}`` (order
            ``[0, 2]``).

    Returns:
        TensOps: Mixed Kronecker delta ``delta^i{}_j``, order ``[1, 1]``
            (contravariant index at position 0, covariant index at
            position 1).

    Raises:
        ValueError: If ``metric`` is not a covariant ``MetrOps`` (order
            ``[0, 2]``).
    """
    validate_metrops(metric)

    if metric.order != [0, 2]:
        raise ValueError(
            "kronecker_delta() expects the covariant metric tensor, with order "
            f"[0, 2]; received order {metric.order}."
        )

    inv_metric = inverse(metric)  # g^{ik}, order [2, 0]

    # delta^i{}_j = g^{ik} g_{kj}: contract the inverse metric's second
    # contravariant index (k) with the metric's first covariant index (k),
    # leaving the inverse's i (contravariant) and the metric's j (covariant)
    # -> a mixed tensor of order [1, 1].
    return contraction(inv_metric, metric, [[1], [None]], [[None], [0]])


def symmetrization_covariant(T: TensOps, sym_indices: list[int]) -> TensOps:
    """Symmetrize a tensor over selected covariant indices.

    Args:
        T (TensOps): Tensor to symmetrize.
        sym_indices (list[int]): Positions of the covariant indices to
            symmetrize.

    Returns:
        TensOps: Symmetrized tensor with the same variance structure as ``T``.

    Raises:
        ValueError: If ``sym_indices`` does not have exactly 2 entries, or if
            ``T`` has no covariant indices.
    """
    if len(sym_indices) != 2:
        raise ValueError(
            "symmetrization_covariant() requires exactly 2 indices to "
            f"symmetrize; received {len(sym_indices)}.")

    validate_tensops(T)

    if T.covariant is None:
        raise ValueError(
            "symmetrization_covariant() requires T to have covariant "
            f"indices; received covariant=None (order {T.order}).")

    backend = get_backend()

    T_cov_indices = [i + 1 for i in T.covariant]
    indices_to_permute = [T_cov_indices[i] for i in sym_indices]
    permutations_all = list(permutations(indices_to_permute))

    T_sym = backend.zeros_like(T.values)
    indices_original = list(range(0, len(T.values.shape)))

    for permutation_i in permutations_all:
        indices_permutated = indices_original.copy()
        values_dict = {i: indices_permutated[i] for i in indices_to_permute}
        permuted_values = [values_dict[i] for i in permutation_i]

        for idx, val in zip(indices_to_permute, permuted_values):
            indices_permutated[idx] = val

        T_sym += backend.permute(T.values, tuple(indices_permutated))

    T_sym = 1/math.factorial(len(sym_indices)) * T_sym

    return TensOps(
        T_sym,
        domain_dim=T.domain,
        codomain_dim=T.codomain,
        order=T.order,
        contravariant_indices=T.contravariant,
        covariant_indices=T.covariant,
    )


def symmetrization_contravariant(T: TensOps, sym_indices: list[int]) -> TensOps:
    """Symmetrize a tensor over selected contravariant indices.

    Args:
        T (TensOps): Tensor to symmetrize.
        sym_indices (list[int]): Positions of the contravariant indices to
            symmetrize.

    Returns:
        TensOps: Symmetrized tensor with the same variance structure as ``T``.

    Raises:
        ValueError: If ``sym_indices`` does not have exactly 2 entries, or if
            ``T`` has no contravariant indices.
    """
    if len(sym_indices) != 2:
        raise ValueError(
            "symmetrization_contravariant() requires exactly 2 indices to "
            f"symmetrize; received {len(sym_indices)}.")

    validate_tensops(T)

    if T.contravariant is None:
        raise ValueError(
            "symmetrization_contravariant() requires T to have "
            f"contravariant indices; received contravariant=None (order "
            f"{T.order}).")

    backend = get_backend()

    T_contra_indices = [i + 1 for i in T.contravariant]
    indices_to_permute = [T_contra_indices[i] for i in sym_indices]
    permutations_all = list(permutations(indices_to_permute))

    T_sym = backend.zeros_like(T.values)
    indices_original = list(range(0, len(T.values.shape)))

    for permutation_i in permutations_all:
        indices_permutated = indices_original.copy()
        values_dict = {i: indices_permutated[i] for i in indices_to_permute}
        permuted_values = [values_dict[i] for i in permutation_i]

        for idx, val in zip(indices_to_permute, permuted_values):
            indices_permutated[idx] = val

        T_sym += backend.permute(T.values, tuple(indices_permutated))

    T_sym = 1/math.factorial(len(sym_indices)) * T_sym

    return TensOps(
        T_sym,
        domain_dim=T.domain,
        codomain_dim=T.codomain,
        order=T.order,
        contravariant_indices=T.contravariant,
        covariant_indices=T.covariant,
    )


def antisymmetrization_covariant(T: TensOps, sym_indices: list[int]) -> TensOps:
    """Antisymmetrize a tensor over selected covariant indices.

    Args:
        T (TensOps): Tensor to antisymmetrize.
        sym_indices (list[int]): Positions of the covariant indices to
            antisymmetrize, given in ascending order.

    Returns:
        TensOps: Antisymmetrized tensor with the same variance structure as
            ``T``.

    Raises:
        ValueError: If ``sym_indices`` does not have exactly 2 entries, if it
            is not sorted in ascending order, or if ``T`` has no covariant
            indices.
    """
    if len(sym_indices) != 2:
        raise ValueError(
            "antisymmetrization_covariant() requires exactly 2 indices to "
            f"antisymmetrize; received {len(sym_indices)}.")
    if sym_indices != sorted(sym_indices):
        raise ValueError(
            "antisymmetrization_covariant() requires sym_indices in "
            "ascending order: the permutation sign is computed relative to "
            "the sorted order, so an unsorted input would silently flip the "
            "global sign of the result.")

    validate_tensops(T)

    if T.covariant is None:
        raise ValueError(
            "antisymmetrization_covariant() requires T to have covariant "
            f"indices; received covariant=None (order {T.order}).")

    backend = get_backend()

    T_cov_indices = [i + 1 for i in T.covariant]
    indices_to_permute = [T_cov_indices[i] for i in sym_indices]
    permutations_all = list(permutations(indices_to_permute))

    T_sym = backend.zeros_like(T.values)
    indices_original = list(range(0, len(T.values.shape)))

    for permutation_i in permutations_all:
        indices_permutated = indices_original.copy()
        values_dict = {i: indices_permutated[i] for i in indices_to_permute}
        permuted_values = [values_dict[i] for i in permutation_i]

        for idx, val in zip(indices_to_permute, permuted_values):
            indices_permutated[idx] = val

        T_sym += (permutation_sign(list(permutation_i)) *
                  backend.permute(T.values, tuple(indices_permutated)))

    T_sym = 1/math.factorial(len(sym_indices)) * T_sym

    return TensOps(
        T_sym,
        domain_dim=T.domain,
        codomain_dim=T.codomain,
        order=T.order,
        contravariant_indices=T.contravariant,
        covariant_indices=T.covariant,
    )


def antisymmetrization_contravariant(
    T: TensOps, sym_indices: list[int]
) -> TensOps:
    """Antisymmetrize a tensor over selected contravariant indices.

    Args:
        T (TensOps): Tensor to antisymmetrize.
        sym_indices (list[int]): Positions of the contravariant indices to
            antisymmetrize, given in ascending order.

    Returns:
        TensOps: Antisymmetrized tensor with the same variance structure as
            ``T``.

    Raises:
        ValueError: If ``sym_indices`` does not have exactly 2 entries, if it
            is not sorted in ascending order, or if ``T`` has no
            contravariant indices.
    """
    if len(sym_indices) != 2:
        raise ValueError(
            "antisymmetrization_contravariant() requires exactly 2 indices "
            f"to antisymmetrize; received {len(sym_indices)}.")
    if sym_indices != sorted(sym_indices):
        raise ValueError(
            "antisymmetrization_contravariant() requires sym_indices in "
            "ascending order: the permutation sign is computed relative to "
            "the sorted order, so an unsorted input would silently flip the "
            "global sign of the result.")

    validate_tensops(T)

    if T.contravariant is None:
        raise ValueError(
            "antisymmetrization_contravariant() requires T to have "
            f"contravariant indices; received contravariant=None (order "
            f"{T.order}).")

    backend = get_backend()

    T_contra_indices = [i + 1 for i in T.contravariant]
    indices_to_permute = [T_contra_indices[i] for i in sym_indices]
    permutations_all = list(permutations(indices_to_permute))

    T_sym = backend.zeros_like(T.values)
    indices_original = list(range(0, len(T.values.shape)))

    for permutation_i in permutations_all:
        indices_permutated = indices_original.copy()
        values_dict = {i: indices_permutated[i] for i in indices_to_permute}
        permuted_values = [values_dict[i] for i in permutation_i]

        for idx, val in zip(indices_to_permute, permuted_values):
            indices_permutated[idx] = val

        T_sym += (permutation_sign(list(permutation_i)) *
                  backend.permute(T.values, tuple(indices_permutated)))

    T_sym = 1/math.factorial(len(sym_indices)) * T_sym

    return TensOps(
        T_sym,
        domain_dim=T.domain,
        codomain_dim=T.codomain,
        order=T.order,
        contravariant_indices=T.contravariant,
        covariant_indices=T.covariant,
    )


def symmetrize(T: TensOps, indices: list[int], order: str) -> TensOps:
    """Symmetrize a tensor over specified indices.

    Args:
        T (TensOps): Tensor to symmetrize.
        indices (list[int]): Index positions to symmetrize over.
        order (str): Either ``"contravariant"`` or ``"covariant"``.

    Returns:
        TensOps: Symmetrized tensor.

    Raises:
        ValueError: If ``order`` is not ``"contravariant"`` or ``"covariant"``.
    """
    if order == 'contravariant':
        return symmetrization_contravariant(T, sym_indices=indices)
    elif order == 'covariant':
        return symmetrization_covariant(T, sym_indices=indices)
    else:
        raise ValueError(
            f"symmetrize() requires order to be 'covariant' or "
            f"'contravariant'; received {order!r}.")


def antisymmetrize(T: TensOps, indices: list[int], order: str) -> TensOps:
    """Antisymmetrize a tensor over specified indices.

    Args:
        T (TensOps): Tensor to antisymmetrize.
        indices (list[int]): Index positions to antisymmetrize over.
        order (str): Either ``"contravariant"`` or ``"covariant"``.

    Returns:
        TensOps: Antisymmetrized tensor.

    Raises:
        ValueError: If ``order`` is not ``"contravariant"`` or ``"covariant"``.
    """
    if order == 'contravariant':
        return antisymmetrization_contravariant(T, sym_indices=indices)
    elif order == 'covariant':
        return antisymmetrization_covariant(T, sym_indices=indices)
    else:
        raise ValueError(
            f"antisymmetrize() requires order to be 'covariant' or "
            f"'contravariant'; received {order!r}.")


def cross_prod(u: TensOps, v: TensOps, E: TensOps | None = None) -> TensOps:
    """Compute the cross product of two vectors.

    For 2-component vectors, the cross product is the exterior product
    ``u1*v2 - u2*v1`` (a scalar) and needs no Levi-Civita tensor. For
    3-component vectors, it's ``(u x v)_i = E_{ijk} u^j v^k``: contracting
    two of the three covariant indices of the Levi-Civita tensor ``E``
    (order ``[0, 3]``, see :func:`levi_civita_tensor`) against ``u`` and
    ``v`` leaves its third, covariant index free, so the result is a
    covector (order ``[0, 1]``).

    Args:
        u (TensOps): First vector, represented as a first-order contravariant
            tensor.
        v (TensOps): Second vector, represented as a first-order contravariant
            tensor.
        E (TensOps | None): Covariant Levi-Civita tensor (order ``[0, 3]``),
            required when ``u``/``v`` have 3 components. Must be omitted
            for 2-component vectors.

    Returns:
        TensOps: Tensor containing the cross product result.

    Raises:
        ValueError: If the inputs are not vectors, their codomain dimensions
            are not 2 or 3, ``E`` is missing for 3-component vectors, or
            ``E`` is given for 2-component vectors.
    """
    # Validate that both inputs are valid TensOps vectors
    validate_tensops(u)
    validate_tensops(v)

    # Ensure both tensors are first-order (vectors)
    if u.order != [1, 0] or v.order != [1, 0]:
        raise ValueError(
            "To compute cross products both tensors must be vectors (1st order tensors)."
        )

    if u.codomain[0] == v.codomain[0] == 2:
        if E is not None:
            raise ValueError(
                "cross_prod doesn't use a Levi-Civita tensor for "
                "2-component vectors; omit the 'E' argument."
            )
        backend = get_backend()
        result = backend.unsqueeze(
            u.values[:, 0] * v.values[:, 1] - u.values[:, 1] * v.values[:, 0], 1
        )
        return TensOps(
            result,
            domain_dim=u.domain,
            codomain_dim=[1],
            order=[0, 0],
            contravariant_indices=None,
            covariant_indices=None,
        )
    elif u.codomain[0] == v.codomain[0] == 3:
        if E is None:
            raise ValueError(
                "cross_prod() requires the covariant Levi-Civita tensor 'E' "
                "for 3-component vectors; see levi_civita_tensor."
            )
        return contraction(contraction(E, u, [[None], [1]], [[0], [None]]),
                           v, [[None], [1]], [[0], [None]])
    else:
        raise ValueError(
            "To compute cross product vectors codomain must be 2 or 3.")


def levi_civita_tensor(
    lc_symbol: Any,
    domain_dim: int,
    codomain_dim: list[int],
    order: list[int],
    metric: MetrOps | None = None,
    inv_metric: MetrOps | None = None,
) -> TensOps:
    """Convert a Levi-Civita symbol into a genuine Levi-Civita tensor.

    The raw symbol (see :func:`levi_civita_symbol`) is a tensor density,
    not a tensor: it only transforms correctly once scaled by a volume
    form. Exactly one of ``metric`` or ``inv_metric`` must be given, and
    it decides both the scaling factor and the variance of the result:

    * ``metric`` (covariant, ``g_{ij}``): result is fully covariant,
      order ``[0, N]``, scaled by ``sqrt(det(g))``.
    * ``inv_metric`` (contravariant, ``g^{ij}``): result is fully
      contravariant, order ``[N, 0]``, scaled by
      ``sqrt(det(g^{ij})) = 1 / sqrt(det(g))``.

    Args:
        lc_symbol (Any): Raw backend array produced by
            :func:`levi_civita_symbol`.
        domain_dim (int): Domain dimension of ``lc_symbol``.
        codomain_dim (list[int]): Codomain of ``lc_symbol``, e.g.
            ``[3, 3, 3]`` for a 3rd-order symbol in a 3D space.
        order (list[int]): Two-element list whose entries sum to ``N``,
            the number of indices of the Levi-Civita symbol.
        metric (MetrOps | None): Covariant metric tensor, with order
            ``[0, 2]``. Mutually exclusive with ``inv_metric``.
        inv_metric (MetrOps | None): Contravariant (inverse) metric
            tensor, with order ``[2, 0]``. Mutually exclusive with
            ``metric``.

    Returns:
        TensOps: Levi-Civita tensor, scaled by the corresponding volume
            form, with the variance matching whichever metric was given.

    Raises:
        ValueError: If ``metric``/``inv_metric`` are both given or both
            omitted, if the one given doesn't have the expected order, or
            if the scaled symbol, ``domain_dim``, ``codomain_dim``, or
            ``order`` are mutually inconsistent (see :class:`TensOps`).
    """
    if (metric is None) == (inv_metric is None):
        raise ValueError(
            "levi_civita_tensor() requires exactly one of 'metric' or "
            "'inv_metric' (not both, not neither)."
        )

    n_indices = sum(order)

    if metric is not None:
        validate_metrops(metric)
        if metric.order != [0, 2]:
            raise ValueError(
                "levi_civita_tensor() expects 'metric' to be the covariant "
                f"metric tensor, with order [0, 2]; received order "
                f"{metric.order}."
            )
        volume_form = VolFormOps(metric)
        tensor_order = [0, n_indices]
        contravariant_indices = None
        covariant_indices = list(range(n_indices))
    else:
        validate_metrops(inv_metric)
        if inv_metric.order != [2, 0]:
            raise ValueError(
                "levi_civita_tensor() expects 'inv_metric' to be the "
                "contravariant (inverse) metric tensor, with order "
                f"[2, 0]; received order {inv_metric.order}."
            )
        volume_form = VolFormOps(inv_metric)
        tensor_order = [n_indices, 0]
        contravariant_indices = list(range(n_indices))
        covariant_indices = None

    # lc_symbol is only a tensor density: the volume form is what actually
    # makes it transform as a tensor. Broadcast it (order [0, 0]) across
    # lc_symbol's codomain axes manually, the same way TensOps.__mul__
    # would, since lc_symbol isn't wrapped as a TensOps until it is an
    # actual tensor.
    backend = get_backend()
    volume_values = volume_form.values
    for _ in range(len(codomain_dim) - 1):
        volume_values = backend.unsqueeze(volume_values, 1)

    return TensOps(
        volume_values * lc_symbol,
        domain_dim=domain_dim,
        codomain_dim=codomain_dim,
        order=tensor_order,
        contravariant_indices=contravariant_indices,
        covariant_indices=covariant_indices,
    )
