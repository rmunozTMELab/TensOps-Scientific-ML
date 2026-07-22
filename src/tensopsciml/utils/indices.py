"""Index validation utilities supporting tensor contraction helpers."""

from collections import Counter


def reindex_by_sorted_order(nested_lists: list[list[int]]) -> list[list[int]]:
    """Reindex nested integer lists based on sorted unique values.

    Args:
        nested_lists (list[list[int]]): List containing inner lists of integers.

    Returns:
        list[list[int]]: Nested list with identical shape whose values
            correspond to the rank of each original value within the sorted
            set of unique values.

    Raises:
        TypeError: If ``nested_lists`` is not a list of lists.

    Examples:
        >>> reindex_by_sorted_order([[3, 1], [2, 4]])
        [[2, 0], [1, 3]]
    """

    if not isinstance(nested_lists, list) or not all(isinstance(group, list) for group in nested_lists):
        raise TypeError("`nested_lists` must be a list of lists of integers.")

    unique_values = sorted(
        {item for sublist in nested_lists for item in sublist})
    value_to_index = {value: index for index,
                      value in enumerate(unique_values)}
    return [[value_to_index[item] for item in sublist] for sublist in nested_lists]


def check_indices_list_double(indices_list: list[list[int]]) -> None:
    """Validate that nested contraction indices contain two sublists.

    Args:
        indices_list (list[list[int]]): Nested list describing contraction indices.

    Raises:
        ValueError: If the provided structure is not shaped like ``[[], []]``.
    """

    if not isinstance(indices_list, list) or len(indices_list) != 2 or not all(
        isinstance(group, list) for group in indices_list
    ):
        raise ValueError(
            "Indices to be contracted must be provided as a list shaped [[], []].")


def combine_characters_delete_repeated(first: str, second: str) -> str:
    """Concatenate index labels and remove those that appear twice.

    Args:
        first (str): Index labels for the first tensor.
        second (str): Index labels for the second tensor.

    Returns:
        str: String containing only the labels that appear exactly once
            across both inputs.
    """

    counter = Counter(f"{first}{second}")
    return "".join(char for char in f"{first}{second}" if counter[char] == 1)


def check_contraction_order(first: list[list[int]], second: list[list[int]]) -> None:
    """Ensure index counts are compatible for a double contraction.

    Args:
        first (list[list[int]]): Index specification for the first tensor.
        second (list[list[int]]): Index specification for the second tensor.

    Raises:
        TypeError: If ``first`` or ``second`` is not a list.
        ValueError: If the number of contravariant versus covariant indices
            is inconsistent.
    """

    if not isinstance(first, list) or not isinstance(second, list):
        raise TypeError(
            "Index specifications must be provided as lists of lists.")

    first_counts = [len([value for value in group if value is not None])
                    for group in first]
    second_counts = [len([value for value in group if value is not None])
                     for group in second]
    if [first_counts[0] - second_counts[1], first_counts[1] - second_counts[0]] != [0, 0]:
        raise ValueError(
            "Contravariant and covariant indices must be compatible for double contraction.")


def check_not_repeat_indices(index_groups: list[list[int]]) -> None:
    """Verify that index positions do not repeat within the same variance group.

    Args:
        index_groups (list[list[int]]): Collection of index lists grouped by
            variance type.

    Raises:
        TypeError: If ``index_groups`` is not a list of lists.
        ValueError: If any sublist contains duplicate values.
    """

    if not isinstance(index_groups, list) or not all(isinstance(group, list) for group in index_groups):
        raise TypeError("Index groups must be provided as a list of lists.")

    for sublist in index_groups:
        if len(sublist) != len(set(sublist)):
            raise ValueError("A tensor index cannot be contracted twice.")


def check_index_letter_capacity(
    letter_bank: str, n_indices: int, caller: str
) -> None:
    """Guard against silent index misassignment beyond the bank's safe range.

    ``contraction()`` and ``self_contraction()`` assign one character of
    ``letter_bank`` per tensor index (by position), then call
    ``''.join(sorted(...))`` on those characters to build both the ``einsum``
    formula and (independently, via :func:`reindex_by_sorted_order`) the
    output's index metadata. Both steps silently assume that alphabetically
    sorting the assigned characters reproduces the original position order --
    true only while the bank's character-to-position mapping is itself
    monotonically increasing. ``'ijklmpqrstuvwa'`` (the bank currently used by
    both callers) satisfies this for its first 13 characters (``i`` through
    ``w``) but breaks at the 14th (``'a'``, which alphabetically precedes all
    the others). Past that point ``einsum`` contracts whichever axis the
    *sorted* position happens to land on -- not the one the caller asked for
    -- silently returning wrong values instead of raising.

    No tensor in this library's physical domain currently has anywhere near
    13 free/contracted indices in a single call (the largest tensors here are
    rank 3-4), so this never fires today. It exists purely so that if a
    future higher-order use case ever approaches the limit, it fails loudly
    here instead of silently computing a wrong contraction. TODO: replace the
    bank with a character set that is monotonic by construction (see project
    discussion), which would remove this ceiling entirely.

    Args:
        letter_bank (str): The einsum letter bank used by the caller.
        n_indices (int): Number of tensor indices being assigned characters
            from ``letter_bank`` in this call (e.g. ``sum(T.order) +
            sum(S.order)`` for :func:`~tensopsciml.algebra.contraction`).
        caller (str): Name of the calling function, for the error message.

    Raises:
        ValueError: If ``n_indices`` exceeds the bank's monotonic prefix
            length, where the sort-based axis reordering is no longer
            guaranteed safe.
    """

    safe_capacity = len(letter_bank)
    for i in range(1, len(letter_bank)):
        if letter_bank[i] <= letter_bank[i - 1]:
            safe_capacity = i
            break

    if n_indices > safe_capacity:
        raise ValueError(
            f"{caller}() supports at most {safe_capacity} combined "
            "free/contracted tensor indices in a single call; received "
            f"{n_indices}. Beyond this the internal einsum index-letter bank "
            f"({letter_bank!r}) is no longer guaranteed order-preserving "
            "under alphabetical sorting, which can silently contract the "
            "wrong axis instead of raising. This is a known limitation "
            "(tracked for a future fix replacing the bank with a "
            "monotonic one); no tensor in this library's physical domain "
            "currently reaches this order."
        )


def permutation_sign(indices: list[int]) -> int:
    """Return the sign of the permutation encoded by ``indices``.

    Args:
        indices (list[int]): List describing a permutation.

    Returns:
        int: ``+1`` for even permutations and ``-1`` for odd permutations.

    Raises:
        TypeError: If ``indices`` is not a list.
    """

    if not isinstance(indices, list):
        raise TypeError("Indices must be provided as a list of integers.")

    permutation = sorted(range(len(indices)), key=indices.__getitem__)
    sign = 1
    for i in range(len(permutation)):
        for j in range(i + 1, len(permutation)):
            if permutation[i] > permutation[j]:
                sign *= -1
    return sign


def levi_civita_indices_value(indices: list[int]) -> int:
    """Evaluate the Levi-Civita symbol for the provided indices.

    Args:
        indices (list[int]): Candidate index list.

    Returns:
        int: ``0`` if any index repeats, otherwise the permutation sign.

    Raises:
        TypeError: If ``indices`` is not a list.
    """
    if not isinstance(indices, list):
        raise TypeError("Indices must be provided as a list of integers.")

    if len(set(indices)) != len(indices):
        return 0
    return permutation_sign(indices)


__all__ = [
    "reindex_by_sorted_order",
    "check_indices_list_double",
    "combine_characters_delete_repeated",
    "check_contraction_order",
    "check_not_repeat_indices",
    "check_index_letter_capacity",
    "permutation_sign",
    "levi_civita_indices_value",
]
