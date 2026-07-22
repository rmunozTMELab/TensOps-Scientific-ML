"""Tests for indices.py."""

import unittest

from tensopsciml.utils.indices import (
    check_contraction_order,
    check_index_letter_capacity,
    check_indices_list_double,
    check_not_repeat_indices,
    combine_characters_delete_repeated,
    levi_civita_indices_value,
    permutation_sign,
    reindex_by_sorted_order,
)


class TestIndices(unittest.TestCase):
    """Test suite covering index validation and permutation helpers."""

    def test_reindex_by_sorted_order(self) -> None:
        """Ensure reindex_by_sorted_order maps values to sorted ranks."""
        # Simple case.
        self.assertEqual(reindex_by_sorted_order([[3, 1], [2, 4]]), [[2, 0], [1, 3]])
        # With duplicates.
        self.assertEqual(reindex_by_sorted_order([[10, 10], [5]]), [[1, 1], [0]])
        # Error case.
        with self.assertRaises(TypeError):
            reindex_by_sorted_order("not a list")

    def test_check_indices_list_double_valid(self) -> None:
        """Ensure check_indices_list_double accepts well-formed nested lists."""
        valid_lists = [
            [[], []],
            [[1], [2]],
            [[1, 2], [3]],
            [[None], [None]],
        ]
        for indices in valid_lists:
            with self.subTest(indices=indices):
                try:
                    check_indices_list_double(indices)
                except ValueError:
                    self.fail(
                        f"check_indices_list_double() raised ValueError for {indices}"
                    )

    def test_check_indices_list_double_invalid(self) -> None:
        """Ensure check_indices_list_double rejects malformed nested lists."""
        invalid_lists = [
            [],  # Empty.
            [[]],  # Only one inner list.
            [[1], [2], [3]],  # Three inner lists.
            "not a list",  # Not a list.
            [[1], "not a list"],  # Inner element is not a list.
        ]
        for indices in invalid_lists:
            with self.subTest(indices=indices):
                with self.assertRaises(ValueError):
                    check_indices_list_double(indices)

    def test_combine_characters_delete_repeated(self) -> None:
        """Ensure combine_characters_delete_repeated removes duplicated labels."""
        self.assertEqual(combine_characters_delete_repeated("abc", "bde"), "acde")
        self.assertEqual(combine_characters_delete_repeated("i", "i"), "")
        self.assertEqual(combine_characters_delete_repeated("a", "b"), "ab")
        self.assertEqual(combine_characters_delete_repeated("aa", ""), "")

    def test_check_contraction_order_valid(self) -> None:
        """Ensure check_contraction_order accepts compatible index counts."""
        # Case 1: Matching contravariant/covariant counts.
        # First: 2 contravariant, 1 covariant.
        # Second: 1 contravariant, 2 covariant.
        first = [[1, 2], [3]]
        second = [[4], [5, 6]]
        try:
            check_contraction_order(first, second)
        except ValueError:
            self.fail("check_contraction_order() raised ValueError.")

    def test_check_contraction_order_invalid(self) -> None:
        """Ensure check_contraction_order rejects mismatched index counts."""
        # First: 1 contravariant.
        # Second: 0 covariant (should be 1).
        first = [[1], []]
        second = [[], []]
        with self.assertRaises(ValueError):
            check_contraction_order(first, second)

    def test_check_not_repeat_indices_no_repeats(self) -> None:
        """Ensure check_not_repeat_indices accepts non-repeating groups."""
        valid_groups = [
            [[1, 2], [3, 4]],
            [[1], [2]],
            [[], []],
        ]
        for group in valid_groups:
            with self.subTest(group=group):
                try:
                    check_not_repeat_indices(group)
                except ValueError:
                    self.fail(f"check_not_repeat_indices() raised ValueError for {group}")

    def test_check_not_repeat_indices_with_repeats(self) -> None:
        """Ensure check_not_repeat_indices rejects repeated entries."""
        invalid_groups = [
            [[1, 1], [2]],  # Repeated in first group.
            [[1], [2, 2]],  # Repeated in second group.
            [[1, 2, 1], []],  # Repeated in first group.
        ]
        for group in invalid_groups:
            with self.subTest(group=group):
                with self.assertRaises(ValueError):
                    check_not_repeat_indices(group)

    def test_permutation_sign_even(self) -> None:
        """Ensure permutation_sign returns 1 for even permutations."""
        # [0, 1, 2] -> 0 swaps -> +1.
        self.assertEqual(permutation_sign([0, 1, 2]), 1)
        # [2, 0, 1] -> 2 swaps (0,1,2 -> 2,1,0 -> 2,0,1) -> +1.
        self.assertEqual(permutation_sign([2, 0, 1]), 1)

    def test_permutation_sign_odd(self) -> None:
        """Ensure permutation_sign returns -1 for odd permutations."""
        # [1, 0, 2] -> 1 swap -> -1.
        self.assertEqual(permutation_sign([1, 0, 2]), -1)
        # [0, 2, 1] -> 1 swap -> -1.
        self.assertEqual(permutation_sign([0, 2, 1]), -1)

    def test_check_contraction_order_rejects_non_lists(self) -> None:
        """Index specifications must be lists, not other sequences."""

        with self.assertRaises(TypeError):
            check_contraction_order("not a list", [[], []])
        with self.assertRaises(TypeError):
            check_contraction_order([[], []], ([], []))

    def test_check_not_repeat_indices_rejects_non_lists(self) -> None:
        """Index groups must be a list of lists."""

        with self.assertRaises(TypeError):
            check_not_repeat_indices("not a list")
        with self.assertRaises(TypeError):
            check_not_repeat_indices([(0, 1)])

    def test_permutation_sign_rejects_non_lists(self) -> None:
        """permutation_sign only accepts a list of integers."""

        with self.assertRaises(TypeError):
            permutation_sign((0, 1, 2))

    def test_check_index_letter_capacity_within_monotonic_prefix(self) -> None:
        """Ensure index counts within the bank's monotonic prefix are accepted."""
        # 'ijklmpqrstuvwa' is monotonically increasing for its first 13
        # characters ('i' through 'w'); the 14th ('a') breaks the ordering.
        bank = "ijklmpqrstuvwa"
        for n_indices in (0, 1, 4, 13):
            with self.subTest(n_indices=n_indices):
                try:
                    check_index_letter_capacity(bank, n_indices, "contraction")
                except ValueError:
                    self.fail(
                        f"check_index_letter_capacity() rejected {n_indices} indices."
                    )

    def test_check_index_letter_capacity_beyond_monotonic_prefix(self) -> None:
        """Ensure exceeding the bank's monotonic prefix raises instead of contracting silently."""
        bank = "ijklmpqrstuvwa"
        with self.assertRaisesRegex(ValueError, "at most 13 combined"):
            check_index_letter_capacity(bank, 14, "contraction")

    def test_check_index_letter_capacity_names_the_caller(self) -> None:
        """Ensure the error message names the calling function."""
        with self.assertRaisesRegex(ValueError, "self_contraction"):
            check_index_letter_capacity("ijklmpqrstuvwa", 14, "self_contraction")

    def test_check_index_letter_capacity_fully_monotonic_bank(self) -> None:
        """A fully monotonic bank's capacity is its whole length."""
        check_index_letter_capacity("abc", 3, "contraction")
        with self.assertRaisesRegex(ValueError, "at most 3 combined"):
            check_index_letter_capacity("abc", 4, "contraction")

    def test_levi_civita_indices_value(self) -> None:
        """Ensure levi_civita_indices_value returns the expected sign or zero."""
        # Even permutation.
        self.assertEqual(levi_civita_indices_value([0, 1, 2]), 1)
        # Odd permutation.
        self.assertEqual(levi_civita_indices_value([1, 0, 2]), -1)
        # Repeated index.
        self.assertEqual(levi_civita_indices_value([1, 1, 2]), 0)
        # Type error.
        with self.assertRaises(TypeError):
            levi_civita_indices_value("not a list")


if __name__ == "__main__":
    unittest.main()
