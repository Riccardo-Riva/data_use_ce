"""Tests for decisions and mistake counts, on hand-computed examples."""

import numpy as np
import pytest

from data_use_ce.evaluation import (
    admit_above_threshold,
    admit_top_fraction,
    count_mistakes,
    decision_table,
    random_selection_table,
)

# Three excellent students, then four others. Of the excellent ones, the first and third are
# accepted; of the others, the first is accepted.
REAL = [True, True, True, False, False, False, False]
ADMITTED = [True, False, True, True, False, False, False]


def test_decision_table_counts_each_cell():
    table = decision_table(REAL, ADMITTED)
    assert table.to_numpy().tolist() == [[2, 1], [1, 3]]
    assert list(table.index) == ["Excellent", "Not excellent"]
    assert list(table.columns) == ["Accepted", "Rejected"]


def test_count_mistakes_gives_counts_and_rates():
    mistakes = count_mistakes(REAL, ADMITTED)
    assert mistakes["good_rejected"] == 1
    assert mistakes["bad_accepted"] == 1
    assert mistakes["n_excellent"] == 3
    assert mistakes["n_not_excellent"] == 4
    assert mistakes["good_rejected_rate"] == pytest.approx(1 / 3)
    assert mistakes["bad_accepted_rate"] == pytest.approx(1 / 4)


def test_count_mistakes_rate_is_nan_when_a_class_is_empty():
    mistakes = count_mistakes([False, False], [True, False])
    assert np.isnan(mistakes["good_rejected_rate"])
    assert mistakes["bad_accepted_rate"] == pytest.approx(0.5)


def test_admit_top_fraction_takes_the_highest_scores():
    admitted = admit_top_fraction([0.9, 0.1, 0.5, 0.7], 0.5)
    assert admitted.tolist() == [True, False, False, True]


@pytest.mark.parametrize(("fraction", "expected"), [(0.0, 0), (1.0, 4)])
def test_admit_top_fraction_extremes(fraction, expected):
    assert admit_top_fraction([0.9, 0.1, 0.5, 0.7], fraction).sum() == expected


def test_admit_top_fraction_breaks_ties_at_random_but_reproducibly():
    scores = [0.9, 0.5, 0.5, 0.1]  # one of the two tied students gets the last place
    picks = {tuple(admit_top_fraction(scores, 0.5, seed=s)) for s in range(20)}
    assert picks == {(True, True, False, False), (True, False, True, False)}
    again = admit_top_fraction(scores, 0.5, seed=3)
    assert np.array_equal(again, admit_top_fraction(scores, 0.5, seed=3))


def test_admit_top_fraction_with_all_scores_equal_still_admits_the_right_number():
    assert admit_top_fraction([0.3] * 10, 0.4).sum() == 4


@pytest.mark.parametrize("fraction", [-0.1, 1.1])
def test_admit_top_fraction_rejects_bad_fractions(fraction):
    with pytest.raises(ValueError, match="between 0 and 1"):
        admit_top_fraction([0.5, 0.6], fraction)


def test_admit_above_threshold_includes_the_threshold():
    assert admit_above_threshold([0.4, 0.5, 0.6]).tolist() == [False, True, True]
    assert admit_above_threshold([0.4, 0.5, 0.6], 0.55).tolist() == [False, False, True]


def test_random_selection_table_gives_expected_counts():
    real = [True] * 4 + [False] * 6
    table = random_selection_table(real, n_admitted=5)
    # 5 picks out of 10 students: half of the 4 excellent and half of the 6 others.
    assert table.to_numpy().tolist() == [[2.0, 2.0], [3.0, 3.0]]
