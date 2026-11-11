"""Turning model scores into accept/reject decisions, and counting the mistakes."""

from collections.abc import Sequence

import numpy as np
import pandas as pd

OUTCOMES = ["Excellent", "Not excellent"]
DECISIONS = ["Accepted", "Rejected"]


def admit_top_fraction(scores: Sequence[float], fraction: float, seed: int = 0) -> np.ndarray:
    """Accepts the students with the highest scores.

    Students with equal scores (common with shallow trees) are ordered at random with a fixed
    seed, so that nobody is favoured by their position in the table.

    Args:
        scores: Model score of each student, higher meaning more likely excellent.
        fraction: Share of students to accept, from 0 to 1.
        seed: Seed of the tie-breaking.

    Returns:
        True for each accepted student.

    Raises:
        ValueError: If the fraction is outside 0..1.
    """
    if not 0 <= fraction <= 1:
        raise ValueError(f"fraction must be between 0 and 1, got {fraction}")
    scores = np.asarray(scores, dtype=float)
    tie_break = np.random.default_rng(seed).random(len(scores))
    best_first = np.lexsort((tie_break, -scores))  # last key is the main one
    admitted = np.zeros(len(scores), dtype=bool)
    admitted[best_first[: round(fraction * len(scores))]] = True
    return admitted


def admit_above_threshold(scores: Sequence[float], threshold: float = 0.5) -> np.ndarray:
    """Accepts the students whose score is at least `threshold`."""
    return np.asarray(scores, dtype=float) >= threshold


def decision_table(is_excellent: Sequence[bool], admitted: Sequence[bool]) -> pd.DataFrame:
    """Counts students by real outcome and by decision.

    Args:
        is_excellent: Real outcome of each student.
        admitted: Decision for each student.

    Returns:
        A 2x2 table of counts: rows are Excellent / Not excellent, columns are
        Accepted / Rejected.
    """
    real = np.asarray(is_excellent, dtype=bool)
    accepted = np.asarray(admitted, dtype=bool)
    counts = [
        [(real & accepted).sum(), (real & ~accepted).sum()],
        [(~real & accepted).sum(), (~real & ~accepted).sum()],
    ]
    return pd.DataFrame(counts, index=OUTCOMES, columns=DECISIONS)


def count_mistakes(is_excellent: Sequence[bool], admitted: Sequence[bool]) -> dict[str, float]:
    """Counts the two kinds of mistake.

    Args:
        is_excellent: Real outcome of each student.
        admitted: Decision for each student.

    Returns:
        good_rejected: excellent students that were rejected.
        bad_accepted: non-excellent students that were accepted.
        n_excellent, n_not_excellent: size of each class.
        good_rejected_rate, bad_accepted_rate: the counts divided by the class size (NaN for
            an empty class).
    """
    table = decision_table(is_excellent, admitted)
    n_excellent = int(table.loc["Excellent"].sum())
    n_other = int(table.loc["Not excellent"].sum())
    good_rejected = int(table.loc["Excellent", "Rejected"])
    bad_accepted = int(table.loc["Not excellent", "Accepted"])
    return {
        "good_rejected": good_rejected,
        "bad_accepted": bad_accepted,
        "n_excellent": n_excellent,
        "n_not_excellent": n_other,
        "good_rejected_rate": good_rejected / n_excellent if n_excellent else float("nan"),
        "bad_accepted_rate": bad_accepted / n_other if n_other else float("nan"),
    }


def random_selection_table(is_excellent: Sequence[bool], n_admitted: int) -> pd.DataFrame:
    """Expected decision table when `n_admitted` students are picked at random.

    A model is only useful to the extent that it beats this lottery.

    Args:
        is_excellent: Real outcome of each student.
        n_admitted: Number of students accepted.

    Returns:
        Same layout as `decision_table`, with expected (non-integer) counts.
    """
    real = np.asarray(is_excellent, dtype=bool)
    n_excellent = int(real.sum())
    n_other = len(real) - n_excellent
    good_accepted = n_admitted * n_excellent / len(real)
    bad_accepted = n_admitted - good_accepted
    counts = [
        [good_accepted, n_excellent - good_accepted],
        [bad_accepted, n_other - bad_accepted],
    ]
    return pd.DataFrame(counts, index=OUTCOMES, columns=DECISIONS)
