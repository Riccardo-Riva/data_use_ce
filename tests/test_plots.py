"""Tests for the plot helpers, on a toy table (no display needed)."""

import numpy as np
import pandas as pd
import pytest

from data_use_ce.data import TARGET
from data_use_ce.plots import (
    plot_excellence_rate,
    plot_feature_grid,
    plot_feature_histogram,
    plot_scatter,
)


@pytest.fixture
def toy():
    # Category "a": 4 students, 2 excellent. Category "b": 1 student, excellent.
    return pd.DataFrame(
        {
            "kind": ["a", "a", "a", "a", "b"],
            "level": [1, 2, 3, 4, 4],
            "count": [0, 1, 2, 3, 40],
            TARGET: [True, False, True, False, True],
        }
    )


def test_rate_bars_show_the_excellent_share_and_group_size(toy):
    ax = plot_excellence_rate(toy, "kind")
    orange_heights = [p.get_height() for p in ax.patches[:2]]
    assert orange_heights == pytest.approx([0.5, 1.0])
    assert [t.get_text() for t in ax.texts] == ["50%\n(n=4)", "100%\n(n=1)"]
    assert [t.get_text() for t in ax.get_xticklabels()] == ["a", "b"]


def test_rate_bars_draw_the_overall_share_as_a_dashed_line(toy):
    ax = plot_excellence_rate(toy, "kind")
    assert ax.lines[0].get_ydata()[0] == pytest.approx(0.6)


def test_histogram_bars_are_shares_of_each_class(toy):
    ax = plot_feature_histogram(toy, "count", bins=4)
    heights = [p.get_height() for p in ax.patches]
    assert sum(heights[:4]) == pytest.approx(1.0)  # excellent class
    assert sum(heights[4:]) == pytest.approx(1.0)  # other class


def test_scatter_plots_every_student_once(toy):
    ax = plot_scatter(toy, "level", "count")
    assert sum(len(c.get_offsets()) for c in ax.collections) == len(toy)


def test_scatter_jitter_is_bounded_and_reproducible(toy):
    first = plot_scatter(toy, "level", "count", jitter=0.2, seed=1)
    second = plot_scatter(toy, "level", "count", jitter=0.2, seed=1)
    for a, b in zip(first.collections, second.collections, strict=True):
        assert np.array_equal(a.get_offsets(), b.get_offsets())
        x = a.get_offsets()[:, 0]
        assert np.all(np.abs(x - np.round(x)) <= 0.2)


def test_grid_picks_the_plot_type_and_hides_unused_panels():
    df = pd.DataFrame(
        {
            "kind": ["a", "b"] * 6,
            "level": [1, 2, 3] * 4,
            "many": range(12),  # more than 10 distinct values: histogram
            TARGET: [True, False, False] * 4,
        }
    )
    fig = plot_feature_grid(df, ["kind", "level", "many"], ncols=2)
    kind, level, many, unused = fig.axes[:4]
    assert kind.get_ylabel() == "share of students"
    assert level.get_ylabel() == "share of students"
    assert many.get_ylabel() == "share of its class"
    assert not unused.get_visible()
