"""Plots comparing excellent and non-excellent students, feature by feature."""

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from data_use_ce.data import TARGET

EXCELLENT_COLOR = "orange"
OTHER_COLOR = "steelblue"
# Numeric columns with more distinct values than this are drawn as histograms.
MAX_CATEGORIES = 10


def _legend_handles() -> list[Patch]:
    return [
        Patch(color=EXCELLENT_COLOR, label="Excellent"),
        Patch(color=OTHER_COLOR, label="Not excellent"),
    ]


def _new_axes() -> Axes:
    return plt.subplots(figsize=(5, 3.5))[1]


def plot_excellence_rate(
    df: pd.DataFrame, column: str, ax: Axes | None = None, legend: bool = True
) -> Axes:
    """Draws the share of excellent students for each value of `column`.

    Each bar is split into the excellent share (orange) and the rest (blue), and is labelled
    with that share and with n, the number of students in the category. The dashed line is
    the excellent share over all students: a category with more orange than the line does
    better than average.

    Args:
        df: Students, with the "Excellent" column.
        column: Binary, categorical or ordinal column.
        ax: Axes to draw on. A new figure is created if omitted.
        legend: Whether to add the legend.

    Returns:
        The axes used.
    """
    ax = ax or _new_axes()
    table = df.groupby(column)[TARGET].agg(n="size", rate="mean")
    x = np.arange(len(table))
    ax.bar(x, table["rate"], color=EXCELLENT_COLOR)
    ax.bar(x, 1 - table["rate"], bottom=table["rate"], color=OTHER_COLOR)
    ax.axhline(df[TARGET].mean(), color="black", linestyle="--", linewidth=1)
    for i, (rate, n) in enumerate(zip(table["rate"], table["n"], strict=True)):
        ax.text(i, 1.02, f"{rate:.0%}\n(n={n})", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, table.index.astype(str))
    ax.set_ylim(0, 1.25)
    ax.set_title(column)
    ax.set_ylabel("share of students")
    if legend:
        ax.legend(handles=_legend_handles(), loc="upper right", fontsize=8)
    return ax


def plot_feature_histogram(
    df: pd.DataFrame, column: str, ax: Axes | None = None, bins: int = 10, legend: bool = True
) -> Axes:
    """Draws the distribution of a numeric column for each class, side by side.

    A bar is the share of its own class that falls in the bin, not a count, so the small
    excellent class and the large other class can be compared.

    Args:
        df: Students, with the "Excellent" column.
        column: Numeric column.
        ax: Axes to draw on. A new figure is created if omitted.
        bins: Number of bins, shared by the two classes.
        legend: Whether to add the legend.

    Returns:
        The axes used.
    """
    ax = ax or _new_axes()
    values = df[column].to_numpy()
    is_excellent = df[TARGET].to_numpy()
    edges = np.histogram_bin_edges(values, bins=bins)
    width = np.diff(edges) / 2
    for mask, offset, color in (
        (is_excellent, 0, EXCELLENT_COLOR),
        (~is_excellent, width, OTHER_COLOR),
    ):
        counts, _ = np.histogram(values[mask], bins=edges)
        ax.bar(edges[:-1] + offset, counts / max(mask.sum(), 1), width, align="edge", color=color)
    ax.set_title(column)
    ax.set_ylabel("share of its class")
    if legend:
        ax.legend(handles=_legend_handles(), loc="upper right", fontsize=8)
    return ax


def plot_scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    ax: Axes | None = None,
    jitter: float = 0.0,
    seed: int = 0,
    legend: bool = True,
) -> Axes:
    """Scatter plot of two numeric columns, coloured by class.

    Args:
        df: Students, with the "Excellent" column.
        x: Column on the horizontal axis.
        y: Column on the vertical axis.
        ax: Axes to draw on. A new figure is created if omitted.
        jitter: Half-width of the random shift added to both coordinates, so that students
            with the same whole-number values do not hide each other.
        seed: Seed of the random shift.
        legend: Whether to add the legend.

    Returns:
        The axes used.
    """
    ax = ax or _new_axes()
    rng = np.random.default_rng(seed)
    # The small excellent class is drawn last so it stays visible.
    for is_excellent, color in ((False, OTHER_COLOR), (True, EXCELLENT_COLOR)):
        group = df[df[TARGET] == is_excellent]
        shift_x = rng.uniform(-jitter, jitter, len(group))
        shift_y = rng.uniform(-jitter, jitter, len(group))
        ax.scatter(group[x] + shift_x, group[y] + shift_y, color=color, alpha=0.6, s=18)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    if legend:
        ax.legend(handles=_legend_handles(), loc="upper right", fontsize=8)
    return ax


def plot_feature_grid(df: pd.DataFrame, features: Sequence[str], ncols: int = 3) -> Figure:
    """Plots every feature of a group against excellence, one panel per feature.

    Numeric features with more than MAX_CATEGORIES distinct values get a histogram; all the
    others get the excellent-share bars.

    Args:
        df: Students, with the "Excellent" column.
        features: Columns to plot.
        ncols: Number of panels per row.

    Returns:
        The figure.
    """
    nrows = -(-len(features) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.5 * nrows), squeeze=False)
    for ax, column in zip(axes.flat, features, strict=False):
        is_continuous = (
            pd.api.types.is_numeric_dtype(df[column]) and df[column].nunique() > MAX_CATEGORIES
        )
        plot = plot_feature_histogram if is_continuous else plot_excellence_rate
        plot(df, column, ax=ax, legend=False)
    for ax in axes.flat[len(features) :]:
        ax.set_visible(False)
    fig.legend(handles=_legend_handles(), loc="upper center", ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig
