"""Consistent visual style for the portfolio notebooks.

A single import (`from src.visualization import set_style`) gives every notebook
the same look-and-feel: legible fonts, soft grid, modern palette.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# Color palette — tuned for accessibility (avoids red/green ambiguity).
PALETTE = {
    "primary": "#4C6EF5",
    "secondary": "#15AABF",
    "accent": "#F76707",
    "neutral": "#495057",
    "muted": "#ADB5BD",
    "success": "#37B24D",
    "danger": "#E03131",
}

PALETTE_LIST = [
    "#4C6EF5",
    "#F76707",
    "#15AABF",
    "#37B24D",
    "#E03131",
    "#7048E8",
    "#F59F00",
]


def set_style() -> None:
    """Apply a clean, paper-ready matplotlib/seaborn style."""
    sns.set_theme(
        style="whitegrid",
        context="notebook",
        font_scale=1.0,
        rc={
            "figure.figsize": (8, 5),
            "figure.dpi": 110,
            "savefig.dpi": 150,
            "axes.titleweight": "semibold",
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "axes.edgecolor": "#343A40",
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": "#DEE2E6",
            "grid.linewidth": 0.6,
            "legend.frameon": False,
        },
    )
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=PALETTE_LIST)


def annotate_bars(ax: plt.Axes, fmt: str = "{:.3f}", offset: float = 0.01) -> None:
    """Write value labels on top of every bar in an Axes."""
    ymax = ax.get_ylim()[1]
    for bar in ax.patches:
        height = bar.get_height()
        if height == 0:
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + offset * ymax,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=9,
            color=PALETTE["neutral"],
        )
