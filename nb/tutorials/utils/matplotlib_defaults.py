"""Matplotlib defaults for TensOps tutorials."""

from cycler import cycler
from matplotlib import pyplot as plt

LINEWIDTH = 1.5
TITLE_FONTSIZE = 13
LABEL_FONTSIZE = 11
LEGEND_FONTSIZE = 12
TICK_FONTSIZE = 11
DEFAULT_COLOR = (0.1, 0.0, 0.8)
FIG_WIDTH = 4
FIG_HEIGHT = 3


def apply():
    """Apply the default Matplotlib style for tutorials."""
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif")
    plt.rcParams.update(
        {
            "lines.linewidth": LINEWIDTH,
            "axes.titlesize": TITLE_FONTSIZE,
            "axes.labelsize": LABEL_FONTSIZE,
            "legend.fontsize": LEGEND_FONTSIZE,
            "xtick.labelsize": TICK_FONTSIZE,
            "ytick.labelsize": TICK_FONTSIZE,
            "figure.figsize": (FIG_WIDTH, FIG_HEIGHT),
            "axes.prop_cycle": cycler("color", [DEFAULT_COLOR]),
        }
    )


__all__ = ["apply"]
