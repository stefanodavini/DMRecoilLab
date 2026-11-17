"""Plot detector acceptance curves."""

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from src.utils.tabulated_data import TabulatedData


def plot_detector_acceptance(
    acceptance_curve: TabulatedData,
    ax: Axes | None = None,
    **plot_kwargs,
) -> tuple[Figure, Axes]:
    """Plot a detector's tabulated acceptance versus recoil energy.

    Parameters
    ----------
    acceptance_curve : TabulatedData
        Acceptance values with recoil-energy coordinates in eV.
        Obtain a table from a reference acceptance function or a
        ResponseDetector's acceptance_curve attribute.
    ax : Axes, optional
        Existing matplotlib axis. If None, a new figure is created.
    **plot_kwargs
        Additional keyword arguments passed directly to Axes.plot.

    Returns
    -------
    tuple[Figure, Axes]
        Matplotlib figure and axes objects. Energy is displayed in keV.

    Notes
    -----
    This function plots the supplied table without changing its acceptance
    values or applying detector resolution. Energy calibration and any
    benchmark extrapolations are the responsibility of the table provider.
    The figure is not shown automatically.
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    ax.plot(acceptance_curve.x / 1e3, acceptance_curve.y, **plot_kwargs)
    ax.set_xlabel("NR recoil energy [keVnr]")
    ax.set_ylabel("Acceptance / efficiency")
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)
    if "label" in plot_kwargs:
        ax.legend()

    return fig, ax
