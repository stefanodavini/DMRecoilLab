from src.rates import RateCalculator

from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from numpy.typing import NDArray
import numpy as np

def plot_recoil_spectrum(
    rate_calc: RateCalculator,
    ER: NDArray,
    ax=None,
    label=None) -> tuple[Figure, Axes]:
    """
    Plot a recoil energy spectrum dR/dE.

    Parameters
    ----------
    rate_calc : RateCalculator
        RateCalculator  object.
    ER : NDArray
        Recoil energy grid (eV).
    ax : Axes, optional
        Existing matplotlib axis.
        If None, a new figure is created.
    label: str, optional
        Curve label.

    Returns
    -------
    tuple[Figure, Axes]
        Matplotlib figure and axes objects.

    Notes
    -----
    The plotted quantity is dR/dER.
    Refer to the documentation in RateCalculator
    for units and conventions.
    """

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    dRdER = rate_calc.dRdER(ER)
    ER_keV = ER / 1e3

    ax.plot(ER_keV, dRdER, label=label)
    ax.set_xlabel("recoil energy (keV)")
    ax.set_ylabel("differential rate")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    return fig, ax


def plot_normalized_recoil_spectrum(
    rate_calc: RateCalculator,
    ER: NDArray,
    ax=None,
    label=None,
    ) -> tuple[Figure, Axes]:
    """
    Plot a recoil energy spectrum dR/dE,
    normalized to 1.

    Parameters
    ----------
    rate_calc : RateCalculator
        RateCalculator  object.
    ER : NDArray
        Recoil energy grid (eV).
    ax : Axes, optional
        Existing matplotlib axis.
        If None, a new figure is created.
    label: str, optional
        Curve label.

    Returns
    -------
    tuple[Figure, Axes]
        Matplotlib figure and axes objects.

    Notes
    -----
    The normalization is computed numerically
    over the supplied recoil-energy grid.
    """

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    dRdER = rate_calc.dRdER(ER)
    norm = np.trapezoid(dRdER, ER)
    ER_keV = ER / 1e3

    ax.plot(ER_keV, dRdER / norm, label=label)
    ax.set_xlabel("recoil energy (keV)")
    ax.set_ylabel("normalized differential rate")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    return fig, ax

def plot_integral_rate(
    rate_calc: RateCalculator,
    ER: NDArray,
    xlim=None,
    ylim=None,
    ax=None,
    label=None,
    ) -> tuple[Figure, Axes]:
    """
    Plot the recoil rate above threshold.

    Parameters
    ----------
    rate_calc : RateCalculator
        RateCalculator  object.
    ER : NDArray
        Recoil energy grid (eV).
    ax : Axes, optional
        Existing matplotlib axis.
        If None, a new figure is created.
    label: str, optional
        Curve label.

    Returns
    -------
    tuple[Figure, Axes]
        Matplotlib figure and axes objects.

    Notes
    -----
    The plotted quantity is
    R(Ethr) = ∫_{Ethr}∞ dR/dE' dE'

    where each recoil-energy point is treated
    as a detector threshold.
    """
    
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    intrate = rate_calc.integrated_rate_above_threshold(ER)

    ER_keV = ER / 1e3

    ax.plot(ER_keV, intrate, label=label)
    ax.set_xlabel("threshold recoil energy (keV)")
    ax.set_ylabel("integral rate (counts/detector mass/ exposure time)")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    if xlim is not None:
        ax.set_xlim(*xlim)

    if ylim is not None:
        ax.set_ylim(*ylim)

    return fig, ax