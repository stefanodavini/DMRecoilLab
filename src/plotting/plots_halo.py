from src.halo.base_halo_distribution import IsotropicSpeedDistribution

from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

import numpy as np

def plot_speed_distribution(
        distribution: IsotropicSpeedDistribution,
        vmax=None,
        num=1000,
        ax=None,
        **plot_kwargs
    ) -> tuple[Figure, Axes]:
    """
    Plot a speed distribution.

    Parameters
    ----------
    distribution
        Isotropic speed distribution object.
    vmax : float
        Maximum speed shown on the x axis.
    num : integer
        Number of sampling points.
    ax : Axes, optional
        Existing matplotlib axis.
        If None, a new figure is created.
    **plot_kwargs
        Additional keyword arguments passed directly
        to matplotlib.axes.Axes.plot.

    Returns
    -------
    tuple[Figure, Axes]
        Matplotlib figure and axes objects.
    """

    if vmax is None:
        vmax = distribution.maximum_speed

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    v = np.linspace(0, vmax, num)
    g = distribution.speed_pdf(v=v)

    ax.plot(v, g, **plot_kwargs)
    ax.set_xlabel("Speed (km/s)")
    ax.set_ylabel(r"$g(v) = 4 \pi v^2  \,f(v)$")

    if "label" in plot_kwargs:
        ax.legend()

    return fig, ax


def plot_boosted_speed_distribution(
        distribution: IsotropicSpeedDistribution,
        vboost: float,
        vmax=None,
        num=1000,
        ax=None
    ) -> tuple[Figure, Axes]:
    """
    Compare a galactic-frame speed distribution
    with the corresponding boosted observer-frame
    distribution.

    Parameters
    ----------
    distribution
        Isotropic Speed distribution object.

    vboost : float
        Boost speed
    vmax : float
        Maximum speed shown on the x axis.
    num : int
        Number of sampling points.
    ax : Axes, optional
        Existing matplotlib axis.
        If None, a new figure is created.

    Returns
    -------
    tuple[Figure, Axes]
        Matplotlib figure and axes objects.
    """

    if vmax is None:
        vmax = distribution.maximum_speed + vboost

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    v = np.linspace(0, vmax, num)
    boosted = distribution.boost(vboost=vboost)

    ax.plot(v, distribution.speed_pdf(v=v),
            label="Galactic Frame", lw=2)

    ax.plot(v, boosted.speed_pdf(v=v),
            label=f"Observer frame ({vboost:.0f} km/s)", lw=2)

    ax.set_xlabel("Speed (km/s)")
    ax.set_ylabel(r"$g(v) = 4 \pi v^2 \,f(v)$")
    ax.legend()

    return fig, ax