from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np

from src.analysis.likelihoods import LikelihoodGridVSMassDelta

def plot_q_grid_mass_delta(
    grid: LikelihoodGridVSMassDelta,
    ax: Axes | None = None,
    logx: bool = True,
    cmap: str = "viridis",
) -> tuple[Figure, Axes]:
    """
    Plot the Likelihood-ratio test statistic in the WIMP-mass and
    mass-splitting parameter plane.

    Parameters
    ----------
    grid : LikelihoodGridVSMassDelta
        LikelihoodGridVSMassDelta results returned by
        ``likelihood_grid_si_mass_delta``.

    ax : Axes, optional
        Existing matplotlib axis.
        If None, a new figure is created.

    logx : bool, optional
        Whether to display the WIMP-mass axis on a
        logarithmic scale.

    cmap : str, optional
        Matplotlib colormap used to display the
        likelihood-ratio test statistic.

    Returns
    -------
    tuple[Figure, Axes]
        Matplotlib figure and axes objects.

    Notes
    -----
    The color scale represents the test statistics q.

    The horizontal axis corresponds to the WIMP mass
    and the vertical axis corresponds to the mass
    splitting. The plotted values are stored in
    ``grid.q_cap_25``.
    """

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    M, D = grid.meshgrid

    pcm = ax.pcolormesh(M, D, grid.q_cap_25,
                        shading="auto",
                        cmap=cmap)

    if logx:
        ax.set_xscale("log")
    else:
        ax.ticklabel_format(style="sci", axis="x",
                            scilimits=(0, 0))

    ax.set_xlabel(r"$m_\chi$ [eV]")
    ax.set_ylabel(r"$\delta$ [eV]")

    ax.ticklabel_format(style="sci", axis="y",
                        scilimits=(0, 0))

    ax.plot(grid.best_fit_mass,
            grid.best_fit_delta,
            marker="*",
            markersize=16,
            color="red",
            markeredgecolor="white",
            )

    plt.colorbar(pcm, ax=ax,
                 label=r"$q = - 2 \Delta \ln L$ (capped at 25)")

    return fig, ax

def plot_q_contour_mass_delta(
    grid: LikelihoodGridVSMassDelta,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """
    Plot likelihood-ratio confidence contours in the
    WIMP-mass and mass-splitting plane.

    Parameters
    ----------
    grid : LikelihoodGridVSMassDelta
        Likelihood scan result.
    ax : Axes, optional
        Existing matplotlib axis.

    Returns
    -------
    tuple[Figure, Axes]
        Matplotlib figure and axes objects.

    Notes
    -----
    Contours are drawn at
    q = -2 ΔlogL
    corresponding to the standard two-parameter
    confidence regions
    68.3% : q = 2.30
    95.4% : q = 6.18
    99.7% : q = 11.83
    when the likelihood-ratio statistic follows
    a chi-square distribution with two degrees
    of freedom.
    """

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    levels = [2.30, 6.18, 11.83]

    valid_levels = [
        lev for lev in levels 
        if lev <= np.nanmax(grid.q)
        ]

    M, D = grid.meshgrid

    cs = ax.contour(M, D, grid.q,
        levels=valid_levels,
        colors=["green","orange","red"],
        linewidths=2)

    fmt = {
        2.30: "68.3%",
        6.18: "95.4%",
        11.83: "99.7%",
        }

    ax.clabel(cs, fmt=fmt)

    ax.plot(
        grid.best_fit_mass,
        grid.best_fit_delta,
        marker="*",
        markersize=16,
        color="red",
        markeredgecolor="white",
        label="Best fit"
    )

    ax.set_xscale("log")
    ax.set_xlabel(r"$m_\chi$ [eV]")
    ax.set_ylabel(r"$\delta$ [eV]")
    ax.legend()

    return fig, ax
