from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from src.analysis.experiment_matching import MatchedCountGrid

def plot_matched_counts(
    grid: MatchedCountGrid,
    ax: Axes | None = None,
    logx: bool = True,
    cmap: str = "viridis",
) -> tuple[Figure, Axes]:
    """
    Plot matched signal counts in the WIMP-mass and
    mass-splitting parameter plane.

    Parameters
    ----------
    grid : MatchedCountGrid
        Matched-count results returned by
        ``matched_counts_grid_si``.

    ax : Axes, optional
        Existing matplotlib axis.
        If None, a new figure is created.

    logx : bool, optional
        Whether to display the WIMP-mass axis on a
        logarithmic scale.

    cmap : str, optional
        Matplotlib colormap used to display the
        matched-count values.

    Returns
    -------
    tuple[Figure, Axes]
        Matplotlib figure and axes objects.

    Notes
    -----
    The color scale represents the number of signal
    counts predicted in detector B after matching the
    interaction-model reference cross section to
    reproduce a specified number of counts in
    detector A.

    The horizontal axis corresponds to the WIMP mass
    and the vertical axis corresponds to the mass
    splitting. The plotted values are stored in
    ``grid.counts_b``.
    """
    
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    M, D = grid.meshgrid

    pcm = ax.pcolormesh(
        M,
        D,
        grid.counts_b,
        shading="auto",
        cmap=cmap,
    )

    if logx:
        ax.set_xscale("log")
    else:
        ax.ticklabel_format(style="sci", axis="x",
                            scilimits=(0, 0))

    ax.set_xlabel(r"$m_\chi$ [eV]")
    ax.set_ylabel(r"$\delta$ [eV]")

    ax.ticklabel_format(style="sci", axis="y",
                        scilimits=(0, 0))

    plt.colorbar(
        pcm,
        ax=ax,
        label="Matched counts"
    )

    return fig, ax
