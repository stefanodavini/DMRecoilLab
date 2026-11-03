"""
Plotting utilities for sensitivity and exclusion
curves.

These functions provide visualization helpers for
high-level sensitivity calculations performed by
the analysis package.
"""

from src.halo.base_halo_distribution import IsotropicSpeedDistribution
from src.detectors.detectors import IdealDetector
from src.analysis.si_limits import compute_mass_sigma_limit_si

from numpy.typing import NDArray
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


def plot_mass_sigma_limit_si(
    masses: NDArray,
    detector: IdealDetector,
    delta: float = 0.0,
    halo: IsotropicSpeedDistribution | None = None,
    cl: float = 0.9,
    sigma_reference: float = 1e-45,
    ax: Axes | None = None,
    **plot_kwargs,
    ) -> tuple[Figure, Axes]:
    """
    Plot a spin-independent WIMP sensitivity curve.

    Parameters
    ----------
    masses : NDArray
        WIMP-mass grid in eV.
    detector : IdealDetector
        Detector configuration.
    delta : float, optional
        Inelastic mass splitting in eV.
    halo : IsotropicSpeedDistribution, optional
        Halo speed distribution.
        If None, uses the Standard Halo Model in the
        Earth's frame.
    cl : float, optional
        Confidence level.
    sigma_reference : float, optional
        Reference WIMP-proton cross section used for
        normalization.
    ax : matplotlib.axes.Axes, optional
        Existing axes. If None, a new figure and axes
        are created.
    **plot_kwargs
        Additional keyword arguments passed directly
        to matplotlib.axes.Axes.loglog.
    Returns
    -------
    tuple[Figure, Axes]
        Matplotlib figure and axes objects.
    """

    if ax is None:
        fig, ax = plt.subplots()
        fig.suptitle(r"Zero counts zero backgrounds, SI upper-limit sensitivity")
    else:
        fig = ax.figure

    # mass limit computation here
    sigma = compute_mass_sigma_limit_si(masses=masses,
                                        detector=detector,
                                        delta=delta,
                                        halo=halo,
                                        cl=cl,
                                        sigma_reference=sigma_reference)

    ax.loglog(masses / 1e9, sigma,
              **plot_kwargs)

    ax.set_xlabel(r"$m_\chi$ [GeV]")
    ax.set_ylabel(r"${\sigma}_p$ [cm$^2$]")
    ax.grid(which="both", alpha=0.3)

    if "label" in plot_kwargs:
        ax.legend()

    fig.tight_layout()

    return fig, ax