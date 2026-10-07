from src.kinematics import ScatteringKinematics, ElasticKinematics, InelasticKinematics

from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from numpy.typing import ArrayLike
import numpy as np

def plot_vmin(
    kin: ScatteringKinematics,
    ER: ArrayLike,
    ax=None) -> tuple[Figure, Axes]:
    """
    Plot the minimum WIMP speed required
    to produce a recoil of energy ER.

    Parameters
    ----------
    kin : ScatteringKinematics
        Kinematics model.
    ER_grid : ArrayLike
        Recoil-energy grid in eV.
    ax : Axes, optional
        Existing axes. If None, a new
        figure is created.
    label : str, optional
        Curve label.

    Returns
    -------
    tuple[Figure, Axes]
        Matplotlib figure and axes objects.
    """
    if ax is None:
        fig, ax = plt.subplots()
        fig.suptitle(_format_title(kin))
    else:
        fig = ax.figure

    ER = np.asarray(ER)

    ax.plot(ER, kin.vmin_from_ER(ER))

    ax.set_xlabel("Recoil Energy (eV)")
    ax.set_ylabel(r"$v_{\min}$ ($c=1$)")

    ax.ticklabel_format(
        style="sci", axis="x",
        scilimits=(0, 0))

    ax.grid(True, alpha=0.3)

    return fig, ax

def plot_ER_bounds(kin: ScatteringKinematics,
                   vgrid: ArrayLike,
                   ax=None) -> tuple[Figure, Axes]:
    """
    Plot the minimum and maximum recoil
    energies allowed as functions of the
    incoming WIMP speed.

    Parameters
    ----------
    kin : ScatteringKinematics
        Kinematics model.
    v_grid : ArrayLike
        Speed grid (c=1).
    ax : Axes, optional
        Existing axes.

    Returns
    -------
    tuple[Figure, Axes]
        Matplotlib figure and axes objects.
    """
    if ax is None:
        fig, ax = plt.subplots()
        fig.suptitle(_format_title(kin))
    else:
        fig = ax.figure

    vgrid = np.asarray(vgrid)
    ER_bounds = kin.ER_bounds(vgrid)

    ax.ticklabel_format(style="sci", axis="both", scilimits=(0, 0))
    
    ax.plot(vgrid, ER_bounds[0], label=r"$E_R^{\rm min}$")
    ax.plot(vgrid, ER_bounds[1], label=r"$E_R^{\rm max}$")
    ax.set_xlabel("Speed (c=1)")
    ax.set_ylabel(r"$E_R$ bounds (eV)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    return fig, ax

def plot_q_vs_ER(kin: ScatteringKinematics,
                 ER: ArrayLike,
                 ax=None) -> tuple[Figure, Axes]:
    """
    Plot momentum transfer as a function
    of recoil energy.

    Parameters
    ----------
    kin : ScatteringKinematics
        Kinematics model.
    ER_grid : ArrayLike
        Recoil-energy grid in eV.
    ax : Axes, optional
        Existing axes.

    Returns
    -------
    tuple[Figure, Axes]
        Matplotlib figure and axes objects.

    Notes
    -----
    The relation
    q = sqrt(2 mT ER)
    is independent of the scattering model.
    """
    if ax is None:
        fig, ax = plt.subplots()
        fig.suptitle(_format_title(kin))
    else:
        fig = ax.figure

    ER = np.asarray(ER)

    ax.plot(ER, kin.q_from_ER(ER))
    ax.set_xlabel("Recoil Energy (eV)")
    ax.set_ylabel(r"Momentum Transfer $q$ (eV)")

    ax.ticklabel_format(style="sci", axis="both", scilimits=(0, 0))

    ax.grid(True, alpha=0.3)

    return fig, ax

def plot_elastic_vs_inelastic_vmin(elastic: ElasticKinematics,
        inelastic: InelasticKinematics, 
        ER: ArrayLike,
        ax=None) -> tuple[Figure, Axes]:
    
    if ax is None:
        fig, ax = plt.subplots()
        fig.suptitle(_format_title(inelastic))
    else:
        fig = ax.figure
    
    ER = np.asarray(ER)
    vmin_el = elastic.vmin_from_ER(ER)
    vmin_in = inelastic.vmin_from_ER(ER)

    ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))

    ax.plot(ER, vmin_el, label="elastic")
    ax.plot(ER, vmin_in, label="inelastic")
    ax.set_xlabel("Recoil Energy (eV)")
    ax.set_ylabel(r"$v_{min}$ (c=1)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    return fig, ax

def plot_q_vs_theta(kin: ScatteringKinematics,
                    v: float,
                    theta_grid: ArrayLike,
                    ax=None) -> tuple[Figure, Axes]:
    """
    Plot momentum-transfer solutions as
    functions of scattering angle.

    Parameters
    ----------
    kin : ScatteringKinematics
        Kinematics model.
    v : float
        Incoming WIMP speed.
    theta_grid : ArrayLike
        Scattering-angle grid in radians.
    ax : Axes, optional
        Existing axes.

    Returns
    -------
    tuple[Figure, Axes]
        Matplotlib figure and axes objects.

    Notes
    -----
    The two curves correspond to the
    kinematically allowed solutions
    q_minus and q_plus.
    """
    if ax is None:
        fig, ax = plt.subplots()
        fig.suptitle(_format_title(kin))
    else:
        fig = ax.figure

    theta_grid = np.asarray(theta_grid)
    q_min, q_max = kin.q_from_v_theta(v=v, theta=theta_grid)

    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    ax.plot(theta_grid, q_min, label=r"$q_{\rm min}$")
    ax.plot(theta_grid, q_max, label=r"$q_{\rm max}$")
    ax.set_xlabel(r"$\theta$ (radians)")
    ax.set_ylabel("Transferred momentum (eV)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    return fig, ax

def _format_title(kin: ScatteringKinematics) -> str:
    """
    Generate a descriptive title for a
    kinematics model.

    Parameters
    ----------
    kin : ScatteringKinematics
        Kinematics model.

    Returns
    -------
    str
        Formatted plot title including
        WIMP and target masses.
    """
    title = (
        rf"$m_X$={kin.mass_wimp:.3g} eV, "
        rf"$m_T$={kin.mass_target:.3g} eV"
    )

    if isinstance(kin, InelasticKinematics):
        title += rf", $\delta$={kin.delta:.3g} eV"

    return title