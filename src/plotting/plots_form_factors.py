from src.nuclei.form_factors import FormFactor
from src.kinematics import ScatteringKinematics

from numpy.typing import NDArray
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

def plot_form_factor(form_factor: FormFactor,
                     q: NDArray,
                     ax=None,
                     label=None) -> tuple[Figure, Axes]:
    """
    Plot a form factor's F(q).

    Parameters
    ----------
    form_factor : FormFactor
        FormFactor object.
    q : NDArray
        Transferred momentum grid (eV).
    ax : Axes, optional
        Existing matplotlib axis.
        If None, a new figure is created.
    label: str, optional
        Curve label.

    Returns
    -------
    tuple[Figure, Axes]
        Matplotlib figure and axes objects.
    """

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    
    ax.plot(q, form_factor.F(q=q), label=label)

    ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel(r"$q$ (eV)")
    ax.set_ylabel(r"$F(q)$")
    ax.legend()

    return fig, ax

def plot_response(form_factor: FormFactor,
                  q: NDArray,
                  ax=None,
                  label=None) -> tuple[Figure, Axes]:

    """
    Plot a form factor's response |F(q)|^2.

    Parameters
    ----------
    form_factor : FormFactor
        FormFactor object.
    q : NDArray
        Transferred momentum grid (eV).
    ax : Axes, optional
        Existing matplotlib axis.
        If None, a new figure is created.
    label: str, optional
        Curve label.

    Returns
    -------
    tuple[Figure, Axes]
        Matplotlib figure and axes objects.
    """
    
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    
    ax.plot(q, form_factor.response(q=q), label=label)
    ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel(r"$q$ (eV)")
    ax.set_ylabel(r"$|F(q)|^2$")
    ax.legend()

    return fig, ax

def plot_response_vs_ER(form_factor: FormFactor,
                        kinematics: ScatteringKinematics,
                        ER: NDArray,
                        ax=None,
                        label=None) -> tuple[Figure, Axes]:
    """
    Plot a form factor's response |F(ER)|^2
    as function of recoil energy.

    Parameters
    ----------
    form_factor : FormFactor
        FormFactor object.
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
    """

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    q = kinematics.q_from_ER(ER=ER)

    ax.plot(ER, form_factor.response(q=q), label=label)
    ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel(r"$E_R$ (eV)")
    ax.set_ylabel(r"$|F(q)|^2$")
    ax.legend()

    return fig, ax
    