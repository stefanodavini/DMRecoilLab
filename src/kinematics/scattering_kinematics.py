"""
Scattering kinematics for dark-matter direct detection.

Units
-----
All quantities are expressed in natural units:

    c = 1

with:

    mass      : eV
    energy    : eV
    momentum  : eV
    velocity  : dimensionless
    angles    : radians

ScatteringKinematics extends NuclearRecoilKinematics
with quantities specific to two-body scattering
processes, such as minimum incoming speed and
angular kinematics.

Concrete implementations currently include
elastic and inelastic non-relativistic.
"""

from abc import abstractmethod
from numpy.typing import ArrayLike, NDArray
import numpy as np

from src.kinematics.base_kinematics import NuclearRecoilKinematics


class ScatteringKinematics(NuclearRecoilKinematics):
    """
    Abstract class for dark-matter scattering
    processes producing nuclear recoils.

    Extends NuclearRecoilKinematics with scattering-
    specific quantities such as:
    - minimum speed vmin(ER)
    - angular kinematics
    - kinematic speed masks
    - reduced mass

    Concrete implementations include elastic and
    inelastic scattering models.
    """

    @property
    def mu(self) -> float:
        """
        Reduced mass of the WIMP-target system.

        Returns
        -------
        float
        """
        return self.mX * self.mT / (self.mX + self.mT)

    @abstractmethod
    def vmin_from_ER(self, ER: ArrayLike) -> NDArray:
        """
        Minimum incoming WIMP speed required
        to produce a recoil energy ER.

        Parameters
        ----------
        ER : ArrayLike
            Recoil energy in eV.

        Returns
        -------
        ndarray
            Minimum WIMP speed (c=1).
        """
        pass

    @abstractmethod
    def q_from_v_theta(self,
                       v: ArrayLike,
                       theta: ArrayLike
                       ) -> tuple[NDArray, NDArray]:
        """
        Momentum-transfer solutions for a
        WIMP with incident speed v and
        scattering angle theta.

        Parameters
        ----------
        v : ArrayLike
            Incoming WIMP speed.
        theta : ArrayLike
            Scattering angle in radians.

        Returns
        -------
        tuple of ndarray
            (q_minus, q_plus), corresponding
            to the two solutions of the
            energy-momentum conservation equation.
        """
        pass

    @abstractmethod
    def speed_mask(self, v: ArrayLike) -> NDArray:
        """
        Boolean mask identifying
        kinematically allowed speeds.
        """
        pass


class NonRelativisticKinematics(ScatteringKinematics):
    """
    Abstract class for non-relativistic
    dark-matter scattering kinematics.

    Defines the kinematic relations connecting
    recoil energy ER and momentum transfer q,
    for two-body non-relativistic scattering.

    Subclasses implement specific kinematic
    models (elastic or inelastic).

    Parameters
    ----------
    mX
        WIMP mass in eV.

    mT
        Target mass in eV.

    Notes
    -----
    Derived classes must implement:

    - vmin_from_ER
    - ER_min
    - ER_max
    - q_from_v_theta
    """
    def __init__(self,
                 mX: float,
                 mT: float):
        super().__init__(mX=mX, mT=mT)

    def q_from_ER(self, ER: ArrayLike) -> NDArray:
        """
        Momentum transfer corresponding to a
        recoil energy ER.

        Parameters
        ----------
        ER : ArrayLike
            Recoil energy in eV.

        Returns
        -------
        ndarray

        Notes
        -----
        The relation q = sqrt(2 mT ER)
        follows from non-relativistic nuclear
        recoil kinematics.
        """
        ER = np.asarray(ER)
        return np.sqrt(2 * self.mT * ER)

    def speed_mask(self, v: ArrayLike) -> NDArray:
        """
        Boolean mask identifying
        kinematically allowed speeds.

        Notes
        -----
        For elastic scatterings,
        all speeds are kinematically allowed.
        Endothermic kinematics implementations 
        must override the method.
        """
        v = np.asarray(v)
        return np.ones_like(v, dtype=bool)

    @property
    def threshold_speed(self) -> float:
        """
        Minimum physically allowed incoming WIMP speed.

        Returns
        -------
        float

        Notes
        -----
        Elastic scattering has no threshold and
        returns zero.

        Inelastic scattering models must override
        this method.
        """
        return 0.0



class ElasticKinematics(NonRelativisticKinematics):
    """
    Elastic WIMP-target scattering.

    The initial and final WIMP masses
    are identical.

    Parameters
    ----------
    mX : float
        WIMP mass in eV.

    mT : float
        Target mass in eV.
    """
    def __init__(self, mX: float, mT: float):
        super().__init__(mX=mX, mT=mT)

    def vmin_from_ER(self, ER: ArrayLike) -> NDArray:
        """
        Minimum incoming WIMP speed required
        to produce an elastic recoil of energy ER.

        Notes
        -----
        vmin =  sqrt(mT ER / 2) / mu
        """
        ER = np.asarray(ER)
        return np.sqrt(self.mT * ER / 2) / self.mu 

    def ER_min(self, v: ArrayLike) -> NDArray:
        v = np.asarray(v)
        if v.ndim == 0:
            return 0.
        return np.zeros_like(v)

    def ER_max(self, v: ArrayLike) -> NDArray:
        v = np.asarray(v)
        return 2 * self.mu**2 * v**2 / self.mT

    def q_from_v_theta(self, v: ArrayLike, theta: ArrayLike):
        """
        Momentum transfer for elastic scattering.

        Parameters
        ----------
        v : ArrayLike
            Incoming WIMP speed (c=1).
        theta : ArrayLike
            Target-recoil scattering angle in radians,
            in the laboratory frame,
            measured with respect to the incoming WIMP direction.

        Returns
        -------
        tuple
            (0, q_max)

        Notes
        -----
        For elastic scattering the non trivial solution is:
        q = 2 mu v cos(theta)
        Here theta is the recoil angle in the laboratory frame.
        """
        v = np.asarray(v)
        theta = np.asarray(theta)
        return 0., 2 * self.mu * v * np.cos(theta)
    

class InelasticKinematics(NonRelativisticKinematics):
    """
    Inelastic WIMP-target scattering.

    Parameters
    ----------
    mX : float
        WIMP mass.

    mT : float
        Target mass.

    delta : float
        Mass splitting between initial and
        final WIMP states.

        delta > 0
            Endothermic scattering.

        delta < 0
            Exothermic scattering.

    Notes
    -----
    Kinematically forbidden configurations
    return NaN through the underlying
    square-root operations.
    """
    def __init__(self, mX: float, mT: float, delta: float):
        super().__init__(mX=mX, mT=mT)
        self.delta = delta

    @property
    def is_endothermic(self):
        """
        Whether the scattering process is endothermic.

        Returns
        -------
        bool
            True when delta > 0.
        """
        return self.delta > 0

    @property
    def threshold_speed(self) -> float:
        """
        Minimum speed required for
        endothermic scattering.

        Returns
        -------
        float

        Notes
        -----
        For endothermic (delta > 0):
        v_thr = sqrt(2 delta / mu)
        Otherwise
        v_thr = 0
        """
        if self.is_endothermic:
            return np.sqrt(2 * self.delta / self.mu)
        return 0.

    def speed_mask(self, v: ArrayLike) -> NDArray:
        """
        Boolean mask identifying
        kinematically allowed speeds.

        Parameters
        ----------
        v : ArrayLike

        Returns
        -------
        ndarray of bool
        """
        v = np.asarray(v)
        return v >= self.threshold_speed

    def speed_theta_mask(self, 
                         v: ArrayLike,
                         theta: ArrayLike
                        ) -> NDArray:
        """
        Boolean mask identifying kinematically
        allowed (v, theta) configurations.

        Parameters
        ----------
        v : ArrayLike
            Incoming WIMP speed (c=1).
        theta : ArrayLike
            Target-recoil scattering angle in radians,
            in the laboratory frame,
            measured with respect to the incoming WIMP direction.

        Returns
        -------
        ndarray of bool

        Notes
        -----
        For endothermic scattering, only
        configurations satisfying
        v cos(theta) >= v_threshold
        admit real momentum-transfer
        solutions.
        """
        v = np.asarray(v)
        theta = np.asarray(theta)
        return v * np.cos(theta) >= self.threshold_speed

    def vmin_from_ER(self, ER: ArrayLike) -> NDArray:
        """
        Minimum incoming WIMP speed required to produce
        a recoil of energy ER.

        Parameters
        ----------
        ER : ArrayLike
            Recoil energy in eV.

        Returns
        -------
        ndarray
            Minimum physically allowed WIMP speed (c=1).

        Notes
        -----
        For inelastic scattering,
        vmin = | q/(2μ) + δ/q |
        where
        q = sqrt(2 mT ER).
        """
        ER = np.asarray(ER)
        qmax = self.q_from_ER(ER)
        return np.abs(qmax / (2 * self.mu) + self.delta / qmax)


    def ER_min(self, v: ArrayLike) -> NDArray:
        """
        Minimum recoil energy for inelastic scattering.

        Derived from the analytic roots of
        q²/(2μ) - qv + δ = 0.

        Notes
        _____
        For endothermic scattering,
        speed below the theshold speed are
        kinematically forbidden.

        In this region, the square-root argument
        becomes negative and the method returns NaN.

        Runtime warnings associated with these
        forbidden configuration are suppressed,
        since NaN is the intended representation
        as an unphysical kinematic region.
        """
        v = np.asarray(v)
        muv = self.mu * v
        with np.errstate(invalid="ignore"):
            q = muv - np.sqrt(muv**2 - 2 * self.mu * self.delta)
        return q**2 / (2 * self.mT)

    def ER_max(self, v: ArrayLike) -> NDArray:
        """
        Maximum recoil energy for inelastic scattering.

        Derived from the analytic roots of
        q²/(2μ) - qv + δ = 0.

        Notes
        _____
        For endothermic scattering,
        speed below the theshold speed are
        kinematically forbidden.

        In this region, the square-root argument
        becomes negative and the method returns NaN.

        Runtime warnings associated with these
        forbidden configuration are suppressed,
        since NaN is the intended representation
        as an unphysical kinematic region.
        """
        v = np.asarray(v)
        muv = self.mu * v
        with np.errstate(invalid="ignore"):
            q = muv + np.sqrt(muv**2 - 2 * self.mu * self.delta)
        return q**2 / (2 * self.mT)

    def q_from_v_theta(self, v: ArrayLike, theta: ArrayLike):
        """
        Momentum-transfer solutions for inelastic scattering.

        Parameters
        ----------
        v : ArrayLike
            Incoming WIMP speed (c=1).
        theta : ArrayLike
            Target-recoil scattering angle in radians,
            in the laboratory frame,
            measured with respect to the incoming WIMP direction.

        Returns
        -------
        tuple[ndarray, ndarray]
            Lower and upper momentum-transfer solutions
            obtained from energy-momentum conservation.

        Notes
        -----
        Unphysical configurations return NaN through the
        underlying square-root operation.
        """
        v = np.asarray(v)
        theta = np.asarray(theta)

        muvcosth = self.mu * v * np.cos(theta)
        muvcosth2 = muvcosth**2

        sol_pos = muvcosth + np.sqrt(muvcosth2 - 2 *self.mu * self.delta)
        sol_neg = muvcosth - np.sqrt(muvcosth2 - 2 *self.mu * self.delta)

        return sol_neg, sol_pos
