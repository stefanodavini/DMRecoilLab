"""
Exact chi1 + N -> chi2 + N 
relativistic kinematics for a stationary target.

Units
-----
All quantities are expressed in natural units:

    c = 1

with:

    mass      : eV
    energy    : eV
    momentum  : eV
    velocity  : dimensionless


Symbols
--------
m1 mass of the incoming WIMP chi1
m2 mass of the outgoing WIMP chi2
E1 (also called E internally) is the incoming WIMP *total* energy.
K = E1 - m1 is kinetic energy.
ER is the target recoil *kinetic* energy.
q always denotes spatial momentum magnitude,
Q2 = -t invariant of 4-momentum transfer, not |q|^2.

Physical formulas and their numerically stable implementations
------------------------------------------------------------
With w = sqrt(1-v**2), evaluated as sqrt((1-v)*(1+v)):

Kinetic Energy as function of mass and speed
* K = E1 - m1 = sqrt(m1**2 + p**2) - m1
      = m1*v**2 / (w*(1+w)).

Velocity as function of (kinetic) energy and mass
* v = sqrt(1 - (m1/E)**2)
    = sqrt(K/E)*sqrt((K+2*m1)/E).

Momentum as function of (kinetic) energy and mass
* p = sqrt(E1**2-m1**2) = sqrt(K)*sqrt(E1+m1).

Momentum transfer to the target recoil,
as function of recoil kinetic energy and target mass
* q = sqrt(ER*(ER+2*mT)) = sqrt(ER)*sqrt(ER+2*mT).

Recoil kinetic energy as function of momentum transfer
and target mass
* ER = sqrt(mT**2+q**2)-mT = q*(q/(hypot(mT,q)+mT)).

* p_f_star = sqrt(kallen(s,m2**2,mT**2))/(2*sqrt(s)).
  Near threshold,
  evaluate sqrt(f1)*sqrt(f2)/(2*sqrt(s)), where
  f1 = s-(m2+mT)**2 = 2*mT*(K-K_reaction),
  f2 = s-(m2-mT)**2 = f1+4*m2*mT.
  This avoids subtraction of large
  squared masses before resolving the small available kinetic energy.

* ER_min/max = (2*E1_star*E2_star-m1**2-m2**2
               -/+ 2*p1_star*p2_star)/(2*mT).
  Equivalently, boost the endpoint target momenta to qlo/qhi and use
  the rationalized ER(q) above; this avoids subtracting rest energies.

* For ER > 0, B = mT*ER+(m2**2-m1**2)/2 and q = sqrt(ER*(ER+2*mT)):
  E_min = (ER*B+q*sqrt(B**2+2*mT*ER*m1**2))/(2*mT*ER),
  v_min = sqrt(1-(m1/E_min)**2). Instead evaluate v_min = tanh(abs(y)),
  y = asinh(B/(m1*sqrt(2*mT*ER)))+asinh(sqrt(ER/(2*mT))).

* The angular quadratic has roots (C*b +/- radical)/a. Compute the
  larger-magnitude root without subtraction, then use q1*q2 = c/a.


Validation philosophy
---------------------

Domain violations raise exceptions.

Examples:
* v < 0
* v >= 1
* ER < 0
* E1 < m1
* theta outside [0, pi]

All inputs must be finite.
A violation in any array entry rejects the entire call.

Kinematically forbidden configurations are masked.

Examples:
* v below threshold_speed
* E1 below threshold_energy
* forbidden angular configurations

Masked observables return NaN and
masked predicates return False.

Validity and legacy conventions
-------------------------------

Numerical methods invoke the base _validate_domain_speed and
_validate_domain_ER APIs with require_grid=False,
retaining scalar and broadcast inputs and empty arrays.
The relativistic speed validator extends the base check with v < 1.

Valid-domain configurations may still be forbidden by conservation laws:
closed channels return NaN endpoints and False masks;
angular roots are masked independently.
For ER=0, vmin is 0 in the elastic case and inf for
unequal masses: zero transfer cannot change the WIMP mass.

Threshold comparisons in v and E1 are inclusive and strict:
the immediate representable value below threshold remains forbidden.
_ROUNDING_RTOL applies only to derived kinetic energies,
angular discriminants and outgoing energies;
_RESIDUAL_RTOL applies to conservation residuals.
These are dimensionless relative tolerances, not energy or momentum cutoffs.

The NR classes share the same speed-mask meaning (allowed channel),
but use approximate thresholds without rounding guards.
NR angular roots may be signed; this implementation masks negative roots.
NR InelasticKinematics(delta=0).vmin_from_ER(0) remains NaN, whereas the
exact elastic limit here returns 0.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from src.kinematics.scattering_kinematics import ScatteringKinematics


_ROUNDING_RTOL = 64 * np.finfo(float).eps
_RESIDUAL_RTOL = 4 * _ROUNDING_RTOL


class RelativisticKinematics(ScatteringKinematics):
    """
    Shared, concrete relativistic two-body implementation.

    Parameters
    ----------
    mX, mT : float
        Incoming WIMP mass and target mass, both strictly positive.
    m2 : float
        Outgoing WIMP mass, strictly positive. May be below, equal to,
        or above mX. Use the convenience subclasses for named processes.

    Notes
    -----
    ER denotes target recoil kinetic energy;
    E1 denotes total incoming energy, including rest mass.

    theta in angular methods is the *target recoil* angle relative to the
    incoming WIMP in the laboratory, in [0, pi], not the outgoing WIMP
    angle or a center-of-mass angle.
    At zero incoming momentum the axis can be chosen arbitrarily.

    Domain violations raise ValueError;
    nonscalar masses raise TypeError.
    Forbidden configurations are masked.
    """

    def __init__(self, mX: float, mT: float, m2: float):
        super().__init__(mX=mX, mT=mT)
        if not np.isscalar(m2):
            raise TypeError("Final-state WIMP mass must be scalar")
        if not np.isfinite(m2) or m2 <= 0:
            raise ValueError("Unphysical final-state WIMP mass")
        self.m2 = float(m2)

    @property
    def m1(self) -> float:
        """Incoming WIMP mass"""
        return self.mX

    @property
    def delta(self) -> float:
        """WIMP mass splitting"""
        return self.m2 - self.m1

    @property
    def mass_wimp_final(self) -> float:
        """Outgoing WIMP mass"""
        return self.m2

    @property
    def is_endothermic(self) -> bool:
        """Whether the reaction is endothermic"""
        return self.m2 > self.m1

    @property
    def _reaction_kinetic_threshold(self) -> float:
        """
        Reaction's kinetic threshold,
        allowing for negative values.

        Notes
        ----
        E_thr = m2 + (m2**2-m1**2)/(2*mT);
        K_thr = E_thr - m1.

        The mass-square difference is factorized;
        negative for exothermic scattering.
        """
        return self.delta * (1 + (self.m2 + self.m1) / (2 * self.mT))

    @property
    def threshold_kinetic_energy(self) -> float:
        """
        Physical reaction's kinetic energy threshold.

        Returns only non-negative values.
        """
        return max(0.0, self._reaction_kinetic_threshold)

    @property
    def threshold_energy(self) -> float:
        """
        Minimum total incident energy,
        including incoming rest mass.
        """
        return self.m1 + self.threshold_kinetic_energy

    @property
    def threshold_speed(self) -> float:
        """
        Minimum incident speed capable of producing chi2.

        The value is computed from the exact relativistic
        reaction threshold and adjusted upward by one ULP
        when necessary to ensure that threshold_speed
        always corresponds to an allowed channel.
        """
        k = self.threshold_kinetic_energy
        # Stable form of v_th = sqrt(1 - (m1/(m1+k))**2).
        v = float(np.sqrt(k / (self.m1 + k))
                  * np.sqrt((2 * self.m1 + k) / (self.m1 + k)))
        # For enormous boosts, rounding speed can move energy noticeably
        # below threshold. Choose the next representable speed if needed.
        if v < 1 and self._incident_from_v(v)[2] < k * (1 - _ROUNDING_RTOL):
            v = float(np.nextafter(v, 1.0))
        return v

    @staticmethod
    def _validate_domain_speed(v: ArrayLike,
                               *,
                               require_grid: bool = False
                               ) -> NDArray:
        """
        Extend the base speed-domain check with the relativistic v < 1.
        """
        v = ScatteringKinematics._validate_domain_speed(
            v, require_grid=require_grid
        )
        if np.any(v >= 1):
            raise ValueError("Incoming speed must satisfy v < 1 (c = 1)")
        return v

    def _validate_domain_energy(self, E1: ArrayLike) -> NDArray:
        """
        Validate total energy, independently of final-state threshold.
        """
        E1 = self._validate_domain_nonnegative(E1, "E1")
        if np.any(E1 < self.m1):
            raise ValueError("Total incoming energy E1 must satisfy E1 >= m1")
        return E1

    @staticmethod
    def _validate_domain_theta(theta: ArrayLike) -> NDArray:
        """Validate lab target-recoil angles in radians."""
        theta = np.asarray(theta, dtype=float)
        if np.any(~np.isfinite(theta) | (theta < 0) | (theta > np.pi)):
            raise ValueError("Recoil angle theta must be finite and in [0, pi]")
        return theta

    def speed_mask(self, v: ArrayLike) -> NDArray:
        """
        Allowed incident speeds


        Returns
        -------
        ndarray of bool

            True identifies physically allowed scattering
            configurations. False indicates a closed channel,
            not an invalid input.

        Raises
        ------
        ValueError in case of domain violations.
        """
        v = self._validate_domain_speed(v, require_grid=False)
        K = self._incident_from_v(v)[2]
        return ((v >= self.threshold_speed)
                & (K >= self.threshold_kinetic_energy * (1 - _ROUNDING_RTOL)))

    def energy_mask(self, E1: ArrayLike) -> NDArray:
        """
        Allowed total energies; E1 < m1 is a domain error.

        m1 <= E1 < threshold_energy is a closed channel and returns False.
        """
        E1 = self._validate_domain_energy(E1)
        return E1 >= self.threshold_energy

    def _incident_from_v(self, v: ArrayLike):
        # Fundamental conversion routine used by all speed-based APIs.
        v = self._validate_domain_speed(v, require_grid=False)
        with np.errstate(invalid="ignore", divide="ignore"):
            w = np.sqrt((1 - v) * (1 + v))
            E = self.m1 / w
            p = E * v
            # Stable form of K = sqrt(m1**2+p**2)-m1 = E-m1.
            K = self.m1 * v**2 / (w * (1 + w))
        return E, p, K

    def E1_from_v(self, v: ArrayLike) -> NDArray:
        """Total incident energy from speed;
        
        Invalid speed moduli and v >= 1 raise ValueError.

        Near v=1, floating-point speeds cannot resolve arbitrary boosts;
        use ER_bounds_from_E1 when incident energy is known directly.
        """
        return np.asarray(self._incident_from_v(v)[0])

    def s_from_E1(self, E1: ArrayLike) -> NDArray:
        """
        Mandelstam s evaluated for the initial state.

        Valid incident energies may be below threshold.
        """
        E1 = self._validate_domain_energy(E1)
        s = self.m1**2 + self.mT**2 + 2 * self.mT * E1
        return np.asarray(s)

    def q_from_ER(self, ER: ArrayLike) -> NDArray:
        """
        Spatial momentum transfer
        |q| = sqrt(ER (ER + 2 mT)).
        """
        r = self._validate_domain_ER(ER, require_grid=False)
        # Stable product form of q = sqrt(ER*(ER+2*mT)).
        return np.asarray(np.sqrt(r) * np.sqrt(r + 2 * self.mT))

    def Q2_from_ER(self, ER: ArrayLike) -> NDArray:
        """
        Invariant transfer Q2 = -t = 2 mT ER, not |q| squared.
        """
        ER = self._validate_domain_ER(ER, require_grid=False)
        return np.asarray(2 * self.mT * ER)

    def _bounds(self, E, p, K, valid):
        """
        Compute laboratory recoil-energy endpoints.

        Starting from a valid incident state, determine
        the two extremal recoil configurations corresponding
        to forward and backward scattering in the center-of-mass
        frame, then convert the endpoint target momenta into
        laboratory recoil energies.
        """
        s = (self.m1 + self.mT)**2 + 2 * self.mT * K
        root_s = np.sqrt(s)
        # Physical formula: pf = sqrt(kallen(s,m2**2,mT**2))/(2*root_s).
        # kallen is exposed in utils.py. Keep this specialized factorization
        # here: evaluating kallen(s,...) directly loses precision at threshold.
        above = np.maximum(K - self._reaction_kinetic_threshold, 0.0)
        f1 = 2 * self.mT * above
        f2 = f1 + 4 * self.m2 * self.mT
        pf = np.sqrt(f1) * np.sqrt(f2) / (2 * root_s)
        EN = np.hypot(self.mT, pf)
        boost = E + self.mT
        qlo = (p * EN - boost * pf) / root_s
        qhi = (p * EN + boost * pf) / root_s
        # Stable form of ER = sqrt(mT**2+q**2)-mT at each endpoint.
        lo = qlo * (qlo / (np.hypot(self.mT, qlo) + self.mT))
        hi = qhi * (qhi / (np.hypot(self.mT, qhi) + self.mT))
        if self.m2 == self.m1:
            lo = np.zeros_like(lo)
        return np.where(valid, lo, np.nan), np.where(valid, hi, np.nan)

    def ER_bounds_from_E1(self, E1: ArrayLike) -> tuple[NDArray, NDArray]:
        """
        Recoil endpoints for total incident energy E1 (not kinetic).
        """
        E = np.asarray(E1, dtype=float)
        valid = self.energy_mask(E)
        E = np.where(valid, E, np.nan)
        K = E - self.m1
        # Stable form of p = sqrt(E**2-m1**2).
        p = np.sqrt(K) * np.sqrt(E + self.m1)
        return self._bounds(E, p, K, valid)

    def ER_bounds(self, v: ArrayLike) -> tuple[NDArray, NDArray]:
        """
        Allowed recoil-energy interval for an incident speed v.

        Parameters
        ----------
        v : ArrayLike
            Incoming WIMP speed in units c=1.

        Returns
        -------
        tuple[ndarray, ndarray]
            (ER_min, ER_max) in eV.

        Notes
        -----
        Inputs must satisfy 0 <= v < 1.
        Speeds below the reaction threshold are not domain errors;
        they return NaN endpoints.
        """
        E, p, K = self._incident_from_v(v)
        return self._bounds(E, p, K, self.speed_mask(v))

    def ER_min(self, v: ArrayLike) -> NDArray:
        """
        Lower recoil-energy endpoint for an incident speed v.

        Returns NaN when the reaction channel is closed.
        Domain violations raise ValueError.
        """
        return self.ER_bounds(v)[0]

    def ER_max(self, v: ArrayLike) -> NDArray:
        """
        Upper recoil-energy endpoint for an incident speed v.

        Returns NaN when the reaction channel is closed.
        Domain violations raise ValueError.
        """
        return self.ER_bounds(v)[1]

    def vmin_from_ER(self, ER: ArrayLike) -> NDArray:
        """Exact minimum speed, including exothermic backward recoils.

        Conservation gives p q cos(theta) = E1 ER + B, where
        B = mT ER + (m2**2 - m1**2)/2. Solving |cos(theta)| <= 1
        using rapidity avoids cancellation in gamma_min - 1.

        For zero recoil the result is 0 for elastic scattering, and inf
        for unequal masses (zero transfer cannot change the WIMP mass).
        Negative/nonfinite recoil energies are domain errors (ValueError).
        """
        ER = self._validate_domain_ER(ER, require_grid=False)
        r = np.where(ER > 0, ER, np.nan)
        # Stable factorization of B = mT*ER+(m2**2-m1**2)/2.
        B = self.mT * r + self.delta * (self.m2 + self.m1) / 2
        # Stable rapidity form of vmin = sqrt(1-(m1/E_min)**2);
        # the physical E_min expression is given in the module docstring.
        y = (np.arcsinh(B / (self.m1 * np.sqrt(2 * self.mT * r)))
             + np.arcsinh(np.sqrt(r / (2 * self.mT))))
        result = np.tanh(np.abs(y))
        return np.where(ER == 0, 0.0 if self.delta == 0 else np.inf, result)

    def q_from_v_theta(self,
                       v: ArrayLike,
                       theta: ArrayLike
                       ) -> tuple[NDArray, NDArray]:
        """
        Nonnegative spatial-transfer roots, ordered before masking.

        Inputs broadcast. Each unphysical root is independently NaN;
        exothermic scattering generally has only one physical root.
        Invalid speeds/angles raise ValueError; allowed-domain but forbidden
        speeds/angles produce NaN roots.
        Zero-transfer elastic roots are retained for every valid angle:
        their recoil direction is undefined, but the solution is valid.

        Solve A ER - b q = D, with A = E1 + mT, b = p cos(theta),
        D = (m1**2 - m2**2)/2. Squaring introduces extraneous roots,
        so enforce the original equation and future outgoing energy.
        """
        v, theta = np.broadcast_arrays(
            self._validate_domain_speed(v, require_grid=False),
            self._validate_domain_theta(theta),
        )
        E, p, _ = self._incident_from_v(v)
        A = E + self.mT
        b = p * np.cos(theta)
        D = -self.delta * (self.m2 + self.m1) / 2
        # Stable form of a = A**2-b**2.
        a = (A - b) * (A + b)
        C = A * self.mT + D
        # Stable form of c = A**2*mT**2-C**2.
        c = -D * (2 * A * self.mT + D)
        term1 = (self.mT * b)**2
        term2 = D * (2 * A * self.mT + D)
        disc = term1 + term2
        tolerance = _ROUNDING_RTOL * (np.abs(term1) + np.abs(term2))
        valid = self.speed_mask(v) & (disc >= -tolerance)
        with np.errstate(invalid="ignore", divide="ignore"):
            radical = A * np.sqrt(np.maximum(disc, 0))
            # Stable quadratic roots: calculate the larger-magnitude root
            # without subtraction, then recover the other by their product.
            numerator = C * b + np.copysign(radical, C * b)
            first = numerator / a
            second = np.where(numerator != 0, c / numerator, 0.0)
        roots = (np.minimum(first, second), np.maximum(first, second))
        out = []
        for q in roots:
            # Stable form of recoil = sqrt(mT**2+q**2)-mT.
            recoil = q * (q / (np.hypot(self.mT, q) + self.mT))
            residual = A * recoil - b * q - D
            scale = np.abs(A * recoil) + np.abs(b * q) + abs(D)
            original = np.abs(residual) <= _RESIDUAL_RTOL * scale
            future = E - recoil >= self.m2 - _ROUNDING_RTOL * np.maximum(E, self.m2)
            out.append(np.where(valid & (q >= 0) & original & future, q, np.nan))
        return out[0], out[1]
   
    def speed_theta_mask(self, v: ArrayLike, theta: ArrayLike) -> NDArray:
        """
        Whether at least one physical angular root exists, including q=0.
        """
        lo, hi = self.q_from_v_theta(v, theta)
        return np.isfinite(lo) | np.isfinite(hi)


class RelativisticElasticKinematics(RelativisticKinematics):
    """
    Exact elastic scattering; final WIMP mass equals mX.
    """

    def __init__(self, mX: float, mT: float):
        super().__init__(mX=mX, mT=mT, m2=mX)


class RelativisticInelasticKinematics(RelativisticKinematics):
    """
    Exact inelastic scattering, specified by the final WIMP mass m2.

    m2 > mX is endothermic;
    m2 < mX is exothermic;
    delta is derived.
    Equality is allowed and reproduces RelativisticElasticKinematics.
    """

    def __init__(self, mX: float, mT: float, m2: float):
        super().__init__(mX=mX, mT=mT, m2=m2)
