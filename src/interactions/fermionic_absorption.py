"""
Monochromatic neutral-current fermionic DM absorption.

This module uses the normalization of the review note FermionicXS.pdf:
d_sigma/d_ER = sigma_abs * A**2 * (m_n/m_N)**2 * response(q_abs)
               * delta(E_R-E_abs) / beta.
Here m_N is the nuclear mass. The proton mass is used for the reference
nucleon mass m_n, consistently with the proton reference in SIInteraction.
The recoil is E_abs = m_chi**2/(2*m_N), at leading order in beta and m_chi/m_N.

sigma_abs is the per-nucleon input coefficient defined in the note.
sigma_reference returns this input, so count inference and target matching
use the note's convention. The derived property sigma_NC gives
sigma_abs*(m_n/m_N)**2, the coefficient multiplying A**2 in Dror's expression
for this target. It is not a second independent input: at fixed sigma_abs,
sigma_NC depends on the target. A converted single-target benchmark does
not independently validate the note's relative target scaling.

Units
-----
Mass, recoil energy and momentum: eV (c=1). Speed input: beta=v/c.
sigma_abs and sigma_NC: cm^2. Differential cross section: cm^2/eV.
For beta > 0 and point-like responses, beta*sigma_A = A**2*sigma_NC.
At beta=0, an all-zero row is returned as a numerical convention.

References
----------
FermionicXS.pdf, review note, final differential-cross-section expression.
Dror, Elor and McGehee, arXiv:1908.10861v2, Eqs. (3.4)-(3.8).
"""

import numpy as np
from numpy.typing import NDArray

from src.interactions.base_interaction_model import InteractionModel
from src.interactions.particles import WIMP
from src.interactions.propagators import Propagator
from src.kinematics.absorption_kinematics import AbsorptionKinematics
from src.nuclei.base_nuclear_response import NuclearResponse
from src.nuclei.nucleus import Nucleus
from src.utils.constants import PROTON_MASS_MeV


class FermionicAbsorptionInteraction(InteractionModel):
    """
    Monochromatic fermionic DM absorption, chi + N -> nu + N.

    Parameters
    ----------
    wimp : WIMP
        Incident DM particle; mass in eV.
    nucleus : Nucleus
        Recoiling target nucleus; mass in eV.
    kinematics : AbsorptionKinematics
        Leading-order absorption kinematics, with matching particle masses.
    sigma_abs : float, optional
        Finite, non-negative per-nucleon coefficient defined in the review
        note, in cm^2. Default is 1e-40 cm^2. The interaction applies both
        A**2 and (m_n/m_N)**2 internally. This input is not Dror's sigma_NC;
        use the sigma_NC property to obtain the converted coefficient.
    nuclear_response : NuclearResponse, optional
        Dimensionless cross-section response. Defaults to point-like.
        FormFactor instances already return F(q)^2 through response(q).
    propagator : Propagator, optional
        Dimensionless cross-section response. Defaults to point-like.

    Notes
    -----
    For beta > 0, each speed row contains the same recoil line, with height
    proportional to 1/beta. Multiplication by the physical incident speed
    in RateCalculator cancels this dependence, giving the velocity-independent
    leading-order rate. At beta=0, the row is set to zero as a numerical
    convention, not as the physical limit of the cross section.
    Scattering kinematic masks are not applied. Nuclear and propagator
    responses are evaluated at the true absorption energy, even when its
    nearest grid point has a slightly different energy.
    """

    def __init__(
        self,
        *,
        wimp: WIMP,
        nucleus: Nucleus,
        kinematics: AbsorptionKinematics,
        sigma_abs: float = 1e-40,
        nuclear_response: NuclearResponse | None = None,
        propagator: Propagator | None = None,
    ) -> None:
        if (not isinstance(kinematics, AbsorptionKinematics)
                or kinematics.supports_scattering_interactions):
            raise ValueError(
                "FermionicAbsorptionInteraction requires AbsorptionKinematics "
                "that does not support scattering interactions."
            )
        if not np.isfinite(sigma_abs) or sigma_abs < 0:
            raise ValueError("sigma_abs must be finite and non-negative.")

        super().__init__(
            wimp=wimp,
            nucleus=nucleus,
            kinematics=kinematics,
            nuclear_response=nuclear_response,
            propagator=propagator,
        )
        if not np.isfinite(self.kin.ER) or self.kin.ER <= 0:
            raise ValueError("The absorption recoil energy must be finite and positive.")
        self.sigma_abs = float(sigma_abs)

    @staticmethod
    def _check_input_v_ER(v: NDArray, ER: NDArray) -> tuple[NDArray, NDArray]:
        """
        Convert and validate model-independent speed and recoil-energy grids.

        Parameters
        ----------
        v : ndarray, shape (Nv,)
            Non-empty, finite, non-negative speed grid in units beta=v/c.
        ER : ndarray, shape (NER,)
            Non-empty, finite, non-negative, strictly increasing energy grid.

        Returns
        -------
        tuple of ndarray
            Validated floating-point arrays (v, ER).

        Raises
        ------
        ValueError
            If either grid is empty, not one-dimensional or invalid.
        """
        v = np.asarray(v, dtype=float)
        ER = np.asarray(ER, dtype=float)
        if v.ndim != 1 or ER.ndim != 1 or v.size == 0 or ER.size == 0:
            raise ValueError("v and ER must be non-empty one-dimensional arrays.")
        if not np.all(np.isfinite(v)) or np.any(v < 0):
            raise ValueError("v (beta=v/c) must be finite and non-negative.")
        if not np.all(np.isfinite(ER)) or np.any(ER < 0):
            raise ValueError("ER must be finite and non-negative.")
        spacing = np.diff(ER)
        if not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
            raise ValueError("ER must be strictly increasing with finite spacing.")
        return v, ER

    def dsigma_dER(self, v: NDArray, ER: NDArray) -> NDArray:
        """
        Evaluate the discretized monochromatic differential cross section.

        Parameters
        ----------
        v : ndarray, shape (Nv,)
            Non-empty, finite, non-negative beta=v/c grid.
            Entries at beta=0 produce all-zero rows as a numerical convention.
        ER : ndarray, shape (NER,)
            Finite, non-negative, strictly increasing recoil-energy grid
            in eV. At least two points are required to define an area.

        Returns
        -------
        ndarray, shape (Nv, NER)
            Differential cross section in cm^2/eV. All entries are zero
            when the absorption energy lies outside the supplied grid.

        Raises
        ------
        ValueError
            If input grids are empty, not one-dimensional or invalid,
            or ER contains fewer than two points.

        Notes
        -----
        The single non-zero entry for each positive speed is divided by its
        trapezoidal weight: half the adjacent spacing at an endpoint, or half
        the distance between neighboring points in the interior. For beta > 0,
        integration over a grid containing the line returns
        A**2*sigma_abs*(m_n/m_N)**2/beta times the momentum-transfer response,
        including for non-uniform grids. This is A**2*sigma_NC/beta using
        the converted coefficient exposed by the sigma_NC property.
        """
        v, ER = self._check_input_v_ER(v, ER)
        if ER.size < 2:
            raise ValueError("ER must contain at least two points.")

        dsigma = np.zeros((v.size, ER.size), dtype=float)
        ER_abs = self.kin.ER
        if ER_abs < ER[0] or ER_abs > ER[-1]:
            return dsigma

        idx = int(np.argmin(np.abs(ER - ER_abs)))
        if idx == 0:
            dER = 0.5 * (ER[1] - ER[0])
        elif idx == ER.size - 1:
            dER = 0.5 * (ER[-1] - ER[-2])
        else:
            dER = 0.5 * (ER[idx + 1] - ER[idx - 1])

        response = self.momentum_transfer_response(ER=np.array([ER_abs]))[0]
        safe_v = np.where(v > 0, v, np.inf)
        dsigma[:, idx] = self.nucleus.A**2 * self.sigma_NC * response / (safe_v * dER)
        return dsigma

    @property
    def sigma_reference(self) -> float:
        """Return the note's input coefficient sigma_abs, in cm^2."""
        return self.sigma_abs

    @property
    def sigma_NC(self) -> float:
        """
        Return the coefficient in Dror's normalization for this target.

        The conversion is sigma_NC = sigma_abs*(m_n/m_N)**2, in cm^2,
        where m_n is the reference proton mass and m_N the nuclear mass.
        At fixed sigma_abs this property is target dependent. It does not
        include A**2, the nuclear response or the propagator response.
        """
        m_n = 1e6 * PROTON_MASS_MeV
        return self.sigma_abs * (m_n / self.nucleus.mass)**2


