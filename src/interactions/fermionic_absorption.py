"""
Monochromatic neutral-current fermionic DM absorption in the Dror convention.

At leading order in the incident speed and m_chi/m_N, the recoil is fixed at
E_abs = m_chi**2/(2*m_N), while the cross section scales as 1/beta, beta=v/c.

This module defines sigma_abs = beta * sigma_A(beta, q=0), in cm^2,
and implements d_sigma/d_ER = sigma_abs * response(q_abs) * delta(E_R-E_abs) / beta.
For the isospin-conserving coherent NC operator of Dror et al.,
sigma_abs = A**2 * sigma_NC, with sigma_NC the paper's per-nucleon parameter.
The A**2 factor is therefore supplied in sigma_abs, not applied again here.

Units
-----
Mass, recoil energy and momentum: eV (c=1). Speed input: beta=v/c.
Cross section and sigma_abs: cm^2. Differential cross section: cm^2/eV.
The physical velocity-weighted coefficient is sigma*v = c*sigma_abs,
before nuclear and propagator responses. sigma_abs is not a constant
physical total cross section: sigma_A(beta)=sigma_abs/beta for point-like responses.

References
----------
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
        Finite, non-negative per-nucleus coefficient beta*sigma_A(beta,q=0)
        in cm^2, before momentum-transfer responses. Default is 1e-40 cm^2.
        For Dror's coherent isoscalar NC interaction, pass A**2*sigma_NC.
    nuclear_response : NuclearResponse, optional
        Dimensionless cross-section response. Defaults to point-like.
        FormFactor instances already return F(q)^2 through response(q).
    propagator : Propagator, optional
        Dimensionless cross-section response. Defaults to point-like.

    Notes
    -----
    Each speed row contains the same recoil line, with height proportional
    to 1/beta. Multiplication by the physical incident speed in RateCalculator
    cancels this dependence, giving the velocity-independent leading-order rate.
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

    def dsigma_dER(self, v: NDArray, ER: NDArray) -> NDArray:
        """
        Evaluate the discretized monochromatic differential cross section.

        Parameters
        ----------
        v : ndarray, shape (Nv,)
            Non-empty, finite, strictly positive beta=v/c grid.
            beta=0 is excluded because the cross section diverges as 1/beta;
            its physical product with speed has a finite limiting value.
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
            or the local integration weight is not finite and positive.

        Notes
        -----
        The single non-zero entry is divided by its trapezoidal weight:
        half the adjacent spacing at an endpoint, or half the distance
        between neighboring points in the interior. Therefore integration
        over a grid containing the line returns sigma_abs/beta times the
        momentum-transfer response, including for non-uniform grids.
        """
        v = np.asarray(v, dtype=float)
        ER = np.asarray(ER, dtype=float)
        if v.ndim != 1 or ER.ndim != 1 or v.size == 0 or ER.size == 0:
            raise ValueError("v and ER must be non-empty one-dimensional arrays.")
        if ER.size < 2:
            raise ValueError("ER must contain at least two points.")
        if not np.all(np.isfinite(v)) or np.any(v <= 0):
            raise ValueError("v (beta=v/c) must be finite and strictly positive.")
        if not np.all(np.isfinite(ER)) or np.any(ER < 0):
            raise ValueError("ER must be finite and non-negative.")
        spacing = np.diff(ER)
        if not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
            raise ValueError("ER must be strictly increasing with finite spacing.")

        dsigma = np.zeros((v.size, ER.size), dtype=float)
        ER_abs = self.kin.ER
        if ER_abs < ER[0] or ER_abs > ER[-1]:
            return dsigma

        idx = int(np.argmin(np.abs(ER - ER_abs)))
        if idx == 0:
            dER = 0.5 * spacing[0]
        elif idx == ER.size - 1:
            dER = 0.5 * spacing[-1]
        else:
            dER = 0.5 * (spacing[idx - 1] + spacing[idx])
        if not np.isfinite(dER) or dER <= 0:
            raise ValueError("The local energy integration weight must be positive and finite.")

        response = self.momentum_transfer_response(ER=np.array([ER_abs]))[0]
        dsigma[:, idx] = self.sigma_abs * response / (v * dER)
        return dsigma

    @property
    def sigma_reference(self) -> float:
        """Return sigma_abs = beta*sigma_A(beta,q=0), in cm^2."""
        return self.sigma_abs

