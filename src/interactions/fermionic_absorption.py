"""
Coherent, monochromatic neutral-current fermionic DM absorption.

For chi + A -> nu + A, this module implements

    E_abs = m_chi**2 / (2*m_A),  q_abs = m_chi,
    d_sigma_A/d_ER = sigma_NC * A**2 * response(q_abs)
                    * delta(E_R-E_abs) / beta.

The input sigma_NC is Dror's target-independent per-nucleon coefficient
sigma_NC = m_chi**2 / (4*pi*Lambda**4), converted to cm^2. It is the
velocity-weighted coefficient beta*sigma_NC for a point-like nucleon,
not a velocity-independent physical cross section. The sigma_reference
property returns sigma_NC. The coherent factor A**2 is applied once,
inside this class; there is no additional nuclear-mass rescaling.

response(q) is the product of the nuclear and propagator responses.
FormFactor.response already contains F(q)**2 and must not be squared again.
For a normalized halo, multiplication by the physical flux in RateCalculator
gives an integrated rate per nucleus of

    (rho_chi/m_chi) * c * sigma_NC * A**2 * response(q_abs).

The detector supplies the number of target nuclei and exposure time.

Units
-----
Mass, recoil energy and momentum: eV (c=1). Speed input: beta=v/c.
sigma_NC: cm^2. Differential cross section: cm^2/eV.

References
----------
J. A. Dror, G. Elor and R. McGehee,
"Absorption of Fermionic Dark Matter by Nuclear Targets",
Journal of High Energy Physics, JHEP 02 (2020) 134.
DOI: 10.1007/JHEP02(2020)134.
arXiv:1908.10861v2 [hep-ph], Eqs. (3.4)-(3.8).
The implementation retains only the leading monochromatic, coherent
isoscalar contribution for non-relativistic DM, a massless neutrino and
m_chi << m_A. Velocity broadening and incoherent absorption are excluded.
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
    sigma_NC : float, optional
        Finite, non-negative per-nucleon coefficient sigma_NC of Dror,
        in cm^2. Default is 1e-40 cm^2. Use the same value for all targets
        at fixed DM mass and microscopic coupling. The interaction applies
        A**2 internally. For point-like responses and beta > 0, the total
        physical nuclear cross section is A**2*sigma_NC/beta.
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
        sigma_NC: float = 1e-40,
        nuclear_response: NuclearResponse | None = None,
        propagator: Propagator | None = None,
    ) -> None:
        if (not isinstance(kinematics, AbsorptionKinematics)
                or kinematics.supports_scattering_interactions):
            raise TypeError(
                "FermionicAbsorptionInteraction requires AbsorptionKinematics "
                "that does not support scattering interactions."
            )
        if not np.isfinite(sigma_NC) or sigma_NC < 0:
            raise ValueError("sigma_NC must be finite and non-negative.")

        super().__init__(
            wimp=wimp,
            nucleus=nucleus,
            kinematics=kinematics,
            nuclear_response=nuclear_response,
            propagator=propagator,
        )
        if not np.isfinite(self.kin.ER) or self.kin.ER <= 0:
            raise ValueError("The absorption recoil energy must be finite and positive.")
        self.sigma_NC = float(sigma_NC)

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
        A**2*sigma_NC/beta times the momentum-transfer response,
        including for non-uniform grids. The line must lie inside the grid;
        include its true energy explicitly when applying sharp cuts.
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
        dsigma[:, idx] = self.sigma_A * response / (safe_v * dER)
        return dsigma

    @property
    def sigma_reference(self) -> float:
        """Return the target-independent per-nucleon coefficient, in cm^2."""
        return self.sigma_NC

    @property
    def sigma_A(self) -> float:
        """
        Return the coherent nuclear coefficient A**2*sigma_NC, in cm^2.

        Excludes nuclear and propagator responses and the 1/beta factor
        of the physical absorption cross section.
        """
        return self.nucleus.A**2 * self.sigma_NC

