"""
Event-rate calculations for dark-matter
direct detection.

This module combines
- a WIMP model,
- a halo velocity distribution,
- a scattering interaction,
- an optional detector exposure,

to compute recoil-energy spectra and
integrated event rates.

Units
-----
mass      -> eV
energy    -> eV
momentum  -> eV
density   -> eV/cm^3
cross section -> cm^2
rate      -> s^-1

Velocity conventions
--------------------
Halo models use speeds in km/s.

Interaction and kinematics modules use
dimensionless beta = v/c.

The conversion is handled internally by
RateCalculator.

Count rate conventions
----------------------
If no detector is specified, the returned
quantity is dR/dER with units
- counts / eV / s
for a single target nucleus.

If an IdealDetector is provided
in the class initializer, the result
is multiplied by the detector exposure
 Ntargets * exposure_time
and therefore represents
- counts / eV
for the specified detector configuration.
"""

from src.interactions.particles import WIMP
from src.halo import IsotropicVelocityDistribution
from src.kinematics import ScatteringKinematics
from src.interactions.base_interaction_model import InteractionModel
from src.detectors.detectors import IdealDetector
from src.utils.conversions import convert_from_kmps_to_beta
from src.utils.constants import LOCAL_DM_DENSITY_GeV_cm3, CM_TO_M, KM_TO_M

from scipy.integrate import cumulative_trapezoid
from numpy.typing import NDArray
import numpy as np

rho_dm_eV_cm3 = 1e9 * LOCAL_DM_DENSITY_GeV_cm3

class RateCalculator:

    def __init__(self,
                 wimp: WIMP,
                 halo: IsotropicVelocityDistribution,
                 interaction: InteractionModel,
                 rho_dm=rho_dm_eV_cm3,
                 detector: IdealDetector | None = None):
        
        self.wimp = wimp
        self.halo = halo
        self.interaction = interaction
        self.rho_dm = rho_dm
        self.detector = detector

    def dRdER(self, ER: NDArray, num=1000):
        """
        Differential recoil rate.

        Parameters
        ----------
        ER : ndarray
            Recoil-energy grid in eV.

        num : int, optional
            Number of velocity samples used in
            the numerical integration.

        Returns
        -------
        ndarray
            Differential recoil rate dR/dER.
            
        If detector is None:
            counts / eV / s
            per target nucleus

        If detector is specified:
            counts / eV
        for the detector exposure
        specified by Ntargets * exposure_time

        Notes
        -----
        The velocity integration is performed
        internally from v = 0 to the maximum
        speed supported by the halo model.

        Halo speeds are expressed in km/s and
        are converted internally to beta = v/c
        before being passed to the interaction
        and kinematics modules.

        The calculation evaluates

        dR/dER = (rho_DM / mX) ∫ dv g(v) v (dσ/dER)

        If no detector is specified, the returned
        quantity is dR/dER with units
        - counts / eV / s
        for a single target nucleus.

        If an IdealDetector was provided
        in the class initializer, the
        result is multiplied by the detector
        exposure Ntargets * exposure_time
        and therefore represents
        - counts / eV
        for the specified detector configuration.

        Examples
        --------
        mass_kg=1
        exposure_days=365
        returns
        - counts / eV / kg / year
        after appropriate energy-unit conversion.
        """

        vmin = self.interaction.scattering_kinematics.threshold_speed
        vmin = max(1e-6, vmin)  # avoids possible division by zero
        vmax = self.halo.maximum_speed
        v = np.linspace(vmin, vmax, num=num)

        gv = self.halo.speed_pdf(v)

        # note: dsigma_dER accepts v in units c=1
        beta = convert_from_kmps_to_beta(v)

        # dsigma/dER, in cm^2 / eV
        dsigma = self.interaction.dsigma_dER(ER=ER, v=beta)

        # integral <v dsigma/dE>
        integrand = (v[:, None] * gv[:, None] * dsigma)
        integral = np.trapezoid(integrand, v, axis=0)

        # note: currently n <v dsigma/dE>
        # would have units cm^-3 * km/s * cm^2/eV
        # that is cm^-1 * km / s / eV
        # The conversion factor is written pedantically.
        to_unit_s_eV = KM_TO_M / CM_TO_M

        # differential rate in s^-1 ev^-1 (per single target nucleus)
        drate_per_nucleus = self.wimp_number_density * integral * to_unit_s_eV

        if self.detector is None:
            return drate_per_nucleus

        # return differential rate in eV^-1
        nt = self.detector.number_of_targets
        time_s = self.detector.exposure_seconds
        return nt * time_s * drate_per_nucleus


    def integrated_rate_above_threshold(self, ER: NDArray) -> NDArray:
        rate = self.dRdER(ER)

        integral = cumulative_trapezoid(rate[::-1], -ER[::-1],
                                        initial=0,)[::-1]

        return integral

    @property
    def wimp_number_density(self) -> float:
        return self.rho_dm / self.wimp.mass

    @property
    def ER_max(self) -> float:
        """
        Maximum recoil energy kinematically
        accessible for the fastest WIMPs in
        the halo model.
        """
        vmax = self.halo.maximum_speed
        beta = convert_from_kmps_to_beta(vmax)

        return float(self.scattering_kinematics.ER_max(beta))

    @property
    def scattering_kinematics(self) -> ScatteringKinematics:
        return self.interaction.scattering_kinematics

    @property
    def has_detector(self) -> bool:
        return self.detector is not None

    