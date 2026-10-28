"""
Event-rate calculations for dark-matter direct detection.

This module combines
- a halo velocity distribution,
- a WIMP interaction model,
- an optional detector model,

to compute recoil-energy spectra, cumulative rates,
and expected event counts.

The calculation proceeds in two stages:

1. A physical recoil spectrum is computed from the
   dark-matter distribution and interaction model.

2. If a detector is provided, the detector response
   is applied to transform the true recoil spectrum
   into an observed recoil spectrum.

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
Halo models express speeds in km/s.

Interaction and kinematics modules operate on
dimensionless beta = v/c.

The conversion is handled internally by
RateCalculator.

Rate conventions
----------------
If no detector is specified, differential and
integrated rates are returned per target nucleus.

If a detector is specified, detector response and
exposure scaling are applied automatically and the
results correspond to the detector operation.
The differential rate dRdER is multiplied by the
detector exposure Ntargets * exposure_time and
therefore represents
- counts / eV
for the specified detector configuration.

Expected event counts
---------------------
The method ``expected_counts`` returns the total
number of signal events expected during the detector
exposure after all detector-response effects have
been applied.
"""

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import cumulative_trapezoid

from src.detectors.detectors import BaseDetector
from src.halo.base_halo_distribution import IsotropicSpeedDistribution
from src.interactions.base_interaction_model import InteractionModel
from src.kinematics.scattering_kinematics import ScatteringKinematics
from src.utils.constants import CM_TO_M, KM_TO_M, LOCAL_DM_DENSITY_GeV_cm3
from src.utils.conversions import convert_from_kmps_to_beta

rho_dm_eV_cm3 = 1e9 * LOCAL_DM_DENSITY_GeV_cm3

class RateCalculator:
    """
    Compute recoil spectra, integrated rates, and expected
    event counts for a dark-matter direct-detection experiment.

    A RateCalculator combines
    - a halo speed distribution,
    - a WIMP scattering interaction model,
    - an optional detector,

    and provides methods to evaluate differential recoil
    spectra, integrated rates, and total expected counts.

    The calculator is responsible for converting velocity
    conventions between halo models and interaction modules.
    """

    def __init__(self,
                 halo: IsotropicSpeedDistribution,
                 interaction: InteractionModel,
                 rho_dm=rho_dm_eV_cm3,
                 detector: BaseDetector | None = None):
        
        self.halo = halo
        self.interaction = interaction
        self.wimp = interaction.wimp
        self.rho_dm = rho_dm
        self.detector = detector

        self._check_input()

    def dRdER(self, ER: NDArray, num=1000) -> NDArray:
        """
        Differential recoil spectrum.

        Parameters
        ----------
        ER : NDArray
            Recoil-energy grid in eV.

        num : int, optional
            Number of speed samples used in the
            numerical integration.

        Returns
        -------
        NDArray
            Differential recoil spectrum evaluated on
            ``ER``.

        Notes
        -----
        The physical recoil spectrum is computed from
        .. math::

        \\frac{dR}{dE_R}
        =
        \\frac{\\rho_{DM}}{m_\\chi}
        \\int dv\\, g(v)\\, v\\,
        \\frac{d\\sigma}{dE_R}.

        If no detector is associated with the calculator,
        the result is returned per target nucleus with
        units of counts / eV / s.

        If a detector is associated with the calculator,
        detector response effects are applied and the
        result is scaled by the detector exposure,
        .. math::

        N_{targets} \\times t_{exp},

        yielding the expected differential event spectrum
        for the detector configuration.

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

        # apply detector response effects
        # such as threshold, resolution, acceptance
        spectrum = self.detector.apply_detector_response(
            energy=ER,
            spectrum=drate_per_nucleus)

        # return differential rate in eV^-1
        nt = self.detector.number_of_targets
        time_s = self.detector.exposure_seconds
        return nt * time_s * spectrum


    def integrated_rate_above_energy(self, ER: NDArray) -> NDArray:
        """
        Integrated recoil rate above each recoil energy.

        Parameters
        ----------
        ER : ndarray
            Monotonically increasing recoil-energy grid in eV.

        Returns
        -------
        ndarray
            Cumulative rate evaluated on ``ER``.

        The element ``result[i]`` contains
        .. math::

            \\int_{E_i}^{E_{\\max}}
            \\frac{dR}{dE_R}
            \\, dE_R.

        Notes
        -----
        The calculation is performed using a cumulative
        trapezoidal integration from high recoil energies
        toward low recoil energies.

        The returned units follow the convention of
        ``dRdER``:
        - counts / s per target nucleus if no detector
        is present;
        - expected counts if a detector is present.
        """
        rate = self.dRdER(ER)

        integral = cumulative_trapezoid(rate[::-1], -ER[::-1],
                                        initial=0,)[::-1]

        return integral

    def expected_counts(self, num=500) -> float:
        """
        Total expected signal counts.

        Parameters
        ----------
        num : int, optional
            Number of recoil-energy samples used to
            construct the integration grid.

        Returns
        -------
        float
            Total number of signal events expected in
            the detector exposure.

        Raises
        ------
        ValueError
            If no detector is associated with the
            calculator.

        Notes
        -----
        A recoil-energy grid is constructed internally
        between zero recoil energy and the maximum
        kinematically accessible recoil energy.

        Detector-response effects such as thresholds,
        finite energy resolution, and acceptance are
        applied through the detector model before the
        final integration is performed.

        The returned quantity is dimensionless and
        represents the total number of expected signal
        events for the detector configuration.
        """
        if not self.has_detector:
            raise ValueError("Method expected_counts requires a detector.")

        # Start from zero recoil energy rather than the
        # detector threshold. Thresholds are implemented
        # through the detector-response model and may
        # interact with energy smearing.

        # MINOR-MINOR:ISSUE:
        # Current detector-response calculations use a 
        # recoil-energy grid terminating at the kinematic 
        # endpoint EmaxE​. 
        # For detectors with finite energy resolution,
        # a small fraction of events may be reconstructed above 
        # Emax​ and are therefore lost.
        # The effect is expected to be negligible 
        # but could be eliminated in the 
        # future by introducing separate true-energy and 
        # reconstructed-energy grids or by extending the 
        # reconstructed-energy range beyond the kinematic endpoint.

        ER = np.linspace(0., self.ER_max, num=num)
        integrated_counts = self.integrated_rate_above_energy(ER)

        return integrated_counts[0]

    def _check_input(self):
        """
        Check that the inputs are valid
        """
        if not np.isfinite(self.rho_dm) or self.rho_dm < 0:
            raise ValueError("Local DM density must be finite and non negative")

        if self.has_detector:
            if self.detector.nucleus != self.interaction.nucleus:
                raise ValueError("Detector and interaction nuclei are different")

    @property
    def wimp_number_density(self) -> float:
        """
        Local WIMP number density.

        Returns
        -------
        float
            Number density corresponding to the local
            dark-matter density over the WIMP mass.
        """
        return self.rho_dm / self.wimp.mass

    @property
    def mass_wimp(self) -> float:
        """
        WIMP mass in eV.
        """
        return self.wimp.mass

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
        """
        Whether a detector is associated with the
        calculator.
        """
        return self.detector is not None

    @property
    def sigma_proton(self) -> float:
        """
        WIMP-proton cross-section parameter used to
        normalize the interaction model.

        Returns
        -------
        float
            Reference WIMP-proton cross section in cm^2.

        Notes
        -----
        Sensitivity calculations operate on this
        quantity and therefore require interaction
        models to expose a meaningful proton-level
        cross-section normalization.
        """
        return self.interaction.sigma_proton
