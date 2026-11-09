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

import warnings
import numpy as np
from numpy.typing import NDArray
from scipy.integrate import cumulative_trapezoid

from src.nuclei.nucleus import Nucleus
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
        - counts / eV
        in the given mass * exposure.
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


    def integrated_rate_above_energy(self, ER: NDArray,
                                     num=1000) -> NDArray:
        """
        Integrated recoil rate above each recoil energy.

        Parameters
        ----------
        ER : ndarray
            Strictly increasing recoil-energy grid in eV.

        num : int, optional
            Number of speed samples used in the
            numerical integration.

        Returns
        -------
        ndarray
            Cumulative rate evaluated on ``ER``.

        The element ``result[i]`` contains
        .. math::

            \\int_{E_i}^{E_{\\max}}
            \\frac{dR}{dE_R}
            \\, dE_R.

        Raises
        ------
        ValueError
            If ``ER`` is not strictly increasing.

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
        ER = np.asarray(ER)

        if np.any(np.diff(ER) <= 0):
            raise ValueError("ER must be strictly increasing.")

        rate = self.dRdER(ER=ER, num=num)

        integral = cumulative_trapezoid(rate[::-1], -ER[::-1],
                                        initial=0,)[::-1]

        return integral

    def normalized_spectrum(self, ER: NDArray, num=1000) -> NDArray:
        """
        Normalized recoil-energy probability density.

        Parameters
        ----------
        ER : NDArray
            Strictly increasing recoil-energy grid in eV.

        num : int, optional
            Number of speed samples used in the velocity
            integration.

        Returns
        -------
        NDArray
            Probability density evaluated on ``ER``.

        Notes
        -----
        The returned quantity is a continuous probability
        density function (PDF),

        .. math::

        p(E_R) = \frac{1}{N} \frac{dR}{dE_R},

        where

        .. math::

        N = \\int dE_R \\, \\frac{dR}{dE_R}.

        The normalization satisfies

        .. math::

        \\int p(E_R) dE_R = 1.

        Consequently, the values returned by this method are
        probability densities rather than discrete
        probabilities. Summing the array elements does not
        generally yield unity.

        If the total expected rate is zero, an array of zeros
        is returned.
        """
        ER = np.asarray(ER)

        norm = self.integrated_rate_above_energy(ER=ER, num=num)[0]

        if norm > 0:
            return self.dRdER(ER=ER, num=num) / norm

        if norm == 0:
            return np.zeros_like(ER)

        if norm < 0:
            raise RuntimeError("Negative norm in dR/dER.")

    def interpolate_normalized_spectrum(self,
                                        ER : NDArray,
                                        num=500) -> NDArray:
        """
        Interpolate the normalized recoil spectrum.

        Parameters
        ----------
        ER : NDArray
            Recoil-energy sample points grid in eV.

        num : int, optional
            Number of recoil-energy samples used to
            construct the interpolation grid.

        Returns
        -------
        NDArray
            Interpolated differential recoil spectrum,
            normalized to one, evaluated on ``ER``.
        """
        ER_grid = np.linspace(0., self.ER_max, num=num)
        f_grid = self.normalized_spectrum(ER=ER_grid)

        return np.interp(ER, ER_grid, f_grid, left=0.0, right=0.0)

    def sample_recoil_energies(self, size=1,
                               seed : int = None) -> NDArray:
        """
        Draw recoil-energy samples from the normalized spectrum.

        Parameters
        ----------
        size : int, optional
            Number of recoil energies to generate.

        seed : int or None, optional
            Seed passed to ``numpy.random.default_rng``.
            If ``None``, NumPy chooses a
            non-deterministic seed.

        Returns
        -------
        NDArray
            Sampled recoil energies in eV.

        Raises
        ------
        ValueError
            If the total expected event rate is zero.

        Notes
        -----
        Samples are generated using inverse-transform
        sampling.

        A recoil-energy grid between zero and ``ER_max`` is
        constructed internally. The cumulative distribution
        function is obtained from the integrated recoil rate,

        .. math::

        F(E_R) = 1 - \frac{R(>E_R)}{R(>0)}.

        Uniform random numbers are then mapped to recoil
        energies through interpolation of the cumulative
        distribution.

        The generated samples therefore represent draws from
        the continuous recoil-energy probability density
        defined by ``normalized_spectrum``.

        When a detector is attached to the rate calculator,
        sampling is performed after detector-response effects
        have been applied. The returned energies therefore
        correspond to reconstructed detector energies rather
        than true recoil energies.

        The sampling distribution is approximated using a
        finite recoil-energy grid and linear interpolation.
        Increasing the internal grid density improves the
        accuracy of the approximation.
        """

        rng = np.random.default_rng(seed=seed)
        ER = np.linspace(0., self.ER_max, 2000)

        surv = self.integrated_rate_above_energy(ER)
        total = surv[0]

        if total <= 0:
            raise ValueError("Cannot sample recoil energies because the"
                             " expected event rate is zero.")

        surv /= total
        cdf = 1.0 - surv
        u = rng.random(size)

        return np.interp(u, cdf, ER)
    

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

    def expected_cross_section_from_counts(self,
                                           counts: float,
                                           num: int = 500) -> float:
        """
        Compute the WIMP-nucleon reference cross section required to
        produce a specified number of expected signal counts.

        Parameters
        ----------
        counts : float
            Target number of expected signal events.
        num : int, optional
            Number of recoil-energy samples used in the
            internal expected-counts calculation.

        Returns
        -------
        float
            Required interaction model reference cross section
            in cm^2.

            Returns ``np.nan`` when the reference model predicts
            zero expected counts.
            In this case no finite rescaling of the reference
            cross section can produce the requested signal yield.

        Notes
        -----
        Assumes the event rate scales linearly with the
        interaction-model reference cross section
        ``sigma_reference``.

        When the reference model predicts zero expected counts,
        the required cross section is undefined and ``np.nan``
        is returned. This behavior is intended to support
        parameter-space scans, where some benchmark points may
        be kinematically inaccessible or otherwise produce no
        observable events.
        """
        if counts < 0:
            raise ValueError("counts must be non-negative")
        
        if counts == 0:
            return 0.0

        reference_counts = self.expected_counts(num=num)

        if reference_counts < 0:
            raise RuntimeError("Expected counts became negative.")

        if reference_counts == 0:
            warnings.warn("Model predicts zero expected counts.",
                          RuntimeWarning)
            return np.nan

        return self.sigma_reference * counts / reference_counts

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
    def sigma_reference(self) -> float:
        """
        Interaction-model reference cross-section parameter.

        Returns
        -------
        float
            Reference interaction cross section in cm^2.

        Notes
        -----
        Sensitivity calculations operate on this
        quantity and therefore require interaction
        models to expose a meaningful reference
        cross-section.
        """
        return self.interaction.sigma_reference

    @property
    def target_nucleus(self) -> Nucleus:
        """
        Target nucleus.
        """
        return self.interaction.nucleus
