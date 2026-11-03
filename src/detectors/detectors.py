"""
Detector models for DMRecoilLab.

This module defines detector descriptions used to convert
physical recoil spectra into observable spectra.

The detector hierarchy separates interaction physics from
detector response effects:

    BaseDetector
        Abstract detector interface.

    IdealDetector
        Ideal detector with perfect energy resolution,
        unit acceptance and a hard analysis threshold.

    ResponseDetector
        Detector model including finite energy resolution
        and energy-dependent acceptance.

Detector response is applied through the
``apply_detector_response`` interface, allowing
RateCalculator to remain independent of the
specific detector implementation.

All detector responses are assumed to act on recoil-energy
spectra expressed as dR/dER.
"""

from numpy.typing import NDArray
import numpy as np
from scipy.interpolate import interp1d

from src.nuclei.nucleus import Nucleus
from src.utils.conversions import convert_from_kg_to_eV
from src.utils.constants import DAY_TO_SECONDS
from src.utils.tabulated_data import TabulatedData

from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass(frozen=True)
class BaseDetector(ABC):
    """
    Abstract detector description.

    Parameters
    ----------
    nucleus : Nucleus
        Target isotope.

    mass_kg : float
        Active detector fiducial mass in kg.

    exposure_days : float, optional
        Exposure time in days.
        Default is 365.

    target_mass_fraction : float, optional
        Mass fraction of the target isotope in the active
        detector mass. Default is 1.0.

    Notes
    -----
    This class stores detector exposure information and
    defines the common interface used to apply detector
    response effects to recoil-energy spectra.

    Derived classes implement specific detector models,
    ranging from ideal threshold detectors to detectors
    with finite resolution and energy-dependent acceptance.
    """
    nucleus: Nucleus
    mass_kg: float
    exposure_days: float = 365
    target_mass_fraction: float = 1.0

    def __post_init__(self):
        """
        Validates the input.
        """
        if not np.isfinite(self.mass_kg) or self.mass_kg < 0:
            raise ValueError("mass_kg must be finite and non negative.")
        if not np.isfinite(self.target_mass_fraction) or not 0 <= self.target_mass_fraction <= 1:
            raise ValueError("target_mass_fraction must be finite, >= 0, and <= 1.")
        if not np.isfinite(self.exposure_days) or self.exposure_days < 0:
            raise ValueError("exposure_days must be a finite non-negative number.")

    @property
    def number_of_targets(self) -> float:
        """
        Number of target nuclei contained in the detector.

        Returns
        -------
        float
            Effective number of target nuclei.

        Notes
        -----
        The result is computed as
        N = M_det * f / M_T
        where
        - M_det is the detector mass (converted to eV),
        - f is the target mass fraction,
        - M_T is the target-nucleus mass (eV).
        """
        return convert_from_kg_to_eV(self.mass_kg * self.target_mass_fraction) / self.nucleus.mass

    @property
    def exposure_seconds(self) -> float:
        """
        Exposure time in seconds.

        Returns
        -------
        float
            Detector exposure converted from days to seconds.
        """
        return DAY_TO_SECONDS * self.exposure_days

    @abstractmethod
    def apply_detector_response(self,
                                energy: NDArray,
                                spectrum: NDArray) -> NDArray:
        """
        Apply the detector response to a recoil spectrum.

        Parameters
        ----------
        energy : NDArray
            Recoil-energy grid in eV.

        spectrum : NDArray
            Differential recoil spectrum evaluated on
            ``energy``.

        Returns
        -------
        NDArray
            Detector-modified spectrum evaluated on the
            same energy grid.

        Notes
        -----
        Implementations must not modify the input arrays
        in-place.

        Examples of detector response include:
        - analysis thresholds,
        - finite energy resolution,
        - energy-dependent acceptance,
        """
        pass


@dataclass(frozen=True)
class IdealDetector(BaseDetector):
    """
    Ideal detector model.

    Parameters
    ----------
    threshold_energy : float, optional
        Analysis threshold in eV.
        The observed energy spectrum below
        this threshold is set to zero.
    
    Notes
    -----
    This detector assumes:
    - perfect energy resolution,
    - unit acceptance at all energies,
    - a hard analysis threshold.

    The detector response is implemented by setting the
    spectrum to zero below ``threshold_energy``.
    """
    threshold_energy: float = 0.0

    def __post_init__(self):
        """
        Validates the input.
        """
        super().__post_init__()

        if not np.isfinite(self.threshold_energy) or self.threshold_energy < 0:
            raise ValueError("threshold_energy must be a finite non-negative number.")

    def apply_detector_response(self,
                                energy: NDArray,
                                spectrum: NDArray) -> NDArray:
        """
        Apply the ideal-detector response.

        Parameters
        ----------
        energy : NDArray
            Recoil-energy grid in eV.

        spectrum : NDArray
            Differential recoil spectrum.

        Returns
        -------
        NDArray
            Copy of the input spectrum with all values below
        ``threshold_energy`` set to zero.
        """
        result = spectrum.copy()
        result[energy < self.threshold_energy] = 0
        return result

@dataclass(frozen=True)
class ResponseDetector(BaseDetector):
    """
    Detector model with finite resolution and
    energy-dependent acceptance.

    Parameters
    ----------
    resolution_sigma : float, optional
        Gaussian energy-resolution width in eV.

    acceptance_curve : TabulatedData, optional
        Tabulated detector acceptance as a function
        of recoil energy.

    Notes
    -----
    The detector response is applied in two stages:
    1. Gaussian energy smearing.
    2. Energy-dependent acceptance.

    The observed spectrum is therefore:

    true spectrum
        ↓
    resolution
        ↓
    acceptance
        ↓
    observed spectrum

    If ``resolution_sigma`` is zero, perfect energy
    resolution is assumed.

    If ``acceptance_curve`` is None, unit acceptance
    is assumed at all energies.
    """
    resolution_sigma: float = 0.
    acceptance_curve: TabulatedData | None = None

    def __post_init__(self):
        """
        Validates the input.
        """
        super().__post_init__()

        if not np.isfinite(self.resolution_sigma) or self.resolution_sigma < 0:
            raise ValueError("resolution_sigma must be a finite non-negative number.")

        if self.acceptance_curve is not None:
            self.acceptance_curve.x_name = "Acceptance Energy"
            self.acceptance_curve.y_name = "Acceptance"
            self.acceptance_curve.check_x_non_negative()
            self.acceptance_curve.check_y_non_negative()
            self.acceptance_curve.check_y_less_equal_one()

            object.__setattr__(self, "_acc_ene", np.asarray(self.acceptance_curve.x))
            object.__setattr__(self, "_acc_val", np.asarray(self.acceptance_curve.y))

            acceptance_interp = interp1d(self._acc_ene,
                                         self._acc_val,
                                         bounds_error=False,
                                         fill_value=(0.0, self._acc_val[-1]))

            object.__setattr__(self, "_acceptance_interp", acceptance_interp)

        else:
            object.__setattr__(self, "_acceptance_interp", None)

    def apply_detector_response(self,
                                energy: NDArray,
                                spectrum: NDArray) -> NDArray:
        """
        Apply the detector response to a recoil spectrum.

        Parameters
        ----------
        energy : NDArray
            Recoil-energy grid in eV.
        spectrum : NDArray
            True recoil spectrum evaluated on
            ``energy``.

        Returns
        -------
        NDArray
            Observed recoil spectrum after detector
            response effects have been applied.

        Notes
        -----
        Detector response is applied in the order
        1. energy resolution;
        2. acceptance.

        The input arrays are not modified.
        """
        observed_spectrum = spectrum.copy()
        observed_spectrum = self._apply_resolution(energy=energy,
                                                   spectrum=observed_spectrum)
        observed_spectrum = self._apply_acceptance(energy=energy,
                                                   spectrum=observed_spectrum)
        return observed_spectrum

    def _apply_resolution(self,
                          energy: NDArray,
                          spectrum: NDArray) -> NDArray:
        """
        Apply Gaussian energy smearing to a recoil spectrum.

        Parameters
        ----------
        energy : NDArray
            Recoil-energy grid in eV.
        spectrum : NDArray
            True recoil spectrum evaluated on
            ``energy``.

        Returns
        -------
        NDArray
            Smeared recoil spectrum evaluated on the
            same energy grid.

        Notes
        -----
        The smearing is implemented through a Gaussian
        response matrix

        .. math::

        G(E, E')

        which maps true recoil energies to reconstructed
        recoil energies.

        The response matrix is normalized numerically so
        that event counts are conserved on the supplied
        energy grid.

        If ``resolution_sigma == 0``, the input spectrum
        is returned unchanged.
        """
        if not self.has_resolution:
            return spectrum

        deltaE = energy[:, None] - energy[None, :]

        kernel = np.exp( -0.5 * (deltaE / self.resolution_sigma)**2 )

        norm = np.trapezoid(kernel, energy, axis=0)
        if np.any(norm == 0):
            raise RuntimeError("Resolution kernel normalization failed.")

        kernel /= norm

        smeared = kernel @ (spectrum * np.gradient(energy))
        return smeared

    def _apply_acceptance(self,
                          energy: NDArray,
                          spectrum: NDArray) -> NDArray:
        """
        Apply detector acceptance to a recoil spectrum.

        Parameters
        ----------
        energy : NDArray
            Recoil-energy grid in eV.
        spectrum : NDArray
            Recoil spectrum evaluated on ``energy``.

        Returns
        -------
        NDArray
            Acceptance-corrected spectrum.

        Notes
        -----
        Acceptance values are obtained by interpolation
        of ``acceptance_curve``.

        For energies below the first tabulated point,
        the acceptance is assumed to be zero.

        For energies above the last tabulated point,
        the acceptance is assumed to remain constant at
        the last tabulated value.

        If no acceptance curve is provided, the input
        spectrum is returned unchanged.
        """
        if not self.has_acceptance:
            return spectrum
        
        eff = self._acceptance_interp(energy)
        return spectrum * eff

    @property
    def has_resolution(self) -> bool:
        """
        Whether finite energy resolution is enabled.
        """
        return self.resolution_sigma > 0

    @property
    def has_acceptance(self) -> bool:
        """
        Whether an acceptance curve is associated
        with the detector.
        """
        return self.acceptance_curve is not None
