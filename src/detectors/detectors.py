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

Detector effects are applied through the
``apply_detector_effects`` interface, allowing
RateCalculator to remain independent of the
specific detector implementation.

All detector responses are assumed to act on recoil-energy
spectra expressed as dR/dER.
"""

from numpy.typing import NDArray
import numpy as np

from src.nuclei.nucleus import Nucleus
from src.utils.conversions import convert_from_kg_to_eV
from src.utils.constants import DAY_TO_SECONDS

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
        Active detector mass in kg.

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
            raise ValueError("mass_kg must be non negative.")
        if not np.isfinite(self.target_mass_fraction) or not 0 <= self.target_mass_fraction <= 1:
            raise ValueError("target_mass_fraction must be finite and in the interval [0, 1].")
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
    def apply_detector_effects(self,
                               energy: NDArray,
                               spectrum: NDArray) -> NDArray:
        """
        Apply detector response effects to a recoil spectrum.

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

        Examples of detector effects include:
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

    def apply_detector_effects(self,
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
