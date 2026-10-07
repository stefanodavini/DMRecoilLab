from src.nuclei.nucleus import Nucleus
from src.utils.conversions import convert_from_kg_to_eV
from src.utils.constants import DAY_TO_SECONDS

from dataclasses import dataclass

@dataclass(frozen=True)
class IdealDetector:
    """
    Ideal detector description.

    Parameters
    ----------
    nucleus
        Target isotope.

    mass_kg
        Active detector mass in kg.

    exposure_days
        Exposure time in days.

    threshold_energy
        Analysis threshold in eV.

    Notes
    -----
    This class assumes:
    - 100% detection efficiency
    - perfect energy resolution
    - isotopically pure target

    It is intended for theoretical
    rate calculations and detector
    comparisons.
    """
    nucleus: Nucleus
    mass_kg: float
    exposure_days: float = 365
    threshold_energy: float = 0.0

    @property
    def number_of_targets(self) -> float:
        """
        Number of nuclei contained in the detector.

        Returns
        -------
        float

        Notes
        -----
        The result is computed as
        N = Mdet / MT
        where
        - Mdet is detector mass (converted to eV)
        - MT is target-nucleus mass (eV)
        """
        return convert_from_kg_to_eV(self.mass_kg) / self.nucleus.mass

    @property
    def exposure_seconds(self) -> float:
        return DAY_TO_SECONDS * self.exposure_days