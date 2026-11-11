"""
Reference detector configurations.

This module provides benchmark detector setups
commonly used in dark-matter direct-detection
studies.

Each detector definition is accompanied by
citation metadata indicating either:
- the publication from which the benchmark
was derived; or
- the design study used for a projected
sensitivity configuration.

The configurations are intended for:
- reproducing published results;
- comparing target materials;
- sensitivity forecasts;
- educational examples and notebooks.

Naming convention
-----------------
Configurations corresponding to published
results are named

    EXPERIMENT_YEAR

for example

    XENON1T_2018
    LZ_2024

Configurations intended for future sensitivity
studies are named

    EXPERIMENT_EXPOSURE

for example

    LZ_10YEAR
    DS20K_10YEAR

Notes
-----
The detector mass corresponds to the fiducial
(or analysis) target mass whenever available.

Threshold energies are approximate analysis
thresholds representative of the corresponding
publication and are intended primarily for
benchmark studies rather than precision
reproductions.

References
----------
See the publication associated with each
configuration for the exact detector
definition and event selection.
"""

from src.detectors.detectors import IdealDetector, ResponseDetector
from src.nuclei.isotopes import Ar40, Xe131
from src.utils.tabulated_data import TabulatedData


def _lz_acceptance() -> TabulatedData:
    """Return the digitized LZ high-energy NR efficiency from arXiv:2410.17036 and arXiv:2609.02823"""
    return TabulatedData(
        x=[
            0, 1e3, 2e3, 3e3, 4e3, 5.4e3, 6e3, 7e3, 8e3,
            10e3, 12e3, 14e3, 20e3, 50e3, 100e3, 150e3, 200e3,
            225e3, 240e3, 250e3, 255e3, 260e3, 265e3, 269.9e3,
            275e3, 280e3, 285e3, 290e3, 295e3, 300e3,
        ],
        y=[
            0.0, 0.0, 0.001000, 0.045376, 0.240994, 0.572080,
            0.681770, 0.796122, 0.863085, 0.916027, 0.936026,
            0.944187, 0.942740, 0.955819, 0.955805, 0.955750,
            0.955799, 0.955605, 0.954773, 0.937306, 0.897118,
            0.815262, 0.670568, 0.498860, 0.312901, 0.161969,
            0.073781, 0.025910, 0.007525, 0.001609,
        ],
        x_name="NR recoil energy [eV]",
        y_name="Signal efficiency",
    )

#
# Xenon
#

XENON100_2012 = IdealDetector(
    nucleus=Xe131,
    mass_kg=34,
    exposure_days=224,
    threshold_energy=6e3,
)
# XENON100_2012
# Aprile et al.
# Dark Matter Results from 225 Live Days of XENON100 Data
# PRL 109, 181301 (2012)
# arXiv:1207.5988

XENON1T_2018 = IdealDetector(
    nucleus=Xe131,
    mass_kg=1300,
    exposure_days=278.8,
    threshold_energy=5e3,
)
# XENON1T_2018
# Aprile et al.
# Dark Matter Search Results from a One Ton-Year Exposure
# PRL 121, 111302 (2018)
# arXiv:1805.12562

XENONnT_2024 = IdealDetector(
    nucleus=Xe131,
    mass_kg=5900,
    exposure_days=278,
    threshold_energy=2e3,
)
# XENONnT_2024
# XENON Collaboration
# Latest SI WIMP search results
# Benchmark based on fiducial LXe mass and representative threshold.

LZ_2024 = IdealDetector(
    nucleus=Xe131,
    mass_kg=5800,
    exposure_days=280,
    threshold_energy=2e3,
)
# LZ_2024
# LZ Collaboration
# First Dark Matter Search Results from the LZ Experiment
# PRL 131, 041002 (2023)
# arXiv:2207.03764

LZ_2026_RESPONSE = ResponseDetector(
    nucleus=Xe131,
    mass_kg=4710,
    exposure_days=220,
    acceptance_curve=_lz_acceptance(),
    resolution_sigma=1300,
    target_mass_fraction=0.21
)
# LZ_2024_RESPONSE
# LZ Collaboration, Search for dark matter particle interactions in an extended
# nuclear recoil energy window with the LZ experiment.
# Fig. S2 WS ROI efficiency, digitized from arXiv:2609.02823v1.
# Resolution is estimated at the 248 keVnr event

LZ_1000DAYS = IdealDetector(
    nucleus=Xe131,
    mass_kg=5800,
    exposure_days=1000,
    threshold_energy=2e3,
)
# LZ_1000DAYS
# Sensitivity benchmark
# Extrapolated from LZ fiducial mass.
# Not tied to a publication.

LZ_10YEAR = IdealDetector(
    nucleus=Xe131,
    mass_kg=5800,
    exposure_days=3650,
    threshold_energy=2e3,
)
# LZ_10YEAR
# Sensitivity benchmark
# Extrapolated from LZ fiducial mass.
# Not tied to a publication.

# Argon

DS50_2018 = IdealDetector(
    nucleus=Ar40,
    mass_kg=46,
    exposure_days=532,
    threshold_energy=30e3,
)
# DS50_2018
# DarkSide Collaboration
# DarkSide-50 532-day WIMP search
# PRD 98, 102006 (2018)
# arXiv:1802.07198

DEAP3600_2019 = IdealDetector(
    nucleus=Ar40,
    mass_kg=3279,
    exposure_days=231,
    threshold_energy=55e3,
)
# DEAP3600_2019
# DEAP Collaboration
# Dark Matter Search with a 231-day Exposure
# PRD 100, 022004 (2019)
# arXiv:1902.04048

DS20K_10YEAR = IdealDetector(
    nucleus=Ar40,
    mass_kg=20000,
    exposure_days=3650,
    threshold_energy=30e3,
)
# DS20K_10YEAR
# Sensitivity benchmark
# Based on DarkSide-20k design report
# Not tied to a single publication.

DEAP3600_5YEAR = IdealDetector(
    nucleus=Ar40,
    mass_kg=3279,
    exposure_days=1825,
    threshold_energy=55e3,
)
# DEAP3600_5YEAR
# Sensitivity benchmark
# Extrapolated from DEAP-3600 fiducial mass.
# Not tied to a publication.