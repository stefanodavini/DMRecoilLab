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

from src.detectors.detectors import IdealDetector
from src.nuclei.isotopes import Xe131, Ar40

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