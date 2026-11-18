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

Some benchmark detectors use IdealDetector with a hard threshold,
while others use ResponseDetector with finite energy resolution
and energy-dependent acceptance.

Acceptance tables are implemented in the private _reference_acceptance module
and re-exported here for existing notebook imports.

References
----------
See the publication associated with each
configuration for the exact detector
definition and event selection.
"""

from src.detectors.detectors import IdealDetector, ResponseDetector
from src.detectors._reference_acceptance import (
    _deap_acceptance,
    _ds20k_acceptance,
    _ds50_acceptance,
    _lz_acceptance,
    _xenont_acceptance,
)
from src.nuclei.isotopes import Ar40, Xe131


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

XENONnT_2024_RESPONSE = ResponseDetector(
    nucleus=Xe131,
    mass_kg=5900,
    exposure_days=278,
    acceptance_curve=_xenont_acceptance(),
    resolution_sigma=3e3,
)
# XENONnT_2024_RESPONSE
# XENON Collaboration, XENONnT Analysis: Signal Reconstruction, Calibration
# and Event Selection, arXiv:2409.08778v1, Fig. 20 (right).
# DOI: https://doi.org/10.48550/arXiv.2409.08778
# The plotted acceptance is a rough 1D projection of the published 2D S1/S2
# acceptance; it is not a direct numerical reproduction of the figure.
# NR energy mapping assumes 6 photons/keVnr, motivated by arXiv:2412.10451v2,
# Fig. 7, and the SR0 gain of 0.151 PE/photon from arXiv:2409.08778v1.
# Since the acceptance is a 2D projection, the resolution_sigma is a simple
# benchmark value; the paper does not provide a single Gaussian NR-resolution
# parameter.

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
    )
# LZ_2026_RESPONSE
# LZ Collaboration, Search for dark matter particle interactions in an extended
# nuclear recoil energy window with the LZ experiment.
# Fig. S2 WS ROI efficiency, digitized from arXiv:2609.02823v1.
# DOI: https://doi.org/10.48550/arXiv.2609.02823
# Resolution is estimated at the 248 keVnr event. The HE SB segment is included
# in the same acceptance table with the requested unit efficiency.

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

DS50_2018_RESPONSE = ResponseDetector(
    nucleus=Ar40,
    mass_kg=31.3,
    exposure_days=532.4,
    acceptance_curve=_ds50_acceptance(),
    resolution_sigma=6e3,
)
# DS50_2018_RESPONSE
# DarkSide Collaboration, DarkSide-50 532-day Dark Matter Search with
# Low-Radioactivity Argon, PRD 98, 102006 (2018), arXiv:1802.07198.
# DOI: https://doi.org/10.1103/PhysRevD.98.102006
# The acceptance is a rough digitization of Fig. 10. The paper does not
# provide a single Gaussian NR-resolution parameter; 6 keV is a benchmark.

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

DEAP3600_2026_RESPONSE = ResponseDetector(
    resolution_sigma=10e3,
)
# DEAP3600_2026_RESPONSE
# DEAP Collaboration, Dark Matter Search with the DEAP-3600 Detector
# using the Profile Likelihood Ratio Method.
# arXiv:2603.13965v1. DOI: https://doi.org/10.48550/arXiv.2603.13965
# The acceptance curve is a rough PE-to-energy digitization of Fig. 8.
# The NR-energy mapping assumes a constant argon quenching factor Q = 0.25;
# see _deap_acceptance for the reference and approximation limits.
# resolution_sigma remains a simple benchmark value; the paper does not
# provide a single Gaussian NR-resolution parameter.

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

DS20K_10YEAR_RESPONSE = ResponseDetector(
    nucleus=Ar40,
    mass_kg=20000,
    exposure_days=3650,
    acceptance_curve=_ds20k_acceptance(),
    resolution_sigma=3e3,
)
# DS20K_10YEAR_RESPONSE
# DarkSide Collaboration, DarkSide-20k design report, Fig. 92
# (older manuscript version: Fig. 97); related publication arXiv:1707.08145.
# DOI: https://doi.org/10.1140/epjp/i2018-11973-4
# Acceptance is a rough digitization of the f200 curve; 3 keV is used as a
# simple response-resolution benchmark.

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
