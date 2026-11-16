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
            # HE SB: unit-efficiency segment, followed by the upper boundary.
            301e3, 302e3, 700e3, 701e3,
        ],
        y=[
            0.0, 0.0, 0.001000, 0.045376, 0.240994, 0.572080,
            0.681770, 0.796122, 0.863085, 0.916027, 0.936026,
            0.944187, 0.942740, 0.955819, 0.955805, 0.955750,
            0.955799, 0.955605, 0.954773, 0.937306, 0.897118,
            0.815262, 0.670568, 0.498860, 0.312901, 0.161969,
            0.073781, 0.025910, 0.007525, 0.001609,
            0.001609, 1.0, 1.0, 0.0,
        ],
        x_name="NR recoil energy [eV]",
        y_name="Signal efficiency",
    )


def _deap_acceptance() -> TabulatedData:
    """Return a rough PE-to-energy mapping of the DEAP total acceptance.

    Figure 8 of arXiv:2603.13965 gives the total WIMP acceptance in PE.
    For this lightweight benchmark, PE is converted to reconstructed
    electron-equivalent energy using the reported 6.1 PE/keV light yield.
    This is not an NR-energy calibration because the energy-dependent NR
    quenching factor is not applied here.
    """
    return TabulatedData(
        # Rough digitization of the black total-acceptance curve in Fig. 8.
        # x is E_ee [eV] = PE / (6.1 PE/keV). Assumed up to 25000 PE since in the paper
        # they show the alpha background
        x=[0, 90e3 / 6.1, 110e3 / 6.1, 130e3 / 6.1, 150e3 / 6.1,
           175e3 / 6.1, 200e3 / 6.1, 225e3 / 6.1, 250e3 / 6.1,
           275e3 / 6.1, 300e3 / 6.1, 6100e3 / 6.1, 6200e3 / 6.1],
        # Rough visual digitization of the black curve in Fig. 8; the paper
        # does not provide these points as a numerical table.
        y=[0, 0.18, 0.22, 0.25, 0.27, 0.29, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0],
        x_name="Reconstructed energy [eVee] (rough PE mapping)",
        y_name="Total WIMP acceptance",
    )


def _ds50_acceptance() -> TabulatedData:
    """Return a rough digitization of the DS-50 overall NR acceptance.

    The points follow the black ``Overall NR Acceptance`` curve in Fig. 10
    of arXiv:1802.07198.  The paper presents the acceptance graphically as a
    function of S1, with the upper axis giving the corresponding NR-energy
    scale, so the values below are an approximate digitization rather than a
    numerical table from the publication.
    """
    return TabulatedData(
        x=[0, 20e3, 30e3, 40e3, 50e3, 60e3, 70e3, 80e3,
           100e3, 120e3, 150e3, 200e3, 1000e3, 1001e3],
        y=[0.0, 0.0, 0.02, 0.15, 0.45, 0.62, 0.68, 0.725,
           0.725, 0.725, 0.725, 0.725, 0.725, 0.0],
        x_name="NR recoil energy [eV] (rough S1 mapping)",
        y_name="Overall NR acceptance",
    )


def _ds20k_acceptance() -> TabulatedData:
    """Return a rough digitization of the DarkSide-20k f200 NR acceptance.

    The curve follows the f200 band in Fig. 92 of the extended DarkSide-20k
    design report. Older manuscript versions label the analogous plot Fig. 97.
    The report presents the curve graphically, so these points are an
    approximate benchmark rather than a numerical table.
    """
    return TabulatedData(
        x=[0, 10e3, 20e3, 25e3, 30e3, 35e3, 40e3, 45e3,
           50e3, 60e3, 80e3, 1000e3, 1001e3],
        y=[0.0, 0.0, 0.01, 0.10, 0.30, 0.55, 0.72, 0.84,
           0.90, 0.90, 0.90, 0.90, 0.0],
        x_name="NR recoil energy [eV] (rough f200 digitization)",
        y_name="NR signal acceptance",
    )


def _xenont_acceptance() -> TabulatedData:
    """Return a rough 1D projection of the XENONnT WIMP acceptance.

    Figure 20 (right) of arXiv:2409.08778 shows the WIMP-search acceptance
    in the two-dimensional uncorrected-S1/S2 space. ResponseDetector stores a
    one-dimensional energy curve, so the values below are a rough projection
    of that panel and not a direct reproduction of the 2D acceptance.
    """
    # Approximate ER conversion reported by the paper:
    # LY = cS1 / E = 5.3325 PE/keV.
    light_yield_pe_per_kev = 5.3325
    s1_pe = [
        0.0,
        1.0 * light_yield_pe_per_kev,
        2.0 * light_yield_pe_per_kev,
        3.0 * light_yield_pe_per_kev,
        4.0 * light_yield_pe_per_kev,
        5.0 * light_yield_pe_per_kev,
        6.0 * light_yield_pe_per_kev,
        8.0 * light_yield_pe_per_kev,
        10.0 * light_yield_pe_per_kev,
        13.0 * light_yield_pe_per_kev,
        50.0 * light_yield_pe_per_kev,
        1000.0 * light_yield_pe_per_kev,
        1001.0 * light_yield_pe_per_kev
    ]

    return TabulatedData(
        x=[s1 / light_yield_pe_per_kev * 1e3 for s1 in s1_pe],
        y=[0.0, 0.10, 0.35, 0.55, 0.70, 0.80, 0.86, 0.90, 0.92, 0.92, 0.92, 0.92, 0],
        x_name="Reconstructed energy [eV] (S1 / 5.3325 PE/keV)",
        y_name="WIMP signal acceptance",
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

XENONnT_2024_RESPONSE = ResponseDetector(
    nucleus=Xe131,
    mass_kg=5900,
    exposure_days=278,
    acceptance_curve=_xenont_acceptance(),
    resolution_sigma=1e3,
)
# XENONnT_2024_RESPONSE
# XENON Collaboration, XENONnT Analysis: Signal Reconstruction, Calibration
# and Event Selection, arXiv:2409.08778v1, Fig. 20 (right).
# The plotted acceptance is a rough 1D projection of the published 2D S1/S2
# acceptance; it is not a direct numerical reproduction of the figure.

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
    resolution_sigma=3e3,
)
# DS50_2018_RESPONSE
# DarkSide Collaboration, DarkSide-50 532-day Dark Matter Search with
# Low-Radioactivity Argon, PRD 98, 102006 (2018), arXiv:1802.07198.
# The acceptance is a rough digitization of Fig. 10. The paper does not
# provide a single Gaussian NR-resolution parameter; 3 keV is a benchmark.

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

DEAP_2026_RESPONSE = ResponseDetector(
    nucleus=Ar40,
    mass_kg=1266,
    exposure_days=790.8,
    acceptance_curve=_deap_acceptance(),
    resolution_sigma=3e3,
)
# DEAP_2026_RESPONSE
# DEAP Collaboration, Dark Matter Search with the DEAP-3600 Detector
# using the Profile Likelihood Ratio Method.
# arXiv:2603.13965v1.
# The acceptance curve is a rough PE-to-energy digitization of Fig. 8.
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
