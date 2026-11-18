"""Acceptance tables and approximate energy mappings for reference detectors."""

import numpy as np

from src.utils.tabulated_data import TabulatedData


def _lz_acceptance(include_he_sb: bool = True) -> TabulatedData:
    """Return the LZ NR efficiency from arXiv:2410.17036 and 2609.02823.

    include_he_sb enables the benchmark unit-efficiency HE sideband from
    302 to 700 keVnr, with a cutoff at 701 keVnr. When disabled, the WS
    efficiency is retained through 300 keVnr and tapers to zero at 301 keVnr
    so ResponseDetector does not extrapolate a nonzero tail indefinitely.
    """
    x = [
        0, 1e3, 2e3, 3e3, 4e3, 5.4e3, 6e3, 7e3, 8e3,
        10e3, 12e3, 14e3, 20e3, 50e3, 100e3, 150e3, 200e3,
        225e3, 240e3, 250e3, 255e3, 260e3, 265e3, 269.9e3,
        275e3, 280e3, 285e3, 290e3, 295e3, 300e3,
    ]
    y = [
        0.0, 0.0, 0.001000, 0.045376, 0.240994, 0.572080,
        0.681770, 0.796122, 0.863085, 0.916027, 0.936026,
        0.944187, 0.942740, 0.955819, 0.955805, 0.955750,
        0.955799, 0.955605, 0.954773, 0.937306, 0.897118,
        0.815262, 0.670568, 0.498860, 0.312901, 0.161969,
        0.073781, 0.025910, 0.007525, 0.001609,
    ]

    if include_he_sb:
        x.extend([301e3, 302e3, 700e3, 701e3])
        y.extend([0.001609, 1.0, 1.0, 0.0])
    else:
        x.append(301e3)
        y.append(0.0)

    return TabulatedData(
        x=x,
        y=y,
        x_name="NR recoil energy [eV]",
        y_name="Signal efficiency",
    )


def _deap_acceptance() -> TabulatedData:
    """Return a rough PE-to-NR-energy mapping of the DEAP total acceptance.

    Figure 8 of arXiv:2603.13965 gives the total WIMP acceptance in PE.
    For this lightweight benchmark, use the reported ER light yield of
    6.1 PE/keVee and a constant argon NR quenching factor Q = E_ee/E_nr = 0.25.
    This constant approximation is used in Cai et al., "Surface backgrounds
    in the DEAP-3600 dark matter experiment", Background Requirement section:
    https://deap3600.ca/Contents/proceedings/DEAP3600LRT2010Cai.pdf
    It is not the energy-dependent SCENE-based calibration used by the
    2026 analysis, and its high-energy extrapolation is only a benchmark.
    """
    light_yield_pe_per_kev = 6.1
    nr_quenching_factor = 0.25
    # PE coordinates of the approximate low-energy acceptance curve.
    pe_values = np.array([
        0, 90, 110, 130, 150, 175, 200, 225, 250, 275, 300,
    ])
    energy_nr_ev = pe_values / (light_yield_pe_per_kev * nr_quenching_factor) * 1e3
    # Extend the plateau to 1 MeVnr and cut off at 1.001 MeVnr.
    # These energy-space boundaries are benchmark assumptions, not PE data.
    energy_nr_ev = np.concatenate((energy_nr_ev, [1000e3, 1001e3]))

    return TabulatedData(
        x=energy_nr_ev,
        # Rough visual digitization of the black curve in Fig. 8; the paper
        # does not provide these points as a numerical table.
        y=[0, 0.18, 0.22, 0.25, 0.27, 0.29, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0],
        x_name="NR recoil energy [eV] (approximate PE mapping, Q=0.25)",
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
    """Return an approximate S1-to-NR-energy mapping of XENONnT acceptance.

    Figure 20 (right) of arXiv:2409.08778 shows the WIMP-search acceptance
    in the two-dimensional uncorrected-S1/S2 space. ResponseDetector stores a
    one-dimensional energy curve, so the values below are an illustrative
    S1 projection, not a new digitization or a full S1/S2 response model.

    We assume a constant NR light yield of 6 photons/keVnr, motivated by
    Fig. 7 of the XENONnT YBe calibration, arXiv:2412.10451v2
    (https://arxiv.org/abs/2412.10451). This is a low-energy approximation,
    not a fitted constant or a calibrated extrapolation to high energies.
    The SR0 gain g1 = 0.151 PE/photon is from arXiv:2409.08778v1, Sec. VII.
    E_nr = S1 / (g1 * NR light yield) approximates uncorrected S1 as cS1,
    neglecting position dependence, fluctuations and S2 selection effects.
    The NR light yield already includes quenching; no extra factor is used.
    """
    nr_light_yield_ph_per_kev = 6.0
    g1_pe_per_photon = 0.151
    # Illustrative low-energy S1 coordinates [PE].
    s1_pe = np.array([
        0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 13.0,
        50.0,
    ])
    energy_nr_ev = s1_pe / (g1_pe_per_photon * nr_light_yield_ph_per_kev) * 1e3
    # Define the benchmark plateau/cutoff in NR energy, not S1 coordinates.
    energy_nr_ev = np.concatenate((energy_nr_ev, [1000e3, 1001e3]))

    return TabulatedData(
        x=energy_nr_ev,
        y=[0.0, 0.10, 0.35, 0.55, 0.70, 0.80, 0.86, 0.90, 0.92, 0.92, 0.92, 0.92, 0],
        x_name="NR recoil energy [eV] (approximate S1 mapping)",
        y_name="WIMP signal acceptance",
    )
