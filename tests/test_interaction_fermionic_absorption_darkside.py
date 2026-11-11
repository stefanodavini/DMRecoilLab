"""Literature benchmark for NC absorption in DarkSide-50.

Dror, Elor and McGehee, arXiv:1908.10861v2:
Eq. (3.7), Fig. 4, and Table 3. The curve uses a ten-event criterion;
it is not an experimental confidence-level limit.
"""

import numpy as np

from src.detectors.detectors import IdealDetector
from src.halo.shm import StandardHaloModel
from src.interactions.fermionic_absorption import FermionicAbsorptionInteraction
from src.interactions.particles import WIMP
from src.kinematics.absorption_kinematics import AbsorptionKinematics
from src.nuclei.form_factors import HelmFormFactor
from src.nuclei.isotopes import Ar40
from src.rates import RateCalculator


def test_fermionic_absorption_dror_darkside_sigma_nc_at_40_mev():
    """Recover the approximate DarkSide-50 ordinate in Fig. 4 at 40 MeV."""
    wimp = WIMP(mass=40.0e6, spin=0.5)  # eV
    kin = AbsorptionKinematics(mX=wimp.mass, mT=Ar40.mass)

    # Table 3 fixes the product M*T = 6786 kg day.
    # Only that product matters; the split below is bookkeeping.
    detector = IdealDetector(
        nucleus=Ar40,
        mass_kg=46.0,
        exposure_days=6786.0 / 46.0,
        threshold_energy=0.6e3,  # eVnr
    )
    assert kin.ER > detector.threshold_energy

    # Arbitrary reference normalization, not the answer from the graph.
    sigma_nc_reference_cm2 = 1.0e-48
    interaction = FermionicAbsorptionInteraction(
        wimp=wimp,
        nucleus=Ar40,
        kinematics=kin,
        sigma_abs=Ar40.A**2 * sigma_nc_reference_cm2,
        nuclear_response=HelmFormFactor(Ar40),
    )
    halo = StandardHaloModel(v0=220.0, vesc=550.0).boost(vboost=240.0)
    calculator = RateCalculator(
        halo=halo,
        interaction=interaction,
        detector=detector,
        rho_dm=0.4e9,  # eV/cm^3 = 0.4 GeV/cm^3
    )

    # The paper's boundary is N=10. This method returns sigma_abs,
    # the per-nucleus coefficient, so divide by A^2 for the Fig. 4 axis.
    sigma_abs_for_ten_events = calculator.expected_cross_section_from_counts(
        counts=10.0, num=1000,
    )
    sigma_nc_for_ten_events = sigma_abs_for_ten_events / Ar40.A**2

    # Approximate, independently read graph ordinate; not an exact datum.
    # Allow graph-reading and nuclear-parameterization differences.
    sigma_nc_fig4_cm2 = 2.8e-48
    np.testing.assert_allclose(
        sigma_nc_for_ten_events,
        sigma_nc_fig4_cm2,
        rtol=0.10,
        atol=0.0,
    )
    print(f"DarkSide-50: E_abs = {kin.ER / 1e3:.6g} keVnr")
    print(f"N=10: sigma_NC = {sigma_nc_for_ten_events:.6g} cm^2")
    print(f"Fig. 4: sigma_NC approximately {sigma_nc_fig4_cm2:.2g} cm^2")

