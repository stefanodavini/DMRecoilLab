from src.interactions.particles import WIMP
from src.nuclei.isotopes import Ar40, Xe131
from src.nuclei.form_factors import TabulatedFormFactor, HelmFormFactor
from src.utils.factory import create_kinematics
from src.utils.factory import create_interaction_si

from src.kinematics.scattering_kinematics import ElasticKinematics, InelasticKinematics

def test_factory_create_kinematics_elastic():
    wimp = WIMP(mass=1e11, spin=0)

    kin = create_kinematics(wimp=wimp, nucleus=Ar40,
                            delta=0)

    assert isinstance(kin, ElasticKinematics)

def test_factory_create_kinematics_inelastic():
    delta = 100e3
    wimp = WIMP(mass=1e11, spin=0)

    kin = create_kinematics(wimp=wimp, nucleus=Ar40,
                            delta=delta)

    assert isinstance(kin, InelasticKinematics)
    assert kin.delta == delta

def test_factory_create_interaction_si_default_form_factor():
    si = create_interaction_si(wimp=WIMP(mass=1e11, spin=0),
                               nucleus=Xe131)

    assert isinstance(si.nuclear_response, HelmFormFactor)

def test_factory_create_interaction_si_custom_form_factor():
    ff = TabulatedFormFactor(q=[0, 1e3, 1e6],
                             F=[1, 1e-1, 1e-2])

    si = create_interaction_si(wimp=WIMP(mass=1e11, spin=0),
                               nucleus=Xe131,
                               form_factor=ff)

    assert si.nuclear_response is ff

def test_factory_create_interaction_si_wimp_nucleus():
    wimp = WIMP(mass=1e11, spin=0)
    si = create_interaction_si(wimp=wimp,
                               nucleus=Ar40)

    assert si.wimp is wimp
    assert si.nucleus == Ar40

def test_factory_create_interaction_si_sigma_reference():
    sigma = 1e-42

    si = create_interaction_si(wimp=WIMP(mass=1e11, spin=0),
                               nucleus=Ar40,
                               sigma_p=sigma)

    assert si.sigma_reference == sigma