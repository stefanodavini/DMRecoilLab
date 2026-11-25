from src.kinematics.base_kinematics import NuclearRecoilKinematics
from src.kinematics.scattering_kinematics import (
    ScatteringKinematics, NonRelativisticKinematics
    )
from src.kinematics.relativistic_kinematics import (
    RelativisticKinematics,
    RelativisticElasticKinematics,
    RelativisticInelasticKinematics,
    kallen
    )

import numpy as np
import pytest

def test_kinematics_relativistic_inheritance():
    kin = RelativisticKinematics(mX=1e11, mT=122e9, m2=1e11)

    assert isinstance(kin, NuclearRecoilKinematics)
    assert isinstance(kin, ScatteringKinematics)
    assert not isinstance(kin, NonRelativisticKinematics)

def test_kinematics_relativistic_elastic_inheritance():
    kin = RelativisticElasticKinematics(mX=1e11, mT=122e9)

    assert isinstance(kin, NuclearRecoilKinematics)
    assert isinstance(kin, ScatteringKinematics)
    assert isinstance(kin, RelativisticKinematics)
    assert not isinstance(kin, NonRelativisticKinematics)

def test_kinematics_relativistic_inelastic_inheritance():
    kin = RelativisticInelasticKinematics(mX=1e11, mT=122e9, m2=1.1e11)

    assert isinstance(kin, NuclearRecoilKinematics)
    assert isinstance(kin, ScatteringKinematics)
    assert isinstance(kin, RelativisticKinematics)
    assert not isinstance(kin, NonRelativisticKinematics)

def test_kinematics_relativistic_unphysical_mass_raises():
    m_unphysical = [-1, 0., np.inf, np.nan]
    m_good = 1e11

    for m in m_unphysical:
        with pytest.raises(ValueError,
                           match="Unphysical dark-matter mass"):
            RelativisticElasticKinematics(mX=m, mT=m_good)

    for m in m_unphysical:
        with pytest.raises(ValueError,
                           match="Unphysical target mass"):
            RelativisticElasticKinematics(mX=m_good, mT=m)

    for m in m_unphysical:
        with pytest.raises(ValueError,
                           match="Unphysical final-state WIMP mass"):
            RelativisticKinematics(mX=m_good, mT=m_good, m2=m)


def test_kinematics_relativistic_nonscalar_m2_raises():
    m_bad = [[], [1e11], [1e11, 2e11], [[],[2e11]]]
    for m in m_bad:
        with pytest.raises(TypeError, match="scalar"):
            RelativisticInelasticKinematics(mX=10, mT=30, m2=m)

@pytest.fixture
def kin():
    return RelativisticInelasticKinematics(mX=10., mT=30., m2=12.)


SPEED_APIS = [
    'E1_from_v', 'speed_mask', 'ER_bounds', 'ER_min', 'ER_max',
    'q_from_v_theta', 'speed_theta_mask',
]

def call_speed_api(
    kin: RelativisticKinematics,
    name: str,
    v):
    return getattr(kin, name)(v, 0.) if 'theta' in name else getattr(kin, name)(v)


@pytest.mark.parametrize('name', SPEED_APIS)
def test_kinematics_relativistic_speed_domain_errors(kin, name):
    bad = [-1., np.nan, np.inf, -np.inf, 1., 1.1, [.2, -1.], [[.2], [np.nan]]]
    for v in bad:
       with pytest.raises(ValueError):
            call_speed_api(kin, name, v=v)

@pytest.mark.parametrize('name', ['q_from_ER', 'Q2_from_ER', 'vmin_from_ER'])
def test_kinematics_relativistic_recoil_domain_errors(kin, name):
    bad = [-1., np.nan, np.inf, -np.inf, [0., -np.inf], [[1.], [np.nan]]]
    for ER in bad:
        with pytest.raises(ValueError):
            getattr(kin, name)(ER)

@pytest.mark.parametrize('name', ['s_from_E1', 'energy_mask', 'ER_bounds_from_E1'])
def test_kinematics_relativistic_total_energy_domain_errors(kin, name):
    bad = [9., -1., np.nan, np.inf, -np.inf, [10., 9.]]
    for E in bad:
        with pytest.raises(ValueError):
            getattr(kin, name)(E)

@pytest.mark.parametrize("m1,M,m2,E", [
    (10., 120e3, 247., 248.),
    (10.,   30.,  10.,  20.),
    (10.,   30.,   4.,  15.),
    (100.,   2.,  110.,900.),
])
def test_kinematics_relativistic_ER_bounds(m1, M, m2, E):
    """
    Compare ER_bounds_from_E1 against the textbook
    two-body center-of-mass endpoint formula.

    This verifies the laboratory endpoint calculation
    independently of the implementation details used
    inside RelativisticKinematics.
    """
    k = RelativisticKinematics(mX=m1, mT=M, m2=m2)
    s = m1*m1 + M*M + 2*M*E
    energies = [(s + m*m - M*M)/(2*np.sqrt(s)) for m in (m1, m2)]
    momenta = [np.sqrt(e*e-m*m) for e,m in zip(energies, (m1,m2))]
    center = (2*energies[0]*energies[1]-m1*m1-m2*m2)/(2*M)
    spread = momenta[0]*momenta[1]/M
    assert np.allclose(k.ER_bounds_from_E1(E1=E),
                       (center-spread, center+spread),
                       rtol=1e-8, atol=1e-10)

@pytest.mark.parametrize("m2", [4., 10., 12.])
def test_kinematics_relativistic_endpoint_inverse_and_four_momentum(m2):
    """
    Verify that recoil endpoints invert through vmin_from_ER
    and satisfy exact four-momentum conservation.

    Both endpoint branches are checked for elastic,
    exothermic, and endothermic scattering.
    """
    k = RelativisticKinematics(mX=10., mT=30., m2=m2)
    v = np.array([max(k.threshold_speed + .05, .3), .9, .99])
    E = k.E1_from_v(v=v)
    p = E*v
    for r in k.ER_bounds(v=v):
        expected_speed = np.zeros_like(v) if np.all(r == 0) else v
        assert np.allclose(k.vmin_from_ER(ER=r), expected_speed,
                           rtol=2e-10, atol=1e-12)
        if np.all(r == 0):
            continue
        q = k.q_from_ER(ER=r)
        cos = ((E+k.mT)*r + (m2*m2-k.m1*k.m1)/2)/(p*q)
        assert np.allclose(np.abs(cos), 1., rtol=1e-10)
        final_p2 = p*p + q*q - 2*p*q*cos
        assert np.allclose((E-r)**2-final_p2, m2*m2, rtol=1e-10)
        assert np.all(E-r >= m2)

def test_kinematics_relativistic_elastic_and_nonrelativistic_limit():
    """
    Verify agreement with exact elastic relativistic
    kinematics and recovery of the standard
    non-relativistic recoil expressions at small speeds.
    """
    k = RelativisticElasticKinematics(mX=50e9, mT=120e9)
    v = np.array([0., 1e-8, 1e-5, 1e-3, .8])
    E = k.E1_from_v(v=v)
    expected = 2*k.mT*(E*v)**2 / (k.m1**2+k.mT**2+2*k.mT*E)
    assert np.allclose(k.ER_max(v=v), expected, rtol=2e-14)
    assert np.allclose(k.ER_min(v=v), 0.)
    assert np.allclose(k.ER_max(v=v[1:4]), 2*k.mu**2*v[1:4]**2/k.mT, rtol=2e-6)
    r = k.ER_max(v=v[1:])
    assert np.allclose(k.vmin_from_ER(ER=r), v[1:], rtol=2e-14, atol=1e-10)

def test_kinematics_relativistic_inelastic_nonrelativistic_limit():
    """
    Verify that exact relativistic inelastic formulas
    reduce to the standard non-relativistic expressions
    for small mass splittings and small incident speeds.
    """
    for delta in [-1e4, 1e4]:
        k = RelativisticInelasticKinematics(mX=50e9, mT=120e9, m2=50e9+delta)
        v = np.array([.001, .002])
        root = np.sqrt((k.mu*v)**2-2*k.mu*delta)
        expected = ((k.mu*v-root)**2/(2*k.mT),
                    (k.mu*v+root)**2/(2*k.mT))
        assert np.allclose(k.ER_bounds(v=v), expected, rtol=1e-5)
        r = np.array([1e3, 1e4, 1e5])
        nr = np.abs(k.mT*r/k.mu+delta)/np.sqrt(2*k.mT*r)
        assert np.allclose(k.vmin_from_ER(ER=r), nr, rtol=1e-5, atol=1e-4)

def test_kinematics_relativistic_endothermic_threshold():
    """
    Verify exact threshold behavior.

    At threshold the recoil interval collapses to a
    single value, threshold_speed is accepted,
    and the immediately smaller representable speed
    remains forbidden.
    """
    k = RelativisticInelasticKinematics(mX=10., mT=120e3, m2=247.)
    expected = (k.m2**2-k.m1**2)/(2*(k.mT+k.m2))
    assert np.allclose(k.ER_bounds(v=k.threshold_speed),
                       [expected, expected], rtol=2e-6)
    assert np.allclose(k.ER_bounds_from_E1(E1=k.threshold_energy),
                       [expected, expected], rtol=2e-6)
    below = np.nextafter(k.threshold_speed, 0)
    assert not k.speed_mask(v=below)
    assert np.all(np.isnan(k.ER_bounds(v=below)))
    assert np.allclose(k.vmin_from_ER(ER=expected),
                       k.threshold_speed, rtol=1e-14, atol=1e-9)

def test_kinematics_relativistic_exothermic_at_rest():
    """
    Verify the special case of exothermic scattering
    from a WIMP initially at rest.

    Mass splitting alone provides the available kinetic
    energy and yields a unique recoil solution.
    """
    k = RelativisticInelasticKinematics(mX=10., mT=30., m2=4.)
    recoil = (k.m1**2-k.m2**2)/(2*(k.m1+k.mT))
    assert np.allclose(k.ER_bounds(v=0.), [recoil, recoil], rtol=1e-14)
    assert np.allclose(k.vmin_from_ER(ER=recoil), 0., atol=1e-15)
    lo, hi = k.q_from_v_theta(v=0., theta=np.array([0., .5, np.pi]))
    assert np.all(np.isnan(lo))
    assert np.allclose(hi, k.q_from_ER(ER=recoil))

@pytest.mark.parametrize("m2", [4., 10., 247.])
def test_kinematics_relativistic_angular_roots_conserve_four_momentum(m2):
    """
    Verify that every finite angular-transfer root
    satisfies exact relativistic four-momentum
    conservation and remains within the recoil
    energy interval allowed by ER_bounds.
    """
    k = RelativisticKinematics(mX=10., mT=30., m2=m2)
    v = max(.9, (1+k.threshold_speed)/2)
    theta = np.linspace(0, np.pi, 501)
    E = k.E1_from_v(v=v)
    p = E*v
    roots = k.q_from_v_theta(v=v, theta=theta)
    assert np.any(np.isfinite(roots))
    for q in roots:
        mask = np.isfinite(q)
        r = q[mask]**2 / (np.hypot(k.mT, q[mask])+k.mT)
        final_p2 = p*p+q[mask]**2-2*p*q[mask]*np.cos(theta[mask])
        assert np.allclose((E-r)**2-final_p2, m2*m2, rtol=1e-9, atol=1e-10)
        assert np.all(q[mask] >= 0)
        lo,hi = k.ER_bounds(v=v)
        assert np.all(r >= lo-1e-10)
        assert np.all(r <= hi+1e-10)

def test_kinematics_relativistic_angular_nonrelativistic_and_forbidden_angles():
    """
    Distinguish domain violations from forbidden physics.

    Invalid inputs raise exceptions, whereas valid-domain
    configurations below threshold are represented through
    False masks or NaN kinematic endpoints.
    """
    k = RelativisticElasticKinematics(mX=50e9, mT=120e9)
    theta = np.array([0., .3, 1.])
    lo, hi = k.q_from_v_theta(v=.001, theta=theta)
    assert np.allclose(lo, 0.)
    assert np.allclose(hi, 2*k.mu*.001*np.cos(theta), rtol=1e-6)
    lo, hi = k.q_from_v_theta(v=0.5, theta=np.pi)
    assert hi == 0. and np.isnan(lo)
    k = RelativisticInelasticKinematics(mX=10., mT=30., m2=12.)
    assert not k.speed_theta_mask(v=0.9, theta=np.pi)


def test_kinematics_relativistic_speed_validator_extends_base():
    """
    Verify that relativistic speed validation inherits
    the base non-negative-domain checks and additionally
    enforces v < 1.
    """
    assert np.allclose(NuclearRecoilKinematics._validate_domain_speed(v=[0., 2.]),
                       [0., 2.])
    with pytest.raises(ValueError, match='v < 1'):
        RelativisticKinematics._validate_domain_speed(v=[0., 2.])
    assert np.allclose(RelativisticKinematics._validate_domain_speed(v=[0., .9]),
                       [0., .9])

def test_kinematics_relativistic_forbidden_configurations_are_masked(kin):
    threshold = kin.threshold_speed
    v = [0., np.nextafter(threshold, 0.), threshold, .9]
    assert np.allclose(kin.speed_mask(v=v), [False, False, True, True])
    for bound in kin.ER_bounds(v=v):
        assert np.all(np.isnan(bound[:2]))
        assert np.all(np.isfinite(bound[2:]))
    E = [kin.m1, np.nextafter(kin.threshold_energy, 0.), kin.threshold_energy]
    assert np.allclose(kin.energy_mask(E1=E), [False, False, True])
    for bound in kin.ER_bounds_from_E1(E1=E):
        assert np.all(np.isnan(bound[:2]))
        assert np.isfinite(bound[2])
    # Theta=pi is in the domain, but forbids endothermic target recoil.
    assert not kin.speed_theta_mask(v=.9, theta=np.pi)
    assert np.all(np.isnan(kin.q_from_v_theta(.9, np.pi)))
    # s is well-defined below the final-state threshold.
    assert np.isfinite(kin.s_from_E1(E1=kin.m1))

def test_kinematics_relativistic_scalar_broadcast_and_empty_array_support(kin):
    """
    Verify NumPy-style shape preservation and broadcasting.

    Scalar inputs remain scalar, multidimensional arrays
    broadcast correctly, and empty arrays are accepted
    without disabling domain validation.
    """
    assert kin.ER_max(v=.9).shape == ()
    assert kin.q_from_ER(ER=0.).shape == ()
    assert kin.ER_max(v=[[.9, .99]]).shape == (1, 2)
    roots = kin.q_from_v_theta(v=np.array([.9,.99])[:,None], theta=[0.,.5,1.])
    assert roots[0].shape == roots[1].shape == (2, 3)
    assert kin.ER_max(v=[]).shape == (0,)
    assert kin.q_from_ER(ER=[]).shape == (0,)
    # Validate original operands even if broadcasting would discard all
    # entries (e.g. an empty speed array and an invalid scalar angle).
    with pytest.raises(ValueError):
        kin.q_from_v_theta(v=[], theta=-1.)
    with pytest.raises(ValueError):
        kin.q_from_v_theta(v=-1., theta=[])

def test_kinematics_relativistic_zero_recoil_and_transfer_definitions(kin):
    """
    Verify the special ER=0 conventions and the
    relationship between spatial momentum transfer q
    and invariant transfer Q2.
    """
    elastic = RelativisticElasticKinematics(mX=10., mT=30.)
    assert elastic.vmin_from_ER(ER=0.) == 0.
    assert np.isinf(kin.vmin_from_ER(ER=0.))
    assert np.isinf(RelativisticKinematics(mX=10., mT=30., m2=4.).vmin_from_ER(ER=0.))
    assert kin.q_from_ER(ER=0.) == kin.Q2_from_ER(ER=0.) == 0.
    recoil = np.array([0., 1., 100.])
    assert np.allclose(kin.q_from_ER(ER=recoil)**2,
                       kin.Q2_from_ER(ER=recoil)+recoil**2)

def test_kinematics_relativistic_random_physical_recoils():
    """
    Property-style test over random physical configurations.

    Random masses, energies, recoils, and scattering angles
    are generated. Reconstructed angular roots must recover
    the original momentum transfer and satisfy kinematic
    constraints.
    """
    rng = np.random.default_rng(42)
    for _ in range(1000):
        m1,M,m2 = 10**rng.uniform(0,2,3)
        kin = RelativisticKinematics(mX=m1, mT=M, m2=m2)
        energy = kin.threshold_energy*rng.uniform(1.01,3.)
        v = np.sqrt((energy-m1)/energy)*np.sqrt((energy+m1)/energy)
        E = kin.E1_from_v(v=v)
        lo,hi = kin.ER_bounds(v=v)
        recoil = lo+(hi-lo)*rng.uniform(.01,.99)
        q = kin.q_from_ER(ER=recoil)
        costheta = ((E+M)*recoil+(m2-m1)*(m2+m1)/2)/(E*v*q)
        assert abs(costheta) <= 1+1e-12
        roots = kin.q_from_v_theta(v=v, theta=np.arccos(np.clip(costheta,-1,1)))
        assert np.any(np.isfinite(roots))
        assert np.nanmin(abs(np.asarray(roots)-q))/q < 1e-7

def test_kinematics_relativistic_kallen():
    """
    Verify symmetry, broadcasting, threshold behavior,
    and scalar-array consistency of the Källén function.
    """
    x = np.array([1,4,9])[:,None]
    y = np.array([1,2])
    z = 3
    expected = x*x + y*y + z*z - 2*x*y - 2*x*z - 2*y*z
    assert np.allclose(kallen(x,y,z), expected)
    assert np.allclose(kallen(z,x,y), expected)
    assert np.allclose(kallen(y,z,x), expected)
    assert kallen(x,y,z).shape == (3,2)
    assert kallen(1,2,3).shape == ()
    assert kallen(1,1,1) == -3.
    assert kallen(x=9, y=4, z=1) == 0.
    assert kallen(x=10**10, y=0, z=0) == 1e20
