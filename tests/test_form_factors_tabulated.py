import numpy as np
import pytest

from src.nuclei.form_factors import TabulatedFormFactor, HelmFormFactor
from src.nuclei.isotopes import Xe131

def test_form_factor_tabulated_file_round_trip():
    output_file = "tests/output/form_factor_roundtrip.txt"
    q_max = 1e8
    helm = HelmFormFactor(Xe131)
    helm.to_file(output_file, q_max=q_max, n_points=1000)
    ff = TabulatedFormFactor.from_file(output_file)
    q = np.linspace(0, q_max, 1000)

    assert np.allclose(helm.F(q=q), ff.F(q=q), atol=1e-4)

def test_form_factor_tabulated_negative_q_raises():
    q = [-1, 0, 1]
    F = [1, 0.5, 0]
    with pytest.raises(ValueError, match="non-negative"):
        TabulatedFormFactor(q=q, F=F)

def test_form_factor_tabulated_abs_F_gt_1_raises():
    q = [0, 1, 2]
    F = [1.1, 0.5, 0]
    with pytest.raises(ValueError, match="<= 1"):
        TabulatedFormFactor(q=q, F=F)
    q = [0, 1, 2]
    F = [-1.1, 0.5, 0]
    with pytest.raises(ValueError, match="<= 1"):
        TabulatedFormFactor(q=q, F=F)

def test_form_factor_tabulated_check_normalization():
    # Raise error by default if F(q=0) != 1
    q = [0, 1, 2]
    F = [0.9, 0.5, 0]
    with pytest.raises(ValueError, match="must equal 1"):
        TabulatedFormFactor(q=q, F=F)
    # The input check_normalization = False
    # bypass the check
    TabulatedFormFactor(q=q, F=F, check_normalization=False)

def test_form_factor_tabulated_interpolation():
    q = [0., 1.]
    F = [1., 0.]

    ff = TabulatedFormFactor(q=q, F=F)
    assert np.all(ff.F(q) == F)
    assert np.isclose(ff.F(0.5), 0.5)
    assert ff.F(2.) == 0.
    assert ff.F(-1.) == 0.
