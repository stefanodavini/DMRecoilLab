import numpy as np
import pytest
import warnings

from src.halo.shm import StandardHaloModel
from src.halo.tabulated_halo import TabulatedSpeedDistribution

def test_tabulated_velocity_distribution_round_trip():
    output_file = "tests/output/shm_roundtrip.txt"
    shm = StandardHaloModel()
    shm.to_file(output_file)
    halo = TabulatedSpeedDistribution.from_file(output_file)
    v = np.linspace(0, shm.maximum_speed, 1000)

    assert np.allclose(halo.velocity_pdf(v), shm.velocity_pdf(v), atol=1e-12)

def test_tabulated_halo_raises_empty_arrays():
    v = []
    f = []
    with pytest.raises(ValueError, match="At least"):
        TabulatedSpeedDistribution(v, f)

def test_tabulated_halo_raises_different_lengths():
    v = [0, 100, 200, 300]
    f = [1, 2, 3]
    with pytest.raises(ValueError, match="length"):
        TabulatedSpeedDistribution(v, f)

def test_tabulated_halo_raises_one_dimensional():
    v = np.asarray([[0., 100, 200, 300],[1, 2, 3, 4]])
    f = [1, 2, 3, 4]
    with pytest.raises(ValueError, match="one-dimensional"):
        TabulatedSpeedDistribution(v, f)
    v = [0., 100, 200, 300]
    f = np.asarray([[1, 2, 3, 4],[0, 100, 200, 300]])
    with pytest.raises(ValueError, match="one-dimensional"):
        TabulatedSpeedDistribution(v, f)

def test_tabulated_halo_raises_negative_v():
    v = [-1, 0, 1]
    f = [1, 2, 3]
    with pytest.raises(ValueError, match="negative"):
        TabulatedSpeedDistribution(v=v, f=f)

def test_tabulated_from_file_invalid_filename():
    with pytest.raises(ValueError, match="filename"):
        TabulatedSpeedDistribution.from_file(None)
    with pytest.raises(ValueError, match="filename"):
        TabulatedSpeedDistribution.from_file(filename="")

def test_tabulated_from_file_raises_empty_file():
    err_file = "tests/input/halo/invalid/empty_file.txt"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError):
            TabulatedSpeedDistribution.from_file(filename=err_file)

def test_tabulated_from_file_raises_not_increasing():
    err_file = "tests/input/halo/invalid/duplicate_v.txt"
    with pytest.raises(ValueError, match="increasing"):
        TabulatedSpeedDistribution.from_file(filename=err_file)
    err_file = "tests/input/halo/invalid/unsorted_v.txt"
    with pytest.raises(ValueError, match="increasing"):
        TabulatedSpeedDistribution.from_file(filename=err_file)

def test_tabulated_from_file_raises_one_point():
    err_file = "tests/input/halo/invalid/one_point.txt"
    with pytest.raises(ValueError):
        TabulatedSpeedDistribution.from_file(filename=err_file)

def test_tabulated_from_file_raises_nan():
    err_file = "tests/input/halo/invalid/nan_f.txt"
    with pytest.raises(ValueError, match="finite"):
        TabulatedSpeedDistribution.from_file(filename=err_file)
    err_file = "tests/input/halo/invalid/nan_v.txt"
    with pytest.raises(ValueError, match="finite"):
        TabulatedSpeedDistribution.from_file(filename=err_file)

def test_tabulated_from_file_raises_negative_f():
    err_file = "tests/input/halo/invalid/negative_f.txt"
    with pytest.raises(ValueError, match="negative"):
        TabulatedSpeedDistribution.from_file(filename=err_file)

def test_tabulated_from_file_raises_one_column():
    err_file = "tests/input/halo/invalid/one_column.txt"
    with pytest.raises(ValueError):
        TabulatedSpeedDistribution.from_file(filename=err_file)

def test_tabulated_from_file_raises_three_columns():
    err_file = "tests/input/halo/invalid/three_columns.txt"
    with pytest.raises(ValueError):
        TabulatedSpeedDistribution.from_file(filename=err_file)

def test_tabulated_maximum_speed():
    v = [0, 100, 200, 300, 400]
    f = [1, 1, 1, 1, 0]
    halo = TabulatedSpeedDistribution(v=v, f=f)
    assert halo.maximum_speed == 300

def test_tabulated_interpolation():
    halo = TabulatedSpeedDistribution(v=[0, 1],
                                      f=[0, 1])
    expected = 1 /(4 * np.pi)

    assert np.isclose(halo.velocity_pdf(0.5),
                      expected, rtol=1e-3)
                      
def test_tabulated_constructor_normalizes_distribution():
    v = [0, 100, 200, 300, 400]
    f = [0, 1, 2, 1, 0]
    halo = TabulatedSpeedDistribution(v=v, f=f)

    assert np.isclose(halo.normalization(v=v), 1., rtol=1e-3)

def test_tabulated_constructor_invalid_normalization():
    v = [0., 100.]
    f = [0., 1.]

    with pytest.raises(ValueError, match="Integral"):
        TabulatedSpeedDistribution(v=v, f=f, normalize=False)

def test_tabulated_already_normalized():
    v = np.linspace(0., 1., 1000)
    f = np.linspace(0., 1., 1000) / np.pi

    halo = TabulatedSpeedDistribution(v=v, f=f, normalize=False)
    assert np.isclose(halo.normalization(v=v), 1., rtol=1e-3)

def test_tabulated_zero_norm_raise():
    v = np.linspace(0., 500., 50)
    f = np.linspace(0., 0., 50)

    with pytest.raises(ValueError, match="normalization"):
        TabulatedSpeedDistribution(v=v, f=f)
