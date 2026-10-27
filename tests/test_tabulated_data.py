import numpy as np
import pytest
import warnings

from src.utils.tabulated_data import TabulatedData

def test_tabulated_data_raises_empty_arrays():
    x = []
    y = []
    with pytest.raises(ValueError, match="At least"):
        TabulatedData(x=x, y=y)

def test_tabulated_data_raises_one_point():
    x = [0]
    y = [1]
    with pytest.raises(ValueError, match="At least"):
        TabulatedData(x=x, y=y)

def test_tabulated_data_raises_different_lengths():
    x = [0, 100, 200, 300]
    y = [1, 2, 3]
    with pytest.raises(ValueError, match="length"):
        TabulatedData(x=x, y=y)

def test_tabulated_data_raises_one_dimensional():
    x = np.asarray([[0., 100, 200, 300], [1, 2, 3, 4]])
    y = [1, 2, 3, 4]
    with pytest.raises(ValueError, match="one-dimensional"):
        TabulatedData(x=x, y=y)
    x = [0., 100, 200, 300]
    y = np.asarray([[1, 2, 3, 4], [0, 100, 200, 300]])
    with pytest.raises(ValueError, match="one-dimensional"):
        TabulatedData(x=x, y=y)

def test_tabulated_data_raises_nan():
    x = [0, 100, np.nan]
    y = [1, 2, 3]
    with pytest.raises(ValueError, match="finite"):
        TabulatedData(x=x, y=y)
    x = [0, 100, 200]
    y = [1, 2, np.nan]
    with pytest.raises(ValueError, match="finite"):
        TabulatedData(x=x, y=y)

def test_tabulated_data_raises_x_not_increasing():
    x = [0, 100, 100]
    y = [1, 2, 3]
    with pytest.raises(ValueError, match="increasing"):
        TabulatedData(x=x, y=y)

def test_tabulated_data_file_round_trip():
    output_file = "tests/output/tabulated_data_roundtrip.txt"
    x = [0, 1, 2]
    y = [0, 100, 200]
    output_data = TabulatedData(x=x, y=y, x_name="x", y_name="f(x)")
    output_data.to_file(filename=output_file)
    input_data = TabulatedData.from_file(output_file)

    assert np.allclose(output_data.x, input_data.x, atol=1e-12)
    assert np.allclose(output_data.y, input_data.y, atol=1e-12)

def test_tabulated_data_from_file_raises_empty_file():
    err_file = "tests/input/tabulated_data/invalid/empty_file.txt"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match="two columns"):
            TabulatedData.from_file(filename=err_file)

def test_tabulated_data_from_file_raises_one_column():
    err_file = "tests/input/tabulated_data/invalid/one_column.txt"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match="two columns"):
            TabulatedData.from_file(filename=err_file)

def test_tabulated_data_from_file_raises_three_columns():
    file = "tests/input/tabulated_data/valid/valid_data.txt"
    TabulatedData.from_file(filename=file)

