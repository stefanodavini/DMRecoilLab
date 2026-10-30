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

def test_tabulated_data_raises_min_points():
    with pytest.raises(ValueError, match="At least"):
         TabulatedData(x=[0,1], y=[1,2], min_points=3)

def test_tabulated_data_raises_inf():
    x = [0, 100, np.inf]
    y = [1, 2, 3]
    with pytest.raises(ValueError, match="finite"):
        TabulatedData(x=x, y=y)
    x = [0, 100, 200]
    y = [1, 2, np.inf]
    with pytest.raises(ValueError, match="finite"):
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

def test_tabulated_data_raises_duplicate_x():
    x = [0, 100, 100]
    y = [1, 2, 3]
    with pytest.raises(ValueError, match="increasing"):
        TabulatedData(x=x, y=y)

def test_tabulated_data_raises_x_not_monotonic():
    x = [0, 200, 100]
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

    assert input_data.size == len(x)
    assert output_data.size == len(x)
    assert np.allclose(output_data.x, input_data.x, atol=1e-12)
    assert np.allclose(output_data.y, input_data.y, atol=1e-12)

def test_tabulated_data_to_file_invalid_filename():
    x = [0, 1, 2]
    y = [0, 100, 200]
    data = TabulatedData(x=x, y=y)
    with pytest.raises(ValueError, match="filename"):
        data.to_file(filename=None)

def test_tabulated_data_from_file_invalid_filename():
    with pytest.raises(ValueError, match="filename"):
        TabulatedData.from_file(filename=None)

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
    err_file = "tests/input/tabulated_data/invalid/three_columns.txt"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match="two columns"):
            TabulatedData.from_file(filename=err_file)

def test_tabulated_data_from_file_valid():
    file = "tests/input/tabulated_data/valid/valid_data.txt"
    data = TabulatedData.from_file(filename=file)
    assert data.size == 3

def test_tabulated_data_check_x_positive():
    x = [0, 1, 2]
    y = [0, 100, 200]
    data = TabulatedData(x=x, y=y)
    with pytest.raises(ValueError, match="positive"):
            data.check_x_positive()

def test_tabulated_data_check_x_non_negative():
    # pass with x = 0
    x = [0, 1, 2]
    y = [0, 100, 200]
    data = TabulatedData(x=x, y=y)
    data.check_x_non_negative()
    # fails for x < 0
    x = [-1, 1, 2]
    y = [0, 100, 200]
    data = TabulatedData(x=x, y=y)
    with pytest.raises(ValueError, match="non-negative"):
            data.check_x_non_negative()

def test_tabulated_data_check_y_positive():
    x = [0, 1, 2]
    y = [0, 100, 200]
    data = TabulatedData(x=x, y=y)
    with pytest.raises(ValueError, match="positive"):
            data.check_y_positive()

def test_tabulated_data_check_y_non_negative():
    # pass with y = 0
    x = [0, 1, 2]
    y = [0, 100, 200]
    data = TabulatedData(x=x, y=y)
    data.check_y_non_negative()
    # fails for y < 0
    x = [0, 1, 2]
    y = [-1, 100, 200]
    data = TabulatedData(x=x, y=y)
    with pytest.raises(ValueError, match="non-negative"):
            data.check_y_non_negative()

def test_tabulated_data_check_y_less_equal_one():
    # pass with y = 1
    x = [0, 1, 2]
    y = [1, 0.5, 0]
    data = TabulatedData(x=x, y=y)
    data.check_y_less_equal_one()
    # fails with y > 1
    x = [0, 1, 2]
    y = [1.1, 0.5, 0]
    data = TabulatedData(x=x, y=y)
    with pytest.raises(ValueError, match="<= 1"):
            data.check_y_less_equal_one()

def test_tabulated_data_check_abs_y_less_equal_one():
    # pass with |y| = 1
    x = [0, 1, 2]
    y = [1, 0, -1]
    data = TabulatedData(x=x, y=y)
    data.check_abs_y_less_equal_one()
    # fails with y < 1
    x = [0, 1, 2]
    y = [0.9, 0.5, -1.2]
    data = TabulatedData(x=x, y=y)
    with pytest.raises(ValueError, match="<= 1"):
            data.check_abs_y_less_equal_one()
