import numpy as np
import pytest

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

