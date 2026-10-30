"""
Utilities for validated tabulated data.

This module provides the TabulatedData class, a reusable
container for two-column tabulated functions.

It centralizes
- validation of tabulated arrays,
- loading data from text files,
- writing data to text files.

Physics-specific constraints are intentionally left to
higher-level classes using TabulatedData.
"""

from dataclasses import dataclass
from typing import Self

import numpy as np
from numpy.typing import ArrayLike

@dataclass
class TabulatedData:
    """
    Container for a tabulated function y(x).

    Parameters
    ----------
    x : ArrayLike
        Monotonically increasing independent variable.
    y : ArrayLike
        Tabulated dependent-variable values.
    x_name : str, optional
        Name of the independent variable (default "x").
    y_name : str, optional
        Name of the dependent variable  (default "y").
    min_points : int, optional
        Minimum number of tabulation points required (default 2).
    """
    x: ArrayLike
    y: ArrayLike
    x_name: str = "x"
    y_name: str = "y"
    min_points: int = 2


    def __post_init__(self):
        """
        Convert inputs to NDArrays and validate the tabulation.
        """
        self.x = np.asarray(self.x, dtype=float)
        self.y = np.asarray(self.y, dtype=float)

        # generic validation
        if self.x.ndim != 1:
            raise ValueError(f"{self.x_name} must be a one-dimensional array.")

        if self.y.ndim != 1:
            raise ValueError(f"{self.y_name} must be a one-dimensional array.")

        if len(self.x) != len(self.y):
            raise ValueError(
                f"{self.x_name} and {self.y_name} must have the same length.")

        if len(self.x) < self.min_points:
            raise ValueError(
                f"At least {self.min_points} tabulation points are required.")

        if np.any(~np.isfinite(self.x)):
            raise ValueError(f"{self.x_name} must contain only finite values.")

        if np.any(~np.isfinite(self.y)):
            raise ValueError(f"{self.y_name} must contain only finite values.")

        if np.any(np.diff(self.x) <= 0):
            raise ValueError(f"{self.x_name} must be strictly increasing.")

    @classmethod
    def from_file(cls, filename: str) -> Self:
        """
        Construct a tabulated distribution from a text file.

        The file must contain exactly two columns.

        Parameters
        ----------
        filename: str
            Input filename.

        Returns
        -------
        TabulatedData
        """
        if not filename:
            raise ValueError("Invalid filename")

        data = np.loadtxt(filename)

        if data.ndim != 2 or data.shape[1] != 2:
            raise ValueError("Input file must contain two columns.")

        return cls(x=data[:, 0], y=data[:, 1])

    def to_file(self, filename: str):
        """
        Save the tabulated data to a two-column text file.

        The first column contains x values and the second
        column contains y values.
        
        Parameters
        ----------
        filename: str
            Output filename.
        """
        if not filename:
            raise ValueError("Invalid filename")
        
        data = np.column_stack([self.x, self.y])
        
        np.savetxt(filename, data, header=f"{self.x_name} {self.y_name}")

    def check_x_positive(self):
        """
        Check the x values are > 0
        """
        if np.any(self.x <= 0):
            raise ValueError(f"{self.x_name} must be positive.")

    def check_x_non_negative(self):
        """
        Check the x values are >= 0
        """
        if np.any(self.x < 0):
            raise ValueError(f"{self.x_name} must be non-negative.")

    def check_y_positive(self):
        """
        Check the y values are > 0
        """
        if np.any(self.y <= 0):
            raise ValueError(f"{self.y_name} must be positive.")

    def check_y_non_negative(self):
        """
        Check the y values are >= 0
        """
        if np.any(self.y < 0):
            raise ValueError(f"{self.y_name} must be non-negative.")

    def check_y_less_equal_one(self):
        """
        Check the y values are <= 1
        """
        if np.any(self.y > 1):
            raise ValueError(f"{self.y_name} must be <= 1.")

    def check_abs_y_less_equal_one(self):
        """
        Check that |y| <= 1
        """
        if np.any(abs(self.y) > 1):
            raise ValueError(f"|{self.y_name}| must be <= 1.")

    @property
    def size(self) -> int:
        """
        Lenght of the data columns
        """
        return len(self.x)
