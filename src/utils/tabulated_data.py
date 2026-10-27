import numpy as np
from dataclasses import dataclass
from numpy.typing import ArrayLike
from typing import Self

@dataclass
class TabulatedData:
    x: ArrayLike
    y: ArrayLike
    x_name: str = "x"
    y_name: str = "y"
    min_points: int = 2

    def __post_init__(self):
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
        Save the speed distribution to a two-column text file.
        
        The first column contains x and the second column
        contains y.
        
        Parameters
        ----------
        filename: str
            Output filename.
        """
        if not filename:
            raise ValueError("Invalid filename")
        
        data = np.column_stack([self.x, self.y])
        
        np.savetxt(filename, data, header=f"{self.x_name} {self.y_name}")
