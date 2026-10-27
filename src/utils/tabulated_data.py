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
        ...

    def to_file(self, filename: str) -> None:
        ...