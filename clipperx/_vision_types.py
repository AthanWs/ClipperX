"""Narrow type boundary for the SciPy assignment used by visual tracks."""

from __future__ import annotations

from importlib import import_module
from typing import Callable, cast

import numpy as np
import numpy.typing as npt


AssignmentSolver = Callable[
    [npt.NDArray[np.float64]],
    tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]],
]
linear_sum_assignment = cast(
    AssignmentSolver, import_module("scipy.optimize").linear_sum_assignment
)
