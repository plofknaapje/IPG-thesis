from itertools import chain, combinations
from typing import List

import numpy as np


def powerset(iterable):
    """
    Generates the powerset of interable.
    powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3).
    """
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def duplicate_array(arrays: list[np.ndarray], x: np.ndarray) -> bool:
    # Checks if arrays contains x.
    for solution in arrays:
        if np.array_equal(solution, x):
            return True
    return False


def abs_error(arr1: np.ndarray, arr2: np.ndarray) -> float:
    return np.abs(arr1 - arr2).sum()


def rel_error(base_arr: np.ndarray, new_arr: np.ndarray) -> float:
    return abs_error(base_arr, new_arr) / base_arr.sum()
