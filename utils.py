from itertools import chain, combinations
import numpy as np

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def duplicate_array(arrays: list, x: np.ndarray) -> bool:
    for solution in arrays:
        if np.array_equal(solution, x):
            return True
    return False