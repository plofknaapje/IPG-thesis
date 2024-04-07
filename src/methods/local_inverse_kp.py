import numpy as np

from problems.knapsack_problem import KnapsackProblem

def generate_problem(
    m: int, r: int = 100, capacity: float | None = None, corr: bool = True, rng=None
) -> KnapsackProblem:
    if rng is None:
        rng = np.random.default_rng()

    payoffs = rng.integers(1, r + 1, m)

    if capacity is None:
        capacity = float(rng.uniform(0.2, 0.8))

    if corr:
        weights = rng.integers(
            np.maximum(payoffs - r / 5, 1), np.minimum(payoffs + r / 5, r + 1), m
        )
    else:
        weights = rng.integers(1, r + 1, m)

    return KnapsackProblem(payoffs, weights, capacity * weights.sum())

