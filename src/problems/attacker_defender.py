from dataclasses import dataclass

import gurobipy as gp
from gurobipy import GRB
import numpy as np
from numpy.random import Generator


@dataclass
class AttackerDefenderGame:
    n: int  # number of targets
    weights: np.ndarray  # costs for defender and attacker (2, n)
    payoffs: np.ndarray  # rewards for defender and attacker (2, n)
    resources: list[int]  # resources of defender and attacker (2, n)
    eps: float  # sunken cost of defender.
    eta: float  # mitigation costs / reward, eta < eps.
    delta: float  # attack cost of defender. delta < eta.
    gamma: float  # opportunit cost of defender.
    solution: np.ndarray | None

    def __init__(
        self,
        n: int,
        weights: np.ndarray,
        payoffs: np.ndarray,
        resources: list[float],
        eps: float | None = None,
        eta: float | None = None,
        delta: float | None = None,
        gamma: float | None = None,
        rng: Generator | None = None,
        solution: np.ndarray | None = None
    ):
        if rng is None:
            rng = np.random.default_rng()

        self.n = n
        self.weights = weights
        self.payoffs = payoffs
        self.resources = [int(resources[i] * weights[i].sum()) for i in [0, 1]]
        if isinstance(eps, float):
            self.eps = eps
        elif isinstance(eta, float):
            self.eps = rng.uniform(eta, 1)
        else:
            self.eps = rng.uniform(0.6, 1)

        if isinstance(delta, float):
            self.delta = delta
        elif isinstance(eta, float):
            self.delta = rng.uniform(0, eta)
        else:
            self.delta = rng.uniform(0, 0.4)

        if isinstance(eta, float):
            self.eta = eta
        else:
            self.eta = rng.uniform(delta, eps)

        if gamma is None:
            self.gamma = rng.uniform(0.2, 0.8)

        self.solution = None
