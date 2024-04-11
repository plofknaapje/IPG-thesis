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
    overcommit: float  # sunken cost of defender.
    mitigated: float  # mitigation costs / reward, eta < eps.
    success: float  # attack cost of defender. delta < eta.
    normal: float  # opportunit cost of defender.
    solution: np.ndarray | None

    def __init__(
        self,
        n: int,
        weights: np.ndarray,
        payoffs: np.ndarray,
        resources: list[float],
        overcommit: float | None = None,
        mitigated: float | None = None,
        success: float | None = None,
        normal: float | None = None,
        rng: Generator | None = None,
        solution: np.ndarray | None = None,
    ):
        if rng is None:
            rng = np.random.default_rng()

        self.n = n
        self.weights = weights
        self.payoffs = payoffs
        self.resources = [int(resources[i] * weights[i].sum()) for i in [0, 1]]
        if isinstance(overcommit, float):
            self.overcommit = overcommit
        elif isinstance(mitigated, float):
            self.overcommit = rng.uniform(mitigated, 1)
        else:
            self.overcommit = rng.uniform(0.6, 1)

        if isinstance(success, float):
            self.success = success
        elif isinstance(mitigated, float):
            self.success = rng.uniform(0, mitigated)
        else:
            self.success = rng.uniform(0, 0.4)

        if isinstance(mitigated, float):
            self.mitigated = mitigated
        else:
            self.mitigated = rng.uniform(success, overcommit)

        if normal is None:
            self.normal = rng.uniform(0.2, 0.8)

        self.solution = None

    def obj_value(self, player: int, solution: np.ndarray) -> int:
        if player == 0:  # Defender
            return self.p[0] @ (
                (1 - solution[0]) * (1 - solution[1])
                + self.mitigated * solution[0] * solution[1]
                + self.overcommit * solution[0] * (1 - solution[1])
                + self.success * (1 - solution[0]) * solution[1]
            )
        elif player == 1:  # Attacker
            return self.p[1] @ (
                -self.normal(1 - solution[0]) * (1 - solution[1])
                + (1 - solution[0]) * solution[1]
                + (1 - self.mitigated) * solution[0] * solution[1]
            )
        else:
            raise ValueError("Unknown player!")

    def solve(self, verbose=False) -> np.ndarray | None:
        if self.solution is not None:
            print("Problem was already solved")
            return self.solution

        result = zero_regrets(self, verbose)
        if result.PNE:
            self.solution = result.X
            return result.X
        else:
            return None


@dataclass
class ADGResult:
    PNR: bool
    X: np.ndarray
    ObjVal: int
    runtime: float


def zero_regrets(adg: AttackerDefenderGame, verbose=False) -> ADGResult:
    return ADGResult()
