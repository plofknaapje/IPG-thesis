from dataclasses import dataclass
from time import time

import gurobipy as gp
from gurobipy import GRB
import numpy as np
from numpy.random import Generator

from problems.base import IPGResult, early_stopping


@dataclass
class CNGParams:
    success: float  # Delta
    mitigated: float  # Eta
    overcommit: float  # Epsilon
    normal: float  # Gamma
    capacity_perc: list[float, float]

    def __init__(
        self,
        success: float | None,
        mitigated: float | None,
        overcommit: float | None,
        normal: float | None,
        capacity_perc: list[float, float] | None = None,
        rng: Generator | None = None,
    ):
        if rng is None:
            rng = np.random.default_rng()

        if overcommit is None or mitigated is None or success is None:
            self.mitigated = np.round(rng.uniform(0.55, 0.85), 2)
            self.overcommit = self.mitigated * 1.25
            self.success = self.mitigated * 0.8
        else:
            self.overcommit = overcommit
            self.success = success
            self.mitigated = mitigated

        if normal is None:
            self.normal = np.round(rng.uniform(0, 0.15), 2)
        else:
            self.normal = normal

        if capacity_perc is None:
            defender_budget = rng.uniform(0.3, 0.8)
            self.capacity_perc = [
                defender_budget,
                defender_budget / rng.integers(5, 10),
            ]
        else:
            self.capacity_perc = capacity_perc

        if self.success >= self.mitigated or self.mitigated >= self.overcommit:
            raise ValueError("Invalid parameters for CNG")


@dataclass
class CriticalNodeGame:
    n: int  # number of targets
    weights: np.ndarray  # costs for defender and attacker (2, n)
    payoffs: np.ndarray  # rewards for defender and attacker (2, n)
    capacity: list[int, int]  # budget/capacity of defender and attacker (2, n)
    overcommit: float  # sunken cost of defender.
    mitigated: float  # mitigation costs / reward, eta < eps.
    success: float  # attack cost of defender. delta < eta.
    normal: float  # opportunit cost of defender.
    solution: np.ndarray | None = None
    result: IPGResult | None = None

    def __init__(
        self,
        weights: np.ndarray,
        payoffs: np.ndarray,
        parameters: CNGParams,
        rng: Generator | None = None,
    ):
        """
        Args:
            weights (np.ndarray): _description_
            payoffs (np.ndarray): _description_
            parameters (CNGParams): Object for storing and generating CriticalNodeGame parameters.
            cap_percentage (list[float, float] | None, optional): Capacity percentage per player. Defaults to None.
            rng (Generator | None, optional): Random number generator. Defaults to None.
        """
        if rng is None:
            rng = np.random.default_rng()

        self.n = weights.shape[1]
        self.weights = weights
        self.payoffs = payoffs

        self.capacity = list(
            [int(parameters.capacity_perc[i] * weights[i].sum()) for i in [0, 1]]
        )
        self.overcommit = parameters.overcommit
        self.mitigated = parameters.mitigated
        self.success = parameters.success
        self.normal = parameters.normal

    def solve(self, timelimit: int | None = None, verbose=False) -> IPGResult:
        """
        Solve the CNG using zero_regrets(). Sets self.solution if this was None.
        Sets self.result too, which records the information of the solution.

        Args:
            timelimit (int | None, optional): Runtime limit in seconds. Defaults to None.
            verbose (bool, optional): Verbose progress reports. Defaults to False.

        Returns:
            IPGResult: Object with all solving information.
        """
        if self.solution is not None:
            print("Problem was already solved")
            return self.result

        self.result = zero_regrets_cng(self, timelimit, verbose)
        self.solution = self.result.X
        return self.result

    def solve_player(
        self,
        defender: bool,
        current_sol: np.ndarray | None = None,
        weights: np.ndarray | None = None,
        payoffs: np.ndarray | None = None,
        timelimit: int | None = None,
    ) -> np.ndarray:
        """
        Solve the partial problem for a single player given the current solution.

        Args:
            defender (bool): Solve for the defender.
            current_sol (np.ndarray): Matrix of current solution.
            weights (np.ndarray | None, optional): Replacement weights matrix. Defaults to None.
            payoffs (np.ndarray | None, optional): Replacement payoff matrix. Defaults to None.
            timelimit (int | None, optional): Soft timelimit for improvement. Defaults to None.

        Raises:
            ValueError: Problem is infeasible.
        Returns:
            np.ndarray: Optimal solution for the player.
        """
        if current_sol is None:
            solution = self.solution
        else:
            solution = current_sol

        if weights is None:
            w = self.weights
        else:
            w = weights

        if payoffs is None:
            p = self.payoffs
        else:
            p = payoffs

        model = gp.Model("CNG player")

        x = model.addMVar((self.n), vtype=GRB.BINARY, name="x")

        if defender:
            attack = solution[1]
            model.setObjective(
                p[0]
                @ (
                    (1 - x) * (1 - attack)
                    + self.mitigated * x * attack
                    + self.overcommit * x * (1 - attack)
                    + self.success * (1 - x) * attack
                ),
                GRB.MAXIMIZE,
            )

            model.addConstr(w[0] @ x <= self.capacity[0])
        else:
            defence = solution[0]
            model.setObjective(
                p[1]
                @ (
                    -self.normal * (1 - defence) * (1 - x)
                    + (1 - defence) * x
                    + (1 - self.mitigated) * defence * x
                ),
                GRB.MAXIMIZE,
            )
            model.addConstr(w[1] @ x <= self.capacity[1])

        if timelimit is None:
            model.optimize()
        else:
            model._timelimit = timelimit
            model._current_obj = self.obj_value(
                defender, current_sol[0], current_sol[1]
            )

            model.optimize(early_stopping)

        if model.Status == GRB.INFEASIBLE:
            raise ValueError("Problem is Infeasible!")

        result = x.X

        model.close()

        return result

    def obj_value(
        self,
        defender: bool,
        def_sol: np.ndarray | None = None,
        att_sol: np.ndarray | None = None,
        payoffs: np.ndarray | None = None,
    ) -> float:
        """
        Calculate the objective value for defender or attacker. If def_sol or att_sol is not given, self.solution is used.

        Args:
            defender (True): Calculate for the defender?
            def_sol (np.ndarray | None, optional): Defender solution. Defaults to None.
            att_sol (np.ndarray | None, optional): Attacker solution. Defaults to None.
            payoffs (np.ndarray | None, optional): Replacement payoff matrix. Defaults to None.

        Returns:
            float: Objective function value for the selected player.
        """
        if def_sol is None:
            defence = self.solution[0]
        else:
            defence = def_sol

        if att_sol is None:
            attack = self.solution[1]
        else:
            attack = att_sol

        if defender:  # Defender
            return self.payoffs[0] @ (
                (1 - defence) * (1 - attack)
                + self.mitigated * defence * attack
                + self.overcommit * defence * (1 - attack)
                + self.success * (1 - defence) * attack
            )
        else:  # Attacker
            return self.payoffs[1] @ (
                -self.normal * (1 - defence) * (1 - attack)
                + (1 - defence) * attack
                + (1 - self.mitigated) * defence * attack
            )


def generate_random_CNG(
    n: int = 20,
    r: int = 50,
    capacity: list[float] | None = None,
    rng: Generator | None = None,
) -> CriticalNodeGame:
    """
    Generate a random instance of the CriticalNodeGame.

    Args:
        n (int, optional): Number of nodes. Defaults to 20.
        r (int, optional): Range of weight and payoff values. Defaults to 50.
        capacity (list[float] | None, optional): Fractional capacity of players. Defaults to None.
        rng (Generator | None, optional): Random number generator. Defaults to None.

    Returns:
        CriticalNodeGame: CNG instance.
    """
    if rng is None:
        rng = np.random.default_rng()

    weights = rng.integers(1, r + 1, (2, n))
    payoffs = weights + rng.integers(1, r + 1, (2, n))

    return CriticalNodeGame(weights, payoffs, capacity, rng)


def zero_regrets_cng(
    cng: CriticalNodeGame, timelimit: int | None = None, verbose=False
) -> IPGResult:
    """
    Solves the CriticalNodeGame instance to a PNE if possible and otherwise to a phi-NE.

    Args:
        cng (CriticalNodeGame): Problem instance.
        timelimit (int | None, optional): Runtime limit in seconds. Defaults to None.
        verbose (bool, optional): Verbally report progress. Defaults to False.

    Raises:
        ValueError: Problem is infeasible.

    Returns:
        IPGResult: Object with all solving information.
    """
    start = time()
    i_range = list(range(cng.n))
    phi_ub = 0

    model = gp.Model("ZeroRegrets CNG")

    phi = model.addVar(lb=0, ub=phi_ub)
    x = model.addMVar((2, cng.n), vtype=GRB.BINARY, name="x")
    defender = x[0]
    attacker = x[1]

    not_def = model.addMVar((cng.n), vtype=GRB.BINARY)
    not_att = model.addMVar((cng.n), vtype=GRB.BINARY)
    def_and_att = model.addMVar((cng.n), vtype=GRB.BINARY)
    not_def_and_not_att = model.addMVar((cng.n), vtype=GRB.BINARY)
    def_and_not_att = model.addMVar((cng.n), vtype=GRB.BINARY)
    not_def_and_att = model.addMVar((cng.n), vtype=GRB.BINARY)

    def_obj = cng.payoffs[0] @ (
        not_def_and_not_att
        + cng.mitigated * def_and_att
        + cng.overcommit * def_and_not_att
        + cng.success * not_def_and_att
    )

    att_obj = cng.payoffs[1] @ (
        -cng.normal * not_def_and_not_att
        + not_def_and_att
        + (1 - cng.mitigated) * def_and_att
    )

    model.setObjective(def_obj + att_obj, GRB.MAXIMIZE)

    model.addConstr(defender @ cng.weights[0] <= cng.capacity[0])
    model.addConstr(attacker @ cng.weights[1] <= cng.capacity[1])

    for i in i_range:
        # NOT Defender
        model.addConstr(not_def[i] == 1 - defender[i])

        # NOT Attacker
        model.addConstr(not_att[i] == 1 - attacker[i])

        # Defender AND Attacker
        model.addConstr(def_and_att[i] == gp.and_(defender[i], attacker[i]))

        # NOT Defender AND NOT Attacker
        model.addConstr(not_def_and_not_att[i] == gp.and_(not_def[i], not_att[i]))

        # Defender AND NOT Attacker
        model.addConstr(def_and_not_att[i] == gp.and_(defender[i], not_att[i]))

        # NOT Defender AND Attacker
        model.addConstr(not_def_and_att[i] == gp.and_(not_def[i], attacker[i]))

    model.optimize()

    if model.Status == GRB.INFEASIBLE:
        raise ValueError("Problem is Infeasible!")

    defender_solutions = set()
    attacker_solutions = set()

    while True:
        new_constraint = False
        current_x = x.X
        current_obj = model.ObjVal

        # Defender
        new_def_x = cng.solve_player(True, x.X, timelimit=1)
        new_def_obj = cng.obj_value(True, new_def_x, attacker.X)

        # Attacker
        new_att_x = cng.solve_player(False, x.X, timelimit=1)
        new_att_obj = cng.obj_value(False, defender.X, new_att_x)

        if (
            def_obj.getValue() + phi_ub <= new_def_obj
            and tuple(new_def_x) not in defender_solutions
        ):
            model.addConstr(
                cng.payoffs[0]
                @ (
                    (1 - new_def_x) * (1 - attacker)
                    + cng.mitigated * new_def_x * attacker
                    + cng.overcommit * new_def_x * (1 - attacker)
                    + cng.success * (1 - new_def_x) * attacker
                )
                <= def_obj + phi
            )
            defender_solutions.add(tuple(new_def_x))
            new_constraint = True

        if (
            att_obj.getValue() + phi_ub <= new_att_obj
            and tuple(new_att_x) not in attacker_solutions
        ):
            model.addConstr(
                cng.payoffs[1]
                @ (
                    -cng.normal * (1 - defender) * (1 - new_att_x)
                    + (1 - defender) * new_att_x
                    + (1 - cng.mitigated) * defender * new_att_x
                )
                <= att_obj + phi
            )
            attacker_solutions.add(tuple(new_att_x))
            new_constraint = True

        if not new_constraint:
            break

        model.optimize()

        while model.Status == GRB.INFEASIBLE:
            print("IPG is not feasible, increasing phi upper bound!")
            phi_ub += 1
            phi.ub = phi_ub
            model.optimize()

        if verbose:
            print(model.ObjVal)

        if timelimit is None:
            continue
        elif time() - start >= timelimit:
            print("Timelimit reached!")
            pne = False
            result = x.X
            objval = model.ObjVal
            phi = phi.X
            model.close()
            return IPGResult(pne, result, objval, time() - start, phi, True)

    if phi_ub >= 1:
        pne = False
        result = current_x
        objval = current_obj
        phi = phi.X
    else:
        pne = True
        result = x.X
        objval = model.ObjVal
        phi = phi.X

    model.close()

    runtime = time() - start

    return IPGResult(pne, result, objval, runtime, phi)


if __name__ == "__main__":
    instance = generate_random_CNG(n=20, r=25)
    print(zero_regrets_cng(instance, True))
