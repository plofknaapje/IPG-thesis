from dataclasses import dataclass
from time import time

import gurobipy as gp
from gurobipy import GRB
import numpy as np
from numpy.random import Generator


@dataclass
class ADGResult:
    PNE: bool
    X: np.ndarray
    ObjVal: float
    runtime: float
    phi: float
    timelimit_reached: bool = False


@dataclass
class AttackerDefenderGame:
    n: int  # number of targets
    weights: np.ndarray  # costs for defender and attacker (2, n)
    payoffs: np.ndarray  # rewards for defender and attacker (2, n)
    capacity: tuple[int, int]  # budget/capacity of defender and attacker (2, n)
    overcommit: float  # sunken cost of defender.
    mitigated: float  # mitigation costs / reward, eta < eps.
    success: float  # attack cost of defender. delta < eta.
    normal: float  # opportunit cost of defender.
    solution: np.ndarray | None
    result: ADGResult | Noe

    def __init__(
        self,
        weights: np.ndarray,
        payoffs: np.ndarray,
        cap_percentage: tuple[float, float] | None = None,
        overcommit: float | None = None,
        mitigated: float | None = None,
        success: float | None = None,
        normal: float | None = None,
        rng: Generator | None = None,
    ):
        """
        Create a new instance of the AttackerDefenderGame.

        Args:
            weights (np.ndarray): _description_
            payoffs (np.ndarray): _description_
            cap_percentage (tuple[float, float] | None, optional): Capacity percentage per player. Defaults to None.
            overcommit (float | None, optional): Reward for protecting safe nodes. Defaults to None.
            mitigated (float | None, optional): Reward for mitigating an attack. Defaults to None.
            success (float | None, optional): Reward for a successful attack. Defaults to None.
            normal (float | None, optional): Attacker opportunity cost. Defaults to None.
            rng (Generator | None, optional): Random number generator. Defaults to None.
        """
        if rng is None:
            rng = np.random.default_rng()

        self.n = weights.shape[1]
        self.weights = weights
        self.payoffs = payoffs
        if cap_percentage is None:
            defender_budget = rng.uniform(0.3, 0.8)
            cap_percentage = [defender_budget, defender_budget / rng.integers(3, 10)]
        self.capacity = tuple(
            [int(cap_percentage[i] * weights[i].sum()) for i in [0, 1]]
        )

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

        self.solution = None
        self.result = None

    def solve(self, verbose=False, timelimit: int | None = None) -> ADGResult:
        """
        Solve self with the zero_regrets algorithm. Updates self.solution/

        Args:
            verbose (bool, optional): Verbose progress reports. Defaults to False.
            timelimit (int | None, optional): Runtime limit in seconds. Defaults to None.

        Returns:
            np.ndarray | None: Solution or None if the result is not a PNE.
        """
        if self.solution is not None:
            print("Problem was already solved")
            return self.result

        self.result = zero_regrets(self, verbose, timelimit)
        self.solution = self.result.X
        return self.result

    def solve_player(self, defender: bool, current_sol: np.ndarray) -> np.ndarray:
        """
        Solve the partial problem for a single player given the current solution.

        Args:
            defender (bool): Solve for the defender.
            current_sol (np.ndarray): Matrix of current solution.

        Raises:
            ValueError: Problem is infeasible.
        Returns:
            np.ndarray: Optimal solution for the player.
        """
        model = gp.Model("ADG player")

        x = model.addMVar((self.n), vtype=GRB.BINARY, name="x")

        if defender:
            opp_sol = current_sol[1]
            model.setObjective(
                self.payoffs[0]
                @ (
                    (1 - x) * (1 - opp_sol)
                    + self.mitigated * x * opp_sol
                    + self.overcommit * x * (1 - opp_sol)
                    + self.success * (1 - x) * opp_sol
                ),
                GRB.MAXIMIZE,
            )

            model.addConstr(self.weights[0] @ x <= self.capacity[0])
        else:
            opp_sol = current_sol[0]
            model.setObjective(
                self.payoffs[1]
                @ (
                    -self.normal * (1 - opp_sol) * (1 - x)
                    + (1 - opp_sol) * x
                    + (1 - self.mitigated) * opp_sol * x
                ),
                GRB.MAXIMIZE,
            )
            model.addConstr(self.weights[1] @ x <= self.capacity[1])

        model.optimize()

        if model.Status == GRB.INFEASIBLE:
            raise ValueError("Problem is Infeasible!")

        result = x.X

        model.close()

        return result

    def obj_value(
        self,
        defender: True,
        def_sol: np.ndarray | None = None,
        att_sol: np.ndarray | None = None,
    ) -> float:
        """
        Calculate the objective value for defender or attacker. If def_sol or att_sol is not given, self.solution is used.

        Args:
            defender (True): Calculate for the defender?
            def_sol (np.ndarray | None, optional): Defender solution. Defaults to None.
            att_sol (np.ndarray | None, optional): Attacker solution. Defaults to None.

        Returns:
            float: Objective function value for the selected player.
        """
        if def_sol is None:
            defender = self.solution[0]
        else:
            defender = def_sol

        if att_sol is None:
            attacker = self.solution[1]
        else:
            attacker = att_sol

        if defender:  # Defender
            return self.payoffs[0] @ (
                (1 - defender) * (1 - attacker)
                + self.mitigated * defender * attacker
                + self.overcommit * defender * (1 - attacker)
                + self.success * (1 - defender) * attacker
            )
        else:  # Attacker
            return self.payoffs[1] @ (
                -self.normal * (1 - defender) * (1 - attacker)
                + (1 - defender) * attacker
                + (1 - self.mitigated) * defender * attacker
            )


def generate_random_ADG(
    n: int = 20,
    r: int = 50,
    capacity: list[float] | None = None,
    rng: Generator | None = None,
) -> AttackerDefenderGame:
    """
    Generate a random instance of the AttackerDefenderGame.

    Args:
        n (int, optional): Number of nodes. Defaults to 20.
        r (int, optional): Range of weight and payoff values. Defaults to 50.
        capacity (list[float] | None, optional): Fractional capacity of players. Defaults to None.
        rng (Generator | None, optional): Random number generator. Defaults to None.

    Returns:
        AttackerDefenderGame: ADG instance.
    """
    if rng is None:
        rng = np.random.default_rng()

    weights = rng.integers(1, r + 1, (2, n))
    payoffs = weights + rng.integers(1, r + 1, (2, n))

    return AttackerDefenderGame(weights, payoffs, capacity, rng)


def zero_regrets(
    adg: AttackerDefenderGame, verbose=False, timelimit: int | None = None
) -> ADGResult:
    """
    Solves the AttackerDefenderGame instance to a PNE if possible and otherwise to a phi-NE.

    Args:
        adg (AttackerDefenderGame): Problem instance.
        verbose (bool, optional): Verbally report progress. Defaults to False.
        timelimit (int | None, optional): Runtime limit in seconds. Defaults to None.

    Raises:
        ValueError: Problem is infeasible.

    Returns:
        ADGResult: Results of the zero_regrets process.
    """
    start = time()
    i_range = list(range(adg.n))
    phi_ub = 0

    model = gp.Model("ZeroRegrets ADG")

    phi = model.addVar(lb=0, ub=phi_ub)
    x = model.addMVar((2, adg.n), vtype=GRB.BINARY, name="x")
    defender = x[0]
    attacker = x[1]

    not_def = model.addMVar((adg.n), vtype=GRB.BINARY)
    not_att = model.addMVar((adg.n), vtype=GRB.BINARY)
    def_and_att = model.addMVar((adg.n), vtype=GRB.BINARY)
    not_def_and_not_att = model.addMVar((adg.n), vtype=GRB.BINARY)
    def_and_not_att = model.addMVar((adg.n), vtype=GRB.BINARY)
    not_def_and_att = model.addMVar((adg.n), vtype=GRB.BINARY)

    def_obj = adg.payoffs[0] @ (
        not_def_and_not_att
        + adg.mitigated * def_and_att
        + adg.overcommit * def_and_not_att
        + adg.success * not_def_and_att
    )

    att_obj = adg.payoffs[1] @ (
        -adg.normal * not_def_and_not_att
        + not_def_and_att
        + (1 - adg.mitigated) * def_and_att
    )

    model.setObjective(def_obj + att_obj, GRB.MAXIMIZE)

    model.addConstr(defender @ adg.weights[0] <= adg.capacity[0])
    model.addConstr(attacker @ adg.weights[1] <= adg.capacity[1])

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
        curr_def_obj = def_obj.getValue()
        new_def_x = adg.solve_player(True, x.X)
        new_def_obj = adg.obj_value(True, new_def_x, attacker.X)

        # Attacker
        curr_att_obj = att_obj.getValue()
        new_att_x = adg.solve_player(False, x.X)
        new_att_obj = adg.obj_value(False, defender.X, new_att_x)

        if (
            curr_def_obj + phi_ub <= new_def_obj
            and tuple(new_def_x) not in defender_solutions
        ):
            model.addConstr(
                adg.payoffs[0]
                @ (
                    (1 - new_def_x) * (1 - attacker)
                    + adg.mitigated * new_def_x * attacker
                    + adg.overcommit * new_def_x * (1 - attacker)
                    + adg.success * (1 - new_def_x) * attacker
                )
                <= def_obj + phi
            )
            defender_solutions.add(tuple(new_def_x))
            new_constraint = True

        if (
            curr_att_obj + phi_ub <= new_att_obj
            and tuple(new_att_x) not in attacker_solutions
        ):
            model.addConstr(
                adg.payoffs[1]
                @ (
                    -adg.normal * (1 - defender) * (1 - new_att_x)
                    + (1 - defender) * new_att_x
                    + (1 - adg.mitigated) * defender * new_att_x
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
            pne = False
            result = x.X
            objval = model.ObjVal
            phi = phi.X
            model.close()
            return ADGResult(pne, result, objval, time() - start, phi, True)

    if phi.X >= 1:
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

    return ADGResult(pne, result, objval, runtime, phi)


if __name__ == "__main__":
    instance = generate_random_ADG(n=20, r=25)
    print(zero_regrets(instance, True))
