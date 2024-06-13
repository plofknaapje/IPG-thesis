from time import time
from typing import List, Tuple, Optional

from pydantic import BaseModel, validate_call
import gurobipy as gp
from gurobipy import GRB
import numpy as np
from numpy.random import Generator

from problems.base import IPGResult, early_stopping, allow_nparray


class CNGParams(BaseModel):
    success: float  # Delta
    mitigated: float  # Eta
    unchallenged: float  # Epsilon
    normal: float  # Gamma
    capacity_perc: Tuple[float, float]

    def __init__(
        self,
        success: Optional[float] = None,
        mitigated: Optional[float] = None,
        unchallenged: Optional[float] = None,
        normal: Optional[float] = None,
        capacity_perc: Optional[Tuple[float, float]] = None,
        rng: Optional[Generator] = None,
    ):
        if rng is None:
            rng = np.random.default_rng()

        if unchallenged is None or mitigated is None or success is None:
            mitigated = np.round(rng.uniform(0.55, 0.85), 2)
            unchallenged = mitigated * 1.25
            success = mitigated * 0.8

        if normal is None:
            normal = np.round(rng.uniform(0, 0.15), 2)

        if capacity_perc is None:
            defender_budget = rng.uniform(0.3, 0.8)
            capacity_perc = (
                defender_budget,
                defender_budget / rng.integers(5, 10),
            )

        if success >= mitigated or mitigated >= unchallenged:
            raise ValueError("Invalid parameters for CNG")

        super().__init__(
            success=success,
            mitigated=mitigated,
            unchallenged=unchallenged,
            normal=normal,
            capacity_perc=capacity_perc,
        )

    def to_array(self) -> np.ndarray:
        arr = np.zeros(4)
        arr[0] = self.success
        arr[1] = self.mitigated
        arr[2] = self.unchallenged
        arr[3] = self.normal

        return arr


class CriticalNodeGame(BaseModel):
    model_config = allow_nparray

    n: int  # number of targets
    weights: np.ndarray  # costs for defender and attacker (2, n)
    payoffs: np.ndarray  # rewards for defender and attacker (2, n)
    capacity: List[int]  # budget/capacity of defender and attacker (2, n)
    overcommit: float  # sunken cost of defender.
    mitigated: float  # mitigation costs / reward, eta < eps.
    success: float  # attack cost of defender. delta < eta.
    normal: float  # opportunit cost of defender.
    solution: Optional[Tuple[np.ndarray, np.ndarray]] = None
    result: Optional[Tuple[IPGResult, IPGResult]] = None
    PNE: Optional[bool] = None

    def __init__(
        self,
        weights: np.ndarray,
        payoffs: np.ndarray,
        parameters: CNGParams,
        rng: Optional[Generator] = None,
    ):
        """
        Args:
            weights (np.ndarray): _description_
            payoffs (np.ndarray): _description_
            parameters (CNGParams): Object for storing and generating CriticalNodeGame parameters.
            cap_percentage (List[float, float], optional): Capacity percentage per player. Defaults to None.
            rng (Generator, optional): Random number generator. Defaults to None.
        """

        if rng is None:
            rng = np.random.default_rng()

        n = weights.shape[1]

        capacity = [int(parameters.capacity_perc[i] * weights[i].sum()) for i in [0, 1]]

        overcommit = parameters.unchallenged
        mitigated = parameters.mitigated
        success = parameters.success
        normal = parameters.normal

        super().__init__(
            n=n,
            weights=weights,
            payoffs=payoffs,
            capacity=capacity,
            overcommit=overcommit,
            mitigated=mitigated,
            success=success,
            normal=normal,
        )

    @validate_call
    def solve(self, timelimit: Optional[int] = None, verbose=False) -> List[IPGResult]:
        """
        Solve the CNG using zero_regrets(). Sets self.solution if this was None.
        Sets self.result too, which records the information of the solution.

        Args:
            timelimit (int, optional): Runtime limit in seconds. Defaults to None.
            verbose (bool, optional): Verbose progress reports. Defaults to False.

        Returns:
            IPGResult: Object with all solving information.
        """
        if self.solution is not None:
            print("Problem was already solved")
            return self.result

        try:
            def_result = zero_regrets_cng(
                self, defender=True, timelimit=timelimit, verbose=verbose
            )
            att_result = zero_regrets_cng(
                self,
                defender=False,
                timelimit=timelimit,
                hotstart=def_result.X,
                verbose=verbose,
            )

            self.result = (def_result, att_result)

        except UserWarning:
            raise UserWarning

        self.solution = (self.result[0].X, self.result[1].X)
        self.PNE = self.result[0].PNE and self.result[1].PNE

        return self.result

    @validate_call(config=allow_nparray)
    def solve_player(
        self,
        defender: bool,
        current_sol: Optional[np.ndarray] = None,
        weights: Optional[np.ndarray] = None,
        payoffs: Optional[np.ndarray] = None,
        params: Optional[np.ndarray] = None,
        timelimit: Optional[int] = None,
    ) -> np.ndarray:
        """
        Solve the partial problem for a single player given the current solution.

        Args:
            defender (bool): Solve for the defender.
            current_sol (np.ndarray): Matrix of current solution.
            weights (np.ndarray, optional): Replacement weights matrix. Defaults to None.
            payoffs (np.ndarray, optional): Replacement payoff matrix. Defaults to None.
            timelimit (int, optional): Soft timelimit for improvement. Defaults to None.

        Raises:
            ValueError: Problem is infeasible.
        Returns:
            np.ndarray: Optimal solution for the player.
        """
        if current_sol is None:
            if defender:
                solution = self.solution[0]
            else:
                solution = self.solution[1]
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

        if params is None:
            success = self.success
            mitigated = self.mitigated
            overcommit = self.overcommit
            normal = self.normal
        else:
            success = params[0]
            mitigated = params[1]
            overcommit = params[2]
            normal = params[3]
            Optional[timelimit] = None

        model = gp.Model("CNG player")

        x = model.addMVar((self.n), vtype=GRB.BINARY, name="x")

        if defender:
            attack = solution[1]
            model.setObjective(
                p[0]
                @ (
                    (1 - x) * (1 - attack)
                    + mitigated * x * attack
                    + overcommit * x * (1 - attack)
                    + success * (1 - x) * attack
                ),
                GRB.MAXIMIZE,
            )

            model.addConstr(w[0] @ x <= self.capacity[0])
        else:
            defence = solution[0]
            model.setObjective(
                p[1]
                @ (
                    -normal * (1 - defence) * (1 - x)
                    + (1 - defence) * x
                    + (1 - mitigated) * defence * x
                ),
                GRB.MAXIMIZE,
            )
            model.addConstr(w[1] @ x <= self.capacity[1])

        if timelimit is None:
            model.optimize()
        else:
            model._timelimit = timelimit
            model._current_obj = self.obj_value(defender, solution[0], solution[1])

            model.optimize(early_stopping)

        if model.Status == GRB.INFEASIBLE:
            raise ValueError("Problem is Infeasible!")

        result = x.X

        model.close()

        return result.astype(int)

    @validate_call
    def solve_greedy(self, timelimit: int = 10) -> np.ndarray:
        start = time()
        solution = np.zeros((2, self.n), dtype=int)
        sols = [set(), set()]
        while time() - start <= timelimit:
            solution[1] = self.solve_player(False, solution)
            solution[0] = self.solve_player(True, solution)

            if tuple(solution[0]) in sols[0] and tuple(solution[1]) in sols[1]:
                break
            else:
                sols[0].add(tuple(solution[0]))
                sols[1].add(tuple(solution[1]))

        return solution

    @validate_call(config=allow_nparray)
    def obj_value(
        self,
        defender: bool,
        def_sol: Optional[np.ndarray] = None,
        att_sol: Optional[np.ndarray] = None,
        payoffs: Optional[np.ndarray] = None,
    ) -> float:
        """
        Calculate the objective value for defender or attacker. If def_sol or att_sol is not given, self.solution is used.

        Args:
            defender (True): Calculate for the defender?
            def_sol (np.ndarray, optional): Defender solution. Defaults to None.
            att_sol (np.ndarray, optional): Attacker solution. Defaults to None.
            payoffs (np.ndarray, optional): Replacement payoff matrix. Defaults to None.

        Returns:
            float: Objective function value for the selected player.
        """
        if def_sol is None:
            if defender:
                defence = self.solution[0][0]
            else:
                defence = self.solution[1][0]
        else:
            defence = def_sol

        if att_sol is None:
            if defender:
                attack = self.solution[0][1]
            else:
                attack = self.solution[1][1]
        else:
            attack = att_sol

        if payoffs is None:
            p = self.payoffs
        else:
            p = payoffs

        if defender:  # Defender
            return p[0] @ (
                (1 - defence) * (1 - attack)
                + self.mitigated * defence * attack
                + self.overcommit * defence * (1 - attack)
                + self.success * (1 - defence) * attack
            )
        else:  # Attacker
            return p[1] @ (
                -self.normal * (1 - defence) * (1 - attack)
                + (1 - defence) * attack
                + (1 - self.mitigated) * defence * attack
            )

    def params(self) -> np.ndarray:
        arr = np.zeros(4)
        arr[0] = self.success
        arr[1] = self.mitigated
        arr[2] = self.overcommit
        arr[3] = self.normal

        return arr


@validate_call(config=allow_nparray)
def generate_random_CNG(
    n: int = 20,
    r: int = 25,
    params: Optional[CNGParams] = None,
    rng: Optional[Generator] = None,
) -> CriticalNodeGame:
    """
    Generate a random instance of the CriticalNodeGame.

    Args:
        n (int, optional): Number of nodes. Defaults to 20.
        r (int, optional): Range of weight and payoff values. Defaults to 50.
        params (CNGParams, optional): CNGParams settings. Defaults to None.
        rng (Generator, optional): Random number generator. Defaults to None.

    Returns:
        CriticalNodeGame: CNG instance.
    """
    if rng is None:
        rng = np.random.default_rng()

    weights = rng.integers(1, r + 1, (2, n))
    payoffs = weights + rng.integers(1, r + 1, (2, n))

    return CriticalNodeGame(weights, payoffs, params, rng)


@validate_call(config=allow_nparray)
def zero_regrets_cng(
    cng: CriticalNodeGame,
    defender: bool = True,
    timelimit: Optional[int] = None,
    hotstart: Optional[np.ndarray] = None,
    verbose=False,
) -> IPGResult:
    """
    Solves the CriticalNodeGame instance to a PNE if possible and otherwise to a phi-NE.

    Args:
        cng (CriticalNodeGame): Problem instance.
        timelimit (int, optional): Runtime limit in seconds. Defaults to None.
        verbose (bool, optional): Verbally report progress. Defaults to False.

    Raises:
        ValueError: Problem is infeasible.

    Returns:
        IPGResult: Object with all solving information.
    """
    start = time()
    if timelimit is not None:
        local_timelimit = timelimit
    i_range = list(range(cng.n))
    phi_ub = 0

    model = gp.Model("ZeroRegrets CNG")

    phi = model.addVar(lb=0, ub=0)
    x = model.addMVar((2, cng.n), vtype=GRB.BINARY, name="x")
    defence = x[0]
    attack = x[1]

    if hotstart is not None:
        x.Start = hotstart

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

    if defender is None:
        model.setObjective(def_obj + att_obj, GRB.MAXIMIZE)
    elif defender:
        model.setObjective(def_obj, GRB.MAXIMIZE)
    else:
        model.setObjective(att_obj, GRB.MAXIMIZE)

    model.addConstr(defence @ cng.weights[0] <= cng.capacity[0])
    model.addConstr(attack @ cng.weights[1] <= cng.capacity[1])

    for i in i_range:
        # NOT Defender
        model.addConstr(not_def[i] == 1 - defence[i])

        # NOT Attacker
        model.addConstr(not_att[i] == 1 - attack[i])

        # Defender AND Attacker
        model.addConstr(def_and_att[i] == gp.and_(defence[i], attack[i]))

        # NOT Defender AND NOT Attacker
        model.addConstr(not_def_and_not_att[i] == gp.and_(not_def[i], not_att[i]))

        # Defender AND NOT Attacker
        model.addConstr(def_and_not_att[i] == gp.and_(defence[i], not_att[i]))

        # NOT Defender AND Attacker
        model.addConstr(not_def_and_att[i] == gp.and_(not_def[i], attack[i]))

    model.optimize()

    if timelimit is not None:
        model.params.TimeLimit = local_timelimit
        model.optimize()
    if timelimit is not None:
        local_timelimit -= model.Runtime

    if model.Status == GRB.INFEASIBLE:
        raise ValueError("Problem is Infeasible!")

    solutions = [set(), set()]

    while True:
        if verbose:
            print(local_timelimit)

        new_constraint = False
        current_x = x.X
        current_obj = model.ObjVal

        # Defender
        new_def_x = cng.solve_player(True, x.X)
        new_def_obj = cng.obj_value(True, new_def_x, attack.X)

        # Attacker
        new_att_x = cng.solve_player(False, x.X)
        new_att_obj = cng.obj_value(False, defence.X, new_att_x)

        if (
            tuple(new_def_x) not in solutions[0]
            and def_obj.getValue() + phi_ub <= new_def_obj
        ):
            model.addConstr(
                cng.payoffs[0]
                @ (
                    (1 - new_def_x) * (1 - attack)
                    + cng.mitigated * new_def_x * attack
                    + cng.overcommit * new_def_x * (1 - attack)
                    + cng.success * (1 - new_def_x) * attack
                )
                <= def_obj + phi
            )
            solutions[0].add(tuple(new_def_x))
            new_constraint = True

        if (
            tuple(new_att_x) not in solutions[1]
            and att_obj.getValue() + phi_ub <= new_att_obj
        ):
            model.addConstr(
                cng.payoffs[1]
                @ (
                    -cng.normal * (1 - defence) * (1 - new_att_x)
                    + (1 - defence) * new_att_x
                    + (1 - cng.mitigated) * defence * new_att_x
                )
                <= att_obj + phi
            )
            solutions[1].add(tuple(new_att_x))
            new_constraint = True

        if not new_constraint:
            break

        if timelimit is not None:
            model.params.TimeLimit = max(1, local_timelimit)
        model.optimize()
        if timelimit is not None:
            local_timelimit -= model.Runtime

        while model.Status == GRB.INFEASIBLE and local_timelimit > 0:
            if verbose:
                print("IPG is not feasible, increasing phi upper bound!")
            phi_ub += 1
            phi.ub = phi_ub
            if timelimit is not None:
                model.params.TimeLimit = max(1, local_timelimit)
            model.optimize()
            if timelimit is not None:
                local_timelimit -= model.Runtime

        if timelimit is not None and (
            model.Status == GRB.TIME_LIMIT or local_timelimit <= 0
        ):
            if verbose:
                print("Timelimit reached!")

            try:
                result = x.X
            except gp.GurobiError:
                raise UserWarning("Timelimit without useful result")

            pne = False
            result = x.X
            objval = model.ObjVal
            phi = phi.X
            model.close()
            return IPGResult(pne, result, objval, time() - start, phi, True)

        if verbose:
            print(model.ObjVal)

    if phi.X > 0:
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
