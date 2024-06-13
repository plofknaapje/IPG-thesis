from time import time
import itertools
from typing import List, Optional

from pydantic import BaseModel, validate_call
import numpy as np
import gurobipy as gp
from gurobipy import GRB

from problems.utils import powerset
from problems.base import IPGResult, early_stopping, allow_nparray

eps = 0.001


class KnapsackPackingGame(BaseModel):
    model_config = allow_nparray
    # Class for storing binary Knapsack Packing Games.
    n: int  # number of players
    m: int  # number of items
    players: List[int]  # list of player indices, length n
    pairs: List[List[int]]  # list of all possible pairs of players
    opps: List[List[int]]  # opponents of each player
    capacity: List[int]  # capacity of each player, length n
    weights: np.ndarray  # weights of the items, (n, m)
    payoffs: np.ndarray  # payoffs of the items, (n, m)
    inter_coefs: np.ndarray  # interaction payoff of the items (n, n, m)
    solution: Optional[np.ndarray] = None
    result: Optional[IPGResult] = None

    def __init__(
        self,
        weights: np.ndarray,
        payoffs: np.ndarray,
        inter_coefs: np.ndarray,
        capacity: List[float],
    ):
        """
        Create a new instance of the KnapsackPackingGame

        Args:
            weights (np.ndarray): Weights matrix.
            payoffs (np.ndarray): Payoffs matrix.
            inter_coefs (np.ndarray): Interaction matrix.
            capacity (List[float]): Fractional capacity per player.
        """
        n, m = weights.shape
        players = list(range(n))
        pairs = list(itertools.permutations(players, 2))
        opps = [[o for p, o in pairs if p == j] for j in players]
        capacity = [int(capacity[j] * weights[j].sum()) for j in players]

        super().__init__(
            n=n,
            m=m,
            players=players,
            pairs=pairs,
            opps=opps,
            capacity=capacity,
            weights=weights,
            payoffs=payoffs,
            inter_coefs=inter_coefs,
        )

    def print_data(self):
        # Prints the payoffs, weights and interaction coefficients of the KPG.
        print("Payoffs")
        print(self.payoffs)
        print("Weights")
        print(self.weights)
        print("Interaction coefficients")
        print(self.inter_coefs)

    @validate_call
    def solve(self, verbose=False, timelimit: Optional[int] = None) -> IPGResult:
        """
        Solve the KPG using zero_regrets(). Sets self.solution if this was None.
        Sets self.result too, which records the information of the solution.

        Args:
            verbose (bool, optional): Transfered to zero_regrets(verbose). Defaults to False.
            timelimit (int, optional): Runtime limit in seconds. Defaults to None.

        Returns:
            IPGResult: Object with all solving information.
        """
        if self.solution is not None:
            print("Already solved")
            return self.result

        self.result = zero_regrets_kpg(self, timelimit, verbose)
        self.solution = self.result.X
        return self.result

    @validate_call(config=allow_nparray)
    def solve_player(
        self,
        player: int,
        current_sol: Optional[np.ndarray] = None,
        weights: Optional[np.ndarray] = None,
        payoffs: Optional[np.ndarray] = None,
        inter_coefs: Optional[np.ndarray] = None,
        timelimit: Optional[int] = None,
    ) -> np.ndarray:
        if current_sol is None:
            if self.solution is None:
                raise ValueError("KPG was not yet solved!")
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

        if inter_coefs is None:
            ic = self.inter_coefs
        else:
            ic = inter_coefs

        model = gp.Model("KPG player")

        x = model.addMVar((self.m), vtype=GRB.BINARY, name="x")

        model.setObjective(
            x @ p[player]
            + gp.quicksum(
                x * solution[opp] @ ic[player, opp] for opp in self.opps[player]
            ),
            GRB.MAXIMIZE,
        )

        model.addConstr(x @ w[player] <= self.capacity[player])

        if timelimit is None:
            model.optimize()
        else:
            model._timelimit = timelimit
            model._current_obj = self.obj_value(player, solution)
            model.optimize(early_stopping)

        if model.Status == GRB.INFEASIBLE:
            raise ValueError("Problem is Infeasible!")

        result = x.X

        model.close()

        return result

    @validate_call(config=allow_nparray)
    def solve_greedy(self) -> np.ndarray:
        change = True

        solution = np.zeros_like(self.payoffs)
        sols = [set() for _ in self.players]

        for p in self.players:
            solution[p] = self.solve_player(p, solution)
            sols[p].add(tuple(solution[p]))

        while change:
            change = False
            for p in self.players:
                solution[p] = self.solve_player(p, solution)
                if tuple(solution[p]) not in sols[p]:
                    change = True
                    sols[p].add(tuple(solution[p]))

        return solution

    @validate_call(config=allow_nparray)
    def obj_value(
        self,
        player: int,
        solution: Optional[np.ndarray] = None,
        player_solution: Optional[np.ndarray] = None,
        payoffs: Optional[np.ndarray] = None,
        inter_coefs: Optional[np.ndarray] = None,
    ) -> int:
        """
        Calculates the objective value for a player based on a given solution.
        If player_solution is not None, its used for the player.
        If solution is not None, it is used instead of self.solution.
        Payoffs and inter_coefs replace self.payoffs and self.inter_coefs.

        Args:
            player (int): Index of the player.
            solution (np.ndarray, optional): Solution matrix. Defaults to None.
            player_solution (np.ndarray, optional): Player solution vector. Defaults to None.
            payoffs (np.ndarray, optional): Replacement matrix. Defaults to None.
            inter_coefs (np.ndarray, optional): Replacement matrix. Defaults to None.

        Returns:
            int: Objective value of player under the given solution, payoffs and inter_coefs.
        """

        if player_solution is not None:
            player_sol = player_solution
        elif solution is not None:
            player_sol = solution[player]
        else:
            player_sol = self.solution[player]

        if payoffs is None:
            payoffs = self.payoffs
        if inter_coefs is None:
            inter_coefs = self.inter_coefs

        if solution is not None:
            others_sol = solution
        else:
            others_sol = self.solution

        value = player_sol @ payoffs[player]

        for opp in self.opps[player]:
            value += player_sol * others_sol[opp] @ inter_coefs[player, opp]

        return value


@validate_call
def generate_random_KPG(
    n=2,
    m=25,
    r=100,
    capacity=0.4,
    corr=True,
    inter_factor=3,
) -> KnapsackPackingGame:
    rng = np.random.default_rng()
    weights = rng.integers(1, r + 1, (n, m))

    if corr:
        lower = np.maximum(weights - r / 5, 1)
        upper = np.minimum(weights + r / 5, r + 1)

    if corr:
        payoffs = rng.integers(lower, upper)
    else:
        payoffs = rng.integers(1, r + 1, (n, m))

    interactions = rng.integers(1, int(r / inter_factor) + 1, (n, n, m))
    mask = rng.integers(0, 2, (n, n, m))
    interactions = interactions * mask

    for i in range(n):
        interactions[i, i, :] = 0

    # Interaction cleanup
    for j in range(m):
        for p in range(n):
            interactions[p, p, j] = 0

    kpg = KnapsackPackingGame(weights, payoffs, interactions, capacity)

    return kpg


@validate_call
def zero_regrets_kpg(
    kpg: KnapsackPackingGame, timelimit: Optional[int] = None, verbose=False
) -> IPGResult:
    """
    Solves kpg using the ZeroRegrets methods of cutting.

    Args:
        kpg (KPG): KPG problem.
        verbose (bool, optional): Print progress?. Defaults to False.
        timelimit (int, optional): Runtime limit in seconds. Defaults to None.

    Returns:
        IPGResult: Object with all solving information.
    """
    # Only for n=2!
    # TODO: extend for higher n.
    start = time()
    phi_ub = 0

    model = gp.Model("ZeroRegrets KPG")
    x = model.addMVar((kpg.n, kpg.m), vtype=GRB.BINARY, name="x")
    z = model.addMVar((kpg.n, kpg.n, kpg.m), vtype=GRB.BINARY, name="z")
    phi = model.addVar(lb=0, ub=phi_ub)

    model.setObjective(
        gp.quicksum(x[p, :] @ kpg.payoffs[p, :] for p in kpg.players)
        + gp.quicksum(
            kpg.inter_coefs[p1, p2, :] @ z[p1, p2, :] for p1, p2 in kpg.pairs
        ),
        GRB.MAXIMIZE,
    )

    for p in kpg.players:
        # Capacity constraint
        model.addConstr(kpg.weights[p, :] @ x[p, :] <= kpg.capacity[p])

    for p1, p2 in kpg.pairs:
        for j in range(kpg.m):
            if p1 > p2:
                model.addConstr(z[p1, p2, j] == z[p2, p1, j])
                continue
            model.addConstr(z[p1, p2, j] == gp.and_(x[p1, j], x[p2, j]))

    # If the highest possible value of an item k is lower than the lowest possible
    # value of an item j, then item j dominates item k.
    dominance = 0
    for p in kpg.players:
        opponent_sets = [s for s in powerset(kpg.players) if p not in s]
        for j in range(kpg.m):
            j_min = min(
                kpg.payoffs[p, j] + sum(kpg.inter_coefs[p, o, j] for o in ops)
                for ops in opponent_sets
            )
            for k in range(kpg.m):
                if k == j or kpg.weights[p, j] > kpg.weights[p, k]:
                    continue
                k_max = max(
                    kpg.payoffs[p, k] + sum(kpg.inter_coefs[p, o, k] for o in ops)
                    for ops in opponent_sets
                )
                if j_min >= k_max + eps:
                    model.addConstr(x[p, k] <= x[p, j])
                    dominance += 1

    model.optimize()

    if model.Status == GRB.INFEASIBLE:
        print("IPG is not feasible (anymore)!")

    if verbose:
        print("====")
    cuts = 0
    unequal_payoff = 0
    solutions = [set() for _ in kpg.players]

    while True:
        new_constraint = False
        current_x = x.X
        current_z = z.X
        current_obj = model.ObjVal
        if verbose:
            print(f"Current total is {current_obj}")

        # Check if a player has net-negative variables and exclude the solutions with them.
        for p in kpg.players:
            for j in range(kpg.m):
                if current_x[p, j] == 0 or (
                    kpg.payoffs[p, j] * current_x[p, j]
                    + sum(
                        kpg.inter_coefs[p, o, j] * current_z[p, o, j]
                        for o in range(kpg.n)
                        if o != p
                    )
                    >= 0
                ):
                    continue
                opponent_set = [
                    o for o in range(kpg.n) if o != p and kpg.inter_coefs[p, o, j] < 0
                ]
                model.addConstr(
                    x[p, j] + gp.quicksum(x[o, j] for o in opponent_set)
                    <= len(opponent_set)
                )
                unequal_payoff += 1

        # Add cuts to the problem for each player which has a better solution.
        for p in kpg.players:
            player_obj = kpg.obj_value(p, solution=current_x)

            new_player_x = kpg.solve_player(p, current_sol=current_x, timelimit=1)
            if tuple(new_player_x) in solutions[p]:
                continue

            new_player_obj = kpg.obj_value(
                p, solution=current_x, player_solution=new_player_x
            )
            if player_obj + phi_ub <= new_player_obj:
                # Add constraint!
                model.addConstr(
                    kpg.payoffs[p, :] @ new_player_x
                    + gp.quicksum(
                        kpg.inter_coefs[p, p2, j] * new_player_x[j] * x[p2, j]
                        for j in range(kpg.m)
                        for p2 in kpg.opps[p]
                    )
                    <= gp.quicksum(kpg.payoffs[p, j] * x[p, j] for j in range(kpg.m))
                    + gp.quicksum(
                        kpg.inter_coefs[p, p2, j] * z[p, p2, j]
                        for j in range(kpg.m)
                        for p2 in kpg.opps[p]
                    )
                    + phi
                )
                cuts += 1
                new_constraint = True
                solutions[p].add(tuple(new_player_x))

        if not new_constraint:
            break

        model.optimize()

        while model.Status == GRB.INFEASIBLE:
            print("IPG is not feasible, increasing phi upper bound!")
            phi_ub += 1
            phi.ub = phi_ub
            model.optimize()

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

    if verbose:
        print("====")
        print(f"Added {dominance} dominance constraints.")
        print(f"Added {cuts} cuts.")
        print(f"Added {unequal_payoff} unequal payoff constraints.")
        print(f"{model.ObjVal} in {time() - start} seconds")

    if phi_ub >= 1:
        pne = False
        result = x.X
        objval = model.ObjVal
        phi = phi.X
    else:
        pne = True
        result = x.X
        objval = model.ObjVal
        phi = phi.X

    model.close()

    return IPGResult(pne, result, objval, time() - start, phi)
