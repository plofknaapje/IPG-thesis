from dataclasses import dataclass
from time import time
import itertools

import numpy as np
import gurobipy as gp
from gurobipy import GRB

from problems.utils import powerset
from problems.base import IPGResult, early_stopping
from problems.knapsack_problem import KnapsackProblem

eps = 0.001


@dataclass
class KnapsackPackingGame:
    # Class for storing binary Knapsack Packing Games.

    n: int  # number of players
    m: int  # number of items
    players: list[int]  # list of player indices, length n
    pairs: list[list[int]]  # list of all possible pairs of players
    opps: list[list[int]]  # opponents of each player
    capacity: list[int]  # capacity of each player, length n
    weights: np.ndarray  # weights of the items, (n, m)
    payoffs: np.ndarray  # payoffs of the items, (n, m)
    inter_coefs: (
        np.ndarray
    )  # interaction payoff of the items (n, n, m) with 0 on diagonals
    solution: np.ndarray | None
    result: IPGResult | None

    def __init__(
        self,
        weights: np.ndarray,
        payoffs: np.ndarray,
        inter_coefs: np.ndarray,
        capacity: list[int],
    ):
        """
        Create a new instance of the KnapsackPackingGame

        Args:
            weights (np.ndarray): Weights matrix.
            payoffs (np.ndarray): Payoffs matrix.
            inter_coefs (np.ndarray): Interaction matrix.
            capacity (list[float]): Fractional capacity per player.
        """
        self.n, self.m = weights.shape
        self.weights = weights
        self.payoffs = payoffs
        self.inter_coefs = inter_coefs
        self.players = list(range(self.n))
        self.pairs = list(itertools.permutations(self.players, 2))
        self.opps = [[o for p, o in self.pairs if p == j] for j in self.players]
        self.capacity = capacity
        self.solution = None
        self.result = None

    def print_data(self):
        # Prints the payoffs, weights and interaction coefficients of the KPG.
        print("Payoffs")
        print(self.payoffs)
        print("Weights")
        print(self.weights)
        print("Interaction coefficients")
        print(self.inter_coefs)

    def solve(self, verbose=False, timelimit: int | None = None) -> IPGResult:
        """
        Solve the KPG using zero_regrets(). Sets self.solution if this was None.
        Sets self.result too, which records the information of the solution.

        Args:
            verbose (bool, optional): Transfered to zero_regrets(verbose). Defaults to False.
            timelimit (int | None, optional): Runtime limit in seconds. Defaults to None.

        Returns:
            IPGResult: Object with all solving information.
        """
        if self.solution is not None:
            print("Already solved")
            return self.result

        self.result = zero_regrets_kpg(self, timelimit, verbose)
        self.solution = self.result.X
        return self.result

    def solve_player(
        self,
        player: int,
        current_sol: np.ndarray | None = None,
        weights: np.ndarray | None = None,
        payoffs: np.ndarray | None = None,
        inter_coefs: np.ndarray | None = None,
        timelimit: int | None = None,
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
                x * solution[opp] @ ic[player, opp]
                for opp in self.opps[player]
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

    def solve_greedy(self) -> np.ndarray:
        # Solves KP without looking at interactions

        initial_x = np.zeros_like(self.payoffs)

        for p in self.players:
            kp = KnapsackProblem(self.payoffs[p], self.weights[p], self.capacity[p])
            initial_x[p] = kp.solve()

        final_x = np.zeros_like(initial_x)
        for p in self.players:
            payoffs = self.payoffs[p] + (self.inter_coefs[p] * initial_x).sum(axis=0)

            kp = KnapsackProblem(payoffs, self.weights[p], self.capacity[p])
            final_x[p] = kp.solve()

        return final_x

    def obj_value(
        self,
        player: int,
        solution: np.ndarray | None = None,
        player_solution: np.ndarray | None = None,
        payoffs: np.ndarray | None = None,
        inter_coefs: np.ndarray | None = None,
    ) -> int:
        """
        Calculates the objective value for a player based on a given solution.
        If player_solution is not None, its used for the player.
        If solution is not None, it is used instead of self.solution.
        Payoffs and inter_coefs replace self.payoffs and self.inter_coefs.

        Args:
            player (int): Index of the player.
            solution (np.ndarray | None, optional): Solution matrix. Defaults to None.
            player_solution (np.ndarray | None, optional): Player solution vector. Defaults to None.
            payoffs (np.ndarray | None, optional): Replacement matrix. Defaults to None.
            inter_coefs (np.ndarray | None, optional): Replacement matrix. Defaults to None.

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

    capacity = [capacity * weights[p].sum() for p in range(n)]

    kpg = KnapsackPackingGame(weights, payoffs, interactions, capacity)

    return kpg


def read_file(file: str) -> KnapsackPackingGame:
    with open(file) as f:
        lines = [line.strip() for line in f]

    n, m = [int(i) for i in lines[0].split(" ")]
    capacity = [int(i) for i in lines[1].split(" ")]
    weights = np.zeros((n, m))
    payoffs = np.zeros((n, m))
    interaction_coefs = np.zeros((n, n, m))

    for j, line in enumerate(lines[2:]):
        line = [int(i) for i in line.split()][1:]
        for p in range(n):
            payoffs[p, j] = line[p * 2]
            weights[p, j] = line[p * 2 + 1]
        coef_list = line[2 * n :]
        index = 0
        for p1 in range(n):
            for p2 in range(n):
                if p1 == p2:
                    continue
                interaction_coefs[p1, p2, j] = coef_list[index]
                index += 1

    kpg = KnapsackPackingGame(weights, payoffs, interaction_coefs, capacity)

    return kpg


def zero_regrets_kpg(
    kpg: KnapsackPackingGame,  timelimit: int | None = None, verbose=False
) -> IPGResult:
    """
    Solves kpg using the ZeroRegrets methods of cutting.

    Args:
        kpg (KPG): KPG problem.
        verbose (bool, optional): Print progress?. Defaults to False.
        timelimit (int | None, optional): Runtime limit in seconds. Defaults to None.

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
