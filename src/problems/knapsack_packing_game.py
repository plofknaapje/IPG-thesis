from dataclasses import dataclass
import time
import itertools

import numpy as np
import gurobipy as gp
from gurobipy import GRB

from problems.utils import powerset

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

    def __init__(
        self,
        weights: np.ndarray,
        payoffs: np.ndarray,
        inter_coefs: np.ndarray,
        capacity: list[float],
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
        self.capacity = [int(cap) for cap in capacity]
        self.solution = None

    def print_data(self):
        # Prints the payoffs, weights and interaction coefficients of the KPG.
        print("Payoffs")
        print(self.payoffs)
        print("Weights")
        print(self.weights)
        print("Interaction coefficients")
        print(self.inter_coefs)

    def solve(self, verbose=False) -> np.ndarray | None:
        """
        Solve the KPG using zero_regrets(). Sets self.solution if this was None.
        Also changes self.PNE. If self.PNE is True, then the solution is an equilibrium.
        If self.PNE is False, the KPG has no pure stable solution.

        Args:
            verbose (bool, optional): Transfered to zero_regrets(verbose). Defaults to False.

        Returns:
            np.ndarray | None: optimal solution for all players if that was found. None otherwise.
        """
        if self.solution is not None:
            print("Already solved")
            return self.solution

        result = zero_regrets(self, verbose)
        if result.PNE:
            self.solution = result.X
            return self.solution
        else:
            return None

    def solve_player(self, player: int, solution: np.ndarray) -> np.ndarray:
        model = gp.Model("KPG player")

        x = model.addMVar((self.m), vtype=GRB.BINARY, name="x")

        model.setObjective(
            x @ self.payoffs[player]
            + gp.quicksum(
                x * solution[opp] @ self.inter_coefs[player, opp]
                for opp in self.opps[player]
            ),
            GRB.MAXIMIZE,
        )

        model.addConstr(x @ self.weights[player] <= self.capacity[player])

        model.optimize()

        if model.Status == GRB.INFEASIBLE:
            raise ValueError("Problem is Infeasible!")

        result = x.X

        model.close()

        return result

    def solve_player_weights(
        self, weights: np.ndarray, player: int, solution: np.ndarray | None = None
    ) -> np.ndarray:
        """
        Solves the KPG from the perspective of one player given the solutions of
        other players using a given set of weights.

        Args:
            weights (np.ndarray): new weights.
            player (int): index of the player.
            solution (np.ndaray | None, optional): external solution. If None, use self.solution. Defaults to None.

        Raises:
            ValueError: if the problem is infeasible.

        Returns:
            np.ndarray: optimal solution to the problem.
        """
        if solution is None:
            solution = self.solution

        model = gp.Model("KPG player weights")

        x = model.addMVar((self.m), vtype=GRB.BINARY, name="x")

        model.setObjective(
            x @ self.payoffs[player]
            + gp.quicksum(
                x * solution[opp] @ self.inter_coefs[player, opp]
                for opp in self.opps[player]
            ),
            GRB.MAXIMIZE,
        )

        model.addConstr(x @ weights[player] <= self.capacity[player])

        model.optimize()

        if model.Status == GRB.INFEASIBLE:
            raise ValueError("Problem is Infeasible!")

        result = x.X

        model.close()

        return result

    def solve_player_payoffs(
        self, payoffs: np.ndarray, player: int, solution: np.ndarray | None = None
    ) -> np.ndarray:
        """
        Solves the KPG from the perspective of one player given the solutions of
        other players using a given set of payoffs.

        Args:
            payoffs (np.ndarray): new payoffs.
            player (int): index of the player.
            solution (np.ndaray | None, optional): external solution. If None, use self.solution. Defaults to None.

        Raises:
            ValueError: if the problem is infeasible.

        Returns:
            np.ndarray: optimal solution to the problem.
        """
        if solution is None:
            solution = self.solution

        model = gp.Model("KPG player payoffs")

        x = model.addMVar((self.m), vtype=GRB.BINARY, name="x")

        model.setObjective(
            x @ payoffs[player]
            + gp.quicksum(
                x * solution[opp] @ self.inter_coefs[player, opp]
                for opp in self.opps[player]
            ),
            GRB.MAXIMIZE,
        )

        model.addConstr(x @ self.weights[player] <= self.capacity[player])

        model.optimize()

        if model.Status == GRB.INFEASIBLE:
            raise ValueError("Problem is Infeasible!")

        result = x.X

        model.close()

        return result

    def obj_value(
        self,
        player: int,
        solution: np.ndarray | None = None,
        player_solution: np.ndarray | None = None,
    ) -> int:
        """
        Calculates the objective value for a player based on a given solution.
        If player_solution is not None, its used for the player.
        If solution is not None, it is used instead of self.solution.

        Args:
            player (int): Index of the player.
            solution (np.ndarray | None, optional): Solution matrix. Defaults to None.
            player_solution (np.ndarray | None, optional): Player solution vector. Defaults to None.

        Returns:
            int: Objective value of player under the given solution.
        """
        if player_solution is not None:
            player_sol = player_solution
        elif solution is not None:
            player_sol = solution[player]
        else:
            player_sol = self.solution[player]

        if solution is not None:
            others_sol = solution
        else:
            others_sol = self.solution

        value = player_sol @ self.payoffs[player]

        for opp in self.opps[player]:
            value += player_sol * others_sol[opp] @ self.inter_coefs[player, opp]

        return value


@dataclass
class KPGResult:
    # Class for storing result of solving KPG instance
    PNE: bool
    X: np.ndarray
    ObjVal: int
    runtime: float


def generate_random_KPG(
    n=2,
    m=25,
    capacity=0.2,
    weight_type="sym",
    payoff_type="sym",
    interaction_type="sym",
) -> KnapsackPackingGame:
    players = list(range(n))

    match weight_type:
        case "sym":
            weight = np.random.randint(1, 101, m)
            weights = np.zeros((n, m))
            for p in players:
                weights[p, :] = weight
        case "asym":
            weights = np.random.randint(1, 101, (n, m))
        case _:
            raise ValueError("Weight type not recognised!")

    match payoff_type:
        case "sym":
            payoff = np.random.randint(1, 101, m)
            payoffs = np.zeros((n, m))
            for p in players:
                payoffs[p, :] = payoff
        case "asym":
            payoffs = np.random.randint(1, 101, (n, m))
        case _:
            raise ValueError("Payoff type not recognised!")

    match interaction_type:
        case "sym":
            coefs = np.random.randint(1, 101, (n, n))
            interaction_coefs = np.zeros((n, n, m))
            for j in range(m):
                interaction_coefs[:, :, j] = coefs
        case "asym":
            interaction_coefs = np.random.randint(1, 101, (n, n, m))
        case "negasym":
            interaction_coefs = np.random.randint(-100, 101, (n, n, m))
        case _:
            raise ValueError("Interaction type not recognised!")

    for p in players:
        interaction_coefs[p, p, :] = 0

    # Interaction cleanup
    for j in range(m):
        for p in range(n):
            interaction_coefs[p, p, j] = 0

    kpg = KnapsackPackingGame(weights, payoffs, interaction_coefs, capacity)

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


def zero_regrets(kpg: KnapsackPackingGame, verbose=False) -> KPGResult:
    """
    Solves kpg using the ZeroRegrets methods of cutting.

    Args:
        kpg (KPG): KPG problem.
        verbose (bool, optional): Print progress?. Defaults to False.

    Returns:
        KPGResult: result of solving the KPG problem.
    """
    # Only for n=2!
    # TODO: extend for higher n.
    start = time.time()

    model = gp.Model("ZeroRegrets KPG")
    x = model.addMVar((kpg.n, kpg.m), vtype=GRB.BINARY, name="x")
    z = model.addMVar((kpg.n, kpg.n, kpg.m), vtype=GRB.BINARY, name="z")

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
            model.addConstr(z[p1, p2, j] <= x[p1, j])
            model.addConstr(z[p1, p2, j] <= x[p2, j])
            model.addConstr(z[p1, p2, j] >= x[p1, j] + x[p2, j] - 1)

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
                if current_x[p, j] == 0:
                    continue
                elif (
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
            obj = kpg.payoffs[p, :] @ current_x[p, :] + sum(
                kpg.inter_coefs[p, p2, :] @ current_z[p, p2, :] for p2 in kpg.opps[p]
            )

            new_x = kpg.solve_player(p, current_x)
            new_obj = kpg.obj_value(p, solution=current_x, player_solution=new_x)
            if new_obj >= obj + eps:
                # Add constraint!
                model.addConstr(
                    kpg.payoffs[p, :] @ new_x
                    + gp.quicksum(
                        kpg.inter_coefs[p, p2, j] * new_x[j] * x[p2, j]
                        for j in range(kpg.m)
                        for p2 in kpg.opps[p]
                    )
                    <= gp.quicksum(kpg.payoffs[p, j] * x[p, j] for j in range(kpg.m))
                    + gp.quicksum(
                        kpg.inter_coefs[p, p2, j] * z[p, p2, j]
                        for j in range(kpg.m)
                        for p2 in kpg.opps[p]
                    )
                )
                cuts += 1
                new_constraint = True

        if not new_constraint:
            break

        model.optimize()

        if model.Status == GRB.INFEASIBLE:
            print("IPG is not feasible (anymore)!")
            break

    runtime = round(time.time() - start, 1)

    if verbose:
        print("====")
        print(f"Added {dominance} dominance constraints.")
        print(f"Added {cuts} cuts.")
        print(f"Added {unequal_payoff} unequal payoff constraints.")

    if model.Status == GRB.INFEASIBLE:
        print(f"Reached INFEASIBLE with {current_obj} in {runtime} seconds")
        return KPGResult(False, current_x, current_obj, runtime)

    if verbose:
        print(f"{model.ObjVal} in {runtime} seconds")

    result = x.X
    objective = model.ObjVal

    model.close()

    return KPGResult(True, result, objective, runtime)


if __name__ == "__main__":
    np.random.seed(42)
    prefix = "instances_kp/generated/"

    for capacity in [2, 5, 8]:
        file = prefix + f"2-25-{capacity}-cij-n.txt"
        print(file)
        instance = read_file(file)
        result = zero_regrets(instance, True)

    # kpg = KPG.generate(m=25, capacity=0.2, weight_type="asym", payoff_type="asym", interaction_type="asym")

    # x, obj, runtime = zero_regrets(kpg)
    # print(runtime)
