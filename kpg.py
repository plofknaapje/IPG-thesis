import numpy as np
import gurobipy as gp
from gurobipy import GRB
from dataclasses import dataclass
import time
import itertools
import utils


@dataclass
class KPG:
    # Class for storing binary Knapsack Packing Games.
    n: int  # number of players
    m: int  # number of items
    players: list  # list of player indices, length n
    capacity: list  # carrying capacity of each player, length n
    weights: np.ndarray  # weights of the items, size (n, m)
    payoffs: np.ndarray  # payoffs of the items, size (n, m)
    # interaction payoff of the items (n, n, m) with 0 on diagonals
    inter_coefs: np.ndarray
    solution: np.ndarray|None

    def __init__(self, weights: np.ndarray, payoffs: np.ndarray, inter_coefs: np.ndarray, capacity: float|list):
        self.n, self.m = weights.shape
        self.weights = weights
        self.payoffs = payoffs
        self.inter_coefs = inter_coefs
        self.players = list(range(self.n))
        self.pairs = list(itertools.permutations(self.players, 2))

        if type(capacity) is float:
            self.capacity = [
                self.weights[p, :].sum() * capacity for p in self.players]
        elif type(capacity) is list:
            self.capacity = capacity
        else:
            raise TypeError(
                f"{type(capacity)} is not a valid type for capacity.")
        self.solution = None

    def print_data(self):
        print("Payoffs")
        print(self.payoffs)
        print("Weights")
        print(self.weights)
        print("Interaction coefficients")
        print(self.inter_coefs)

    def solve_player_weights(self, weights: np.ndarray, p: int) -> np.ndarray:
        pairs = [pair for pair in self.pairs if pair[0] == p]

        model = gp.Model("Knapsack Problem")

        x = model.addMVar((self.m), vtype=GRB.BINARY, name="x")

        model.setObjective(x @ self.payoffs[p, :] +
                           gp.quicksum(x * self.solution[opp, :] @ self.inter_coefs[p, opp, :]
                                       for _, opp in pairs),
                           GRB.MAXIMIZE)

        model.addConstr(x @ weights[p, :] <= self.capacity[p])

        model.optimize()

        if model.Status == GRB.INFEASIBLE:
            raise ValueError("Problem is Infeasible!")

        return x.X

    def solve_player_payoffs(self, payoffs: np.ndarray, p: int) -> np.ndarray:
        pairs = [pair for pair in self.pairs if pair[0] == p]

        model = gp.Model("Knapsack Problem")

        x = model.addMVar((self.m), vtype=GRB.BINARY, name="x")

        model.setObjective(x @ payoffs[p, :] +
                           gp.quicksum(x * self.solution[opp, :] @ self.inter_coefs[p, opp, :]
                                       for _, opp in pairs),
                           GRB.MAXIMIZE)

        model.addConstr(x @ self.weights <= self.capacity[p])

        model.optimize()

        if model.Status == GRB.INFEASIBLE:
            raise ValueError("Problem is Infeasible!")

        return x.X

    def solve(self, verbose=False) -> np.ndarray:
        if self.solution is not None:
            print("Already solved")
            return self.solution

        result = zero_regrets(self, verbose)
        self.solution = result.X
        self.PNE = result.PNE
        return self.solution

    def obj_value(self, p: int, solution: np.ndarray) -> int|float:
        pairs = [pair for pair in self.pairs if pair[0] == p]

        value = solution[p, :] @ self.payoffs[p, :]

        for _, opp in pairs:
            value += solution[p, :] * solution[opp, :] @ self.inter_coefs[p, opp]

        return value

    def solved_obj_value(self, p: int, player_sol=None) -> int|float:
        pairs = [pair for pair in self.pairs if pair[0] == p]

        if player_sol is None:
            value = self.solution[p, :] @ self.payoffs[p, :]

            for _, opp in pairs:
                value += self.solution[p, :] * self.solution[opp, :] @ self.inter_coefs[p, opp, :]
        else:
            value = player_sol @ self.payoffs[p, :]

            for _, opp in pairs:
                value += player_sol * self.solution[opp, :] @ self.inter_coefs[p, opp, :]

        return value


@dataclass
class KPGResult:
    # Class for storing result of solving KPG instance
    PNE: bool
    X: np.ndarray
    ObjVal: int
    runtime: float

def generate_random_KPG(n=2, m=25, capacity=0.2, weight_type="sym", payoff_type="sym", interaction_type="sym") -> KPG:
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

    kpg = KPG(weights, payoffs, interaction_coefs, capacity)

    return kpg

def read_file(file: str) -> KPG:
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
        coef_list = line[2*n:]
        index = 0
        for p1 in range(n):
            for p2 in range(n):
                if p1 == p2:
                    continue
                interaction_coefs[p1, p2, j] = coef_list[index]
                index += 1

    kpg = KPG(weights, payoffs, interaction_coefs, capacity)

    return kpg


def create_player_oracle(kpg: KPG, player: int) -> tuple[gp.Model, gp.MVar, gp.MVar]:

    m = gp.Model(f"LocalKPG[{player}]")
    x = m.addMVar((kpg.n, kpg.m), vtype=GRB.BINARY, name="x")
    z = m.addMVar((kpg.n, kpg.n, kpg.m), vtype=GRB.BINARY, name="z")

    m.setObjective(kpg.payoffs[player, :] @ x[player, :] +
                   gp.quicksum(kpg.inter_coefs[p1, p2, :] @ z[p1, p2, :]
                               for p1, p2 in kpg.pairs if p1 == player),
                   GRB.MAXIMIZE)

    for p in kpg.players:
        # Capacity constraint
        m.addConstr(kpg.weights[p, :] @ x[p, :] <= kpg.capacity[p])

    for p1, p2 in kpg.pairs:
        for j in range(kpg.m):
            if p1 > p2:
                # z value symmetry
                m.addConstr(z[p1, p2, j] == z[p2, p1, j])
                continue
            # z value constraints
            m.addConstr(z[p1, p2, j] <= x[p1, j])
            m.addConstr(z[p1, p2, j] <= x[p2, j])
            m.addConstr(z[p1, p2, j] >= x[p1, j] + x[p2, j] - 1)

    return (m, x, z)


def oracle_optimization(oracle: tuple, kpg: KPG, point_x: np.ndarray, p: int, verbose=False) -> tuple[np.ndarray, int]:
    """ Detect local suboptimal solutions
    Function checks if a solution is suboptimal for player p and returns a local improvement if possible.

    Args:
        kpg (KPG): KPG problem.
        point_x (np.ndarray): point to check.
        p (int): index of player to check.
        verbose (bool, optional): enable verbose output. Defaults to False.

    Raises:
        ValueError: If problem is infeasible, but then there is a problem higher up!

    Returns:
        tuple[np.ndarray, int]: new matrix x as well as new player objective.
    """
    m, x, _ = oracle
    for o in kpg.players:
        # Fix actions of other players
        if o != p:
            for j in range(kpg.m):
                x[o, j].lb = point_x[o, j]
                x[o, j].ub = point_x[o, j]

    m.optimize()

    if verbose:
        for v in m.getVars():
            print(f"{v.VarName} {v.X:g}")

        print(f"Obj: {m.ObjVal:g}")

    if m.Status == GRB.INFEASIBLE:
        raise ValueError("Problem is Infeasible! This is not possible!")

    return x.X, m.ObjVal


def zero_regrets(kpg: KPG, verbose=False, eps=1e-7) -> KPGResult:
    """Optimises kpg using the ZeroRegrets methods of cutting.

    Args:
        kpg (KPG): KPG problem.
        verbose (bool, optional): Print progress?. Defaults to False.

    Returns:
        KPGResult: result of solving the KPG problem.
    """
    # Only for n=2!
    # TODO: extend for higher n.
    start = time.time()

    oracles = {p: create_player_oracle(kpg, p) for p in kpg.players}

    pm = gp.Model("ZeroRegrets")
    x = pm.addMVar((kpg.n, kpg.m), vtype=GRB.BINARY, name="x")
    z = pm.addMVar((kpg.n, kpg.n, kpg.m), vtype=GRB.BINARY, name="z")

    pm.setObjective(gp.quicksum(x[p, :] @ kpg.payoffs[p, :] for p in kpg.players) +
                    gp.quicksum(kpg.inter_coefs[p1, p2, :] @ z[p1, p2, :]
                                for p1, p2 in kpg.pairs),
                    GRB.MAXIMIZE)

    for p in kpg.players:
        # Capacity constraint
        pm.addConstr(kpg.weights[p, :] @ x[p, :] <= kpg.capacity[p])

    for p1, p2 in kpg.pairs:
        for j in range(kpg.m):
            if p1 > p2:
                pm.addConstr(z[p1, p2, j] == z[p2, p1, j])
                continue
            pm.addConstr(z[p1, p2, j] <= x[p1, j])
            pm.addConstr(z[p1, p2, j] <= x[p2, j])
            pm.addConstr(z[p1, p2, j] >= x[p1, j] + x[p2, j] - 1)

    # If the highest possible value of an item k is lower than the lowest possible
    # value of an item j, then item j dominates item k.
    dominance = 0
    for p in kpg.players:
        opponent_sets = [s for s in utils.powerset(kpg.players) if p not in s]
        for j in range(kpg.m):
            j_min = min(kpg.payoffs[p, j] + sum(kpg.inter_coefs[p, o, j] for o in ops)
                        for ops in opponent_sets)
            for k in range(kpg.m):
                if k == j or kpg.weights[p, j] > kpg.weights[p, k]:
                    continue
                k_max = max(kpg.payoffs[p, k] + sum(kpg.inter_coefs[p, o, k] for o in ops)
                            for ops in opponent_sets)
                if j_min >= k_max + eps:
                    pm.addConstr(x[p, k] <= x[p, j])
                    dominance += 1
    if verbose:
        print("====")
    cuts = 0
    unequal_payoff = 0
    while True:
        pm.optimize()

        if pm.Status == GRB.INFEASIBLE:
            print("IPG is not feasible (anymore)!")
            break

        current_x = x.X
        current_z = z.X
        current_obj = pm.ObjVal
        finished = True
        if verbose:
            print(f"Current total is {current_obj}")

        # Check if a player has net-negative variables and exclude the solutions with them.
        for p in kpg.players:
            for j in range(kpg.m):
                if current_x[p, j] == 0: # type: ignore
                    continue
                elif kpg.payoffs[p, j] * current_x[p, j] + \
                    sum(kpg.inter_coefs[p, o, j] * current_z[p, o, j]
                        for o in range(kpg.n) if o != p) >= 0:
                    continue
                opponent_set = [o for o in range(
                    kpg.n) if o != p and kpg.inter_coefs[p, o, j] < 0]
                pm.addConstr(x[p, j] + gp.quicksum(x[o, j]
                                for o in opponent_set) <= len(opponent_set))
                unequal_payoff += 1

        # Add cuts to the problem for each player which has a better solution.
        for p in kpg.players:
            obj = kpg.payoffs[p, :] @ current_x[p, :] + \
                sum(kpg.inter_coefs[p1, p2, :] @ current_z[p1, p2, :]
                    for p1, p2 in kpg.pairs if p1 == p)

            new_x, new_obj = oracle_optimization(oracles[p], kpg, current_x, p)
            if new_obj >= obj + eps:
                # Add constraint!
                pm.addConstr(kpg.payoffs[p, :] @ new_x[p, :] +
                             gp.quicksum(kpg.inter_coefs[p1, p2, j] * new_x[p1, j] * x[p2, j]
                                         for j in range(kpg.m) for p1, p2 in kpg.pairs if p1 == p) <=
                             gp.quicksum(kpg.payoffs[p, j] * x[p, j] for j in range(kpg.m)) +
                             gp.quicksum(kpg.inter_coefs[p1, p2, j] * z[p1, p2, j]
                                         for j in range(kpg.m) for p1, p2 in kpg.pairs if p1 == p))
                cuts += 1
                finished = False

        if finished:
            break

    runtime = round(time.time() - start, 1)

    if verbose:
        print("====")
        print(f"Added {dominance} dominance constraints.")
        print(f"Added {cuts} cuts.")
        print(f"Added {unequal_payoff} unequal payoff constraints.")

    if pm.Status == GRB.INFEASIBLE:
        print(f"Reached INFEASIBLE with {current_obj} in {runtime} seconds")
        return KPGResult(False, current_x, current_obj, runtime)

    if verbose:
        print(f"{pm.ObjVal} in {runtime} seconds")

    return KPGResult(True, x.X, pm.ObjVal, runtime)


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
