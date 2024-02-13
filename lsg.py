from dataclasses import dataclass
import random
from statistics import mean
import numpy as np
import gurobipy as gp
from gurobipy import GRB
@dataclass
class LSG:
    # Class for storing Location Selection Games

    # Sets
    incumbents: list # I
    customer_locs: list # J
    retail_locs: list # K
    potential_locs: dict # K_i \subset K for i \in I
    # retail_strats: dict # i \in I

    # Exogenous parameters
    pop_count: list # (j)
    margin: dict # (i, j)
    costs: dict # (i, k)
    max_dist: int
    distance: dict # (j, k)
    norm_distance: dict # (j, k)
    conveniences: list # (k)
    max_convenience: int
    # observed_locs: list # (i)

    # Endogenous variables
    utility: dict # (i, j, k)
    alpha: int # normalised sensitivity to distance
    betas: list # (i) brand attractiveness
    # delta: list # (i) unilateral imporvement potentials

    def patreons(self, i: int, j: int, x: list) -> float:
        cum_util_i = 0
        for k in self.potential_locs[i]:
            if self.distance[j, k] <= self.max_dist:
                cum_util_i += x[i, k] * self.utility[i, j, k]

        cum_util = 0
        for ii in self.incumbents:
            for k in self.retail_locs:
                cum_util += x[ii, k] * self.utility[ii, j, k]

        return cum_util_i / cum_util


def random_LSG(incumbents: int, customers: int, locations: int, seed: int = 42) -> LSG:
    random.seed(seed)

    # Base parameters
    i = list(range(incumbents))
    j = list(range(customers))
    k = list(range(locations))

    # Potential location assignment
    potential = {ii: [] for ii in i}
    for kk in k:
        potential[random.choice(i)].append(kk)

    population = [random.randint(45, 55) for _ in j]
    margin = {(ii,jj): 1 for ii in i for jj in j}
    costs = {(ii, kk): random.randint(45, 56) for ii in i for kk in k}
    max_dist = 100

    pop_coords = [(random.randint(1, 100), random.randint(1, 100)) for _ in j]
    store_coords = [(random.randint(1, 100), random.randint(1, 100)) for _ in k]

    # Distance between stores and customers
    distance = {}
    norm_distance = {}
    for jj, pop in enumerate(pop_coords):
        for kk, store in enumerate(store_coords):
            # Manhattan distance
            distance[jj, kk] = abs(pop[0] - store[0]) + abs(pop[1] - store[1])
            norm_distance[jj, kk] = (max_dist - distance[jj, kk]) / max_dist

    competition_coords = [(random.randint(1, 100), random.randint(1, 100)) for _ in range(50)]
    conveniences = []
    for kk, store in enumerate(store_coords):
        comp_dists = []
        for comp in competition_coords:
            comp_dists.append(abs(store[0] - comp[0]) + abs(store[1] - comp[1]))
        conveniences.append(mean(comp_dists))

    max_convenience = max(conveniences)

    for cc, conv in enumerate(conveniences):
        conveniences[cc] = (max_convenience - conv) / max_convenience

    alpha = 0.4
    betas = [random.random() for _ in i]

    utility = {(ii, jj, kk): betas[ii] + alpha * distance[jj, kk] + (1 - alpha) * conveniences[kk]
                   for ii in i for jj in j for kk in k}

    return LSG(i, j, k, potential, population, margin, costs, max_dist, distance,
               norm_distance, conveniences, max_convenience, utility, alpha, betas)

example = random_LSG(2, 10, 10)

def solve_LSG(problem: LSG) -> np.ndarray:
    x = np.zeros((len(problem.incumbents), len(problem.retail_locs)))
    solutions = []
    while duplicate_array(solutions, x):
        for player in problem.incumbents:
            x = solve_partial_LSG(problem, x, player)
            solutions.append(x.copy())
    return x

def solve_partial_LSG(problem: LSG, x_hat: np.ndarray, i: int) -> np.ndarray:
    rivals = [ii for ii in problem.incumbents if i != ii]
    potential = problem.potential_locs[i]
    bound_m = [min(problem.utility[i, j, k] for k in potential) / 1
               for j in problem.customer_locs]
    viable = {j: [k for k in problem.retail_locs if problem.distance[j, k] < problem.max_dist]
              for j in problem.customer_locs}

    env = gp.Env()
    env.setParam("OutputFlag", 0)
    model = gp.Model("PartialLSG", env=env)
    x = model.addMVar(len(problem.retail_locs), vtype=GRB.BINARY, name="x")
    y = model.addMVar((len(problem.retail_locs), len(problem.retail_locs)), name="y")
    f = model.addMVar(len(problem.customer_locs), lb=0, ub=1, name="f")

    model.setObjective(gp.quicksum(problem.margin[i,j] * problem.pop_count[j] * f[j]
                                   for j in problem.customer_locs) -
                       gp.quicksum(problem.costs[i,k] * x[k]
                                   for k in potential),
                       GRB.MAXIMIZE)

    # y[j,k] = f[j] * x[k]
    for j in problem.customer_locs:
        for k in potential:
            model.addConstr(y[j,k] <= x[k])
            model.addConstr(y[j,k] <= f[j])
            model.addConstr(y[j,k] >= f[k] - (1 - x[k]))

    for j in problem.customer_locs:
        model.addConstr(f[j] <= bound_m[j] * gp.quicksum(
            x[k] * problem.utility[i, j, k]
            for k in viable[j]
        ))

        model.addConstr(gp.quicksum(y[j,k] * problem.utility[i, j, k] for k in viable[j]) +
                        gp.quicksum(f[j] * x_hat[ii, k] * problem.utility[ii, j, k]
                                    for ii in rivals for k in viable[j]) <=
                        gp.quicksum(x[k] * problem.utility[i, j, k] for k in viable[j]))

    model.optimize()

    if model.Status == GRB.INFEASIBLE:
        print("Model is not feasible (anymore)!")
        return x_hat

    x_hat[i, :] = x.X
    return x_hat

def duplicate_array(solutions: list, x: np.ndarray) -> bool:
    for solution in solutions:
        if np.array_equal(solution, x):
            return True
    return False

print(example)
print(solve_LSG(example))
