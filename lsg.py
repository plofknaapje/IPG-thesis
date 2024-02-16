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
    customers: list # J
    locations: list # K
    potential_locs: dict # K_i \subset K for i \in I
    # retail_strats: dict # i \in I

    # Exogenous parameters
    pop_count: list # (j)
    margin: np.ndarray # (i, j)
    costs: np.ndarray # (i, k)
    max_dist: int
    distance: np.ndarray # (j, k)
    norm_distance: np.ndarray # (j, k)
    conveniences: list # (k)
    max_convenience: int
    # observed_locs: list # (i)

    # Endogenous variables
    utility: np.ndarray # (i, j, k)
    alpha: float # normalised sensitivity to distance
    betas: list # (i) brand attractiveness
    # delta: list # (i) unilateral imporvement potentials

    def update_alpha_beta(self, alpha: float, betas: list):
        self.alpha = alpha
        self.betas = betas
        self.utility = np.array([[[betas[i] + alpha * self.distance[j, k] + (1 - alpha) * self.conveniences[k] 
                                   for k in self.locations] 
                                   for j in self.customers] 
                                   for i in self.incumbents])


def random_LSG(incumbents: int, customers: int, locations: int, alpha: float=0.4,
               betas=None, beta_range: int = 1, seed: int = 42) -> LSG:
    random.seed(seed)
    np.random.seed(seed)

    # Base parameters
    i = list(range(incumbents))
    j = list(range(customers))
    k = list(range(locations))

    # Potential location assignment
    potential = {ii: [] for ii in i}
    for kk in k:
        potential[random.choice(i)].append(kk)

    population = [random.randint(45, 55) for _ in j]
    margin = np.ones((len(i), len(j))) 
    costs = np.random.randint(45, 56, (len(i), len(k)))
    max_dist = 100

    pop_coords = [(random.randint(1, 100), random.randint(1, 100)) for _ in j]
    store_coords = [(random.randint(1, 100), random.randint(1, 100)) for _ in k]

    # Distance between stores and customers
    distance = np.zeros((len(j), len(k)))
    norm_distance = np.zeros((len(j), len(k)))
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

    # alpha = 0.4
    if betas is None:
        betas = [random.random() for _ in i]
        betas = [beta/sum(betas)*beta_range for beta in betas]

    utility = np.array([[[betas[ii] + alpha * distance[jj, kk] + (1 - alpha) * conveniences[kk] for kk in k]
                for jj in j] for ii in i])

    return LSG(i, j, k, potential, population, margin, costs, max_dist, distance,
               norm_distance, conveniences, max_convenience, utility, alpha, betas)


def solve_LSG(problem: LSG, verbose=False) -> np.ndarray:
    x = np.zeros((len(problem.incumbents), len(problem.locations)))
    solutions = []
    while not duplicate_array(solutions, x):
        for player in problem.incumbents:
            solutions.append(x.copy())
            x = solve_partial_LSG(problem, x, player, verbose)
    return x

def solve_partial_LSG(problem: LSG, x_hat: np.ndarray, i: int, verbose=False) -> np.ndarray:
    rivals = [ii for ii in problem.incumbents if i != ii]
    bound_m = [min(problem.utility[i, j, k] for k in problem.locations) / 1
               for j in problem.customers]
    viable = {j: [k for k in problem.locations if problem.distance[j, k] < problem.max_dist]
              for j in problem.customers}

    model = gp.Model("PartialLSG")
    x = model.addMVar(len(problem.locations), vtype=GRB.BINARY, name="x")
    for location in problem.locations:
        if location not in problem.potential_locs[i]:
            x[location].ub = 0
    y = model.addMVar((len(problem.locations), len(problem.locations)), lb=0, ub=1, name="y")
    f = model.addMVar(len(problem.customers), lb=0, ub=1, name="f")

    model.setObjective(gp.quicksum(problem.margin[i,j] * problem.pop_count[j] * f[j]
                                   for j in problem.customers) -
                       gp.quicksum(problem.costs[i,k] * x[k]
                                   for k in problem.locations),
                       GRB.MAXIMIZE)

    # y[j,k] = f[j] * x[k]
    for j in problem.customers:
        for k in problem.locations:
            model.addConstr(y[j,k] <= x[k])
            model.addConstr(y[j,k] <= f[j])
            model.addConstr(y[j,k] >= f[k] - (1 - x[k]))

    for j in problem.customers:
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
    if verbose:
        print(y.X)
        print(f.X)

    x_hat[i, :] = x.X
    return x_hat

def duplicate_array(solutions: list, x: np.ndarray) -> bool:
    for solution in solutions:
        if np.array_equal(solution, x):
            return True
    return False


if __name__ == "__main__":

    example = random_LSG(2, 10, 10)

    for i in range(1):
        sample = random_LSG(2, 10, 10, seed=i)
        solve_LSG(sample)