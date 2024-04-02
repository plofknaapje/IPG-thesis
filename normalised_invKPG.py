import gurobipy as gp
from gurobipy import GRB
from kpg import KPG
import numpy as np
from time import time
from normalised_invKP import inverse_eps_opt

def generate_weight_problems(size: int, n: int, m: int, r :int, capacity: float|list,
                             corr=True, inter_factor=3, rng=None, verbose=False) -> list[KPG]:
    if rng is None:
        rng = np.random.default_rng()

    if type(capacity) is float:
        capacity = [capacity for _ in range(n)]

    problems = []

    weights = rng.integers(1, r+1, (n, m))
    norm_weights = weights / weights.sum(axis=1)[:, np.newaxis]

    if corr:
        lower = np.maximum(weights - r/5, 1)
        upper = np.minimum(weights + r/5, r+1)

    while len(problems) < size:

        if corr:
            payoffs = rng.integers(lower, upper)
        else:
            payoffs = rng.integers(1, r+1, (n, m))

        norm_payoffs = payoffs / payoffs.sum(axis=1)[:, np.newaxis]
        interactions = rng.integers(0, 5, (n, n, m))
        mask = rng.integers(0, 2, (n, n, m))
        interactions = interactions * mask

        norm_interactions = interactions / np.maximum(np.abs(interactions).sum(axis=2)[:, :, np.newaxis], 1)
        for i in range(n):
            norm_interactions[i, i, :] = 0

        problem = KPG(norm_weights, norm_payoffs, norm_interactions / inter_factor, capacity)
        problem.solve(verbose)
        if problem.PNE:
            problems.append(problem)

    return problems


def generate_payoff_problems(size: int, n: int, m: int, r :int, capacity: float|list,
                             corr=True, inter_factor=3, rng=None, verbose=False) -> list[KPG]:
    if rng is None:
        rng = np.random.default_rng()

    if type(capacity) is float:
        capacity = [capacity for _ in range(n)]

    problems = []

    payoffs = rng.integers(1, r+1, (n, m))
    norm_payoffs = payoffs / payoffs.sum(axis=1)[:, np.newaxis]

    if corr:
        lower = np.maximum(payoffs - r/5, 1)
        upper = np.minimum(payoffs + r/5, r+1)

    while len(problems) < size:

        if corr:
            weights = rng.integers(lower, upper)
        else:
            weights = rng.integers(1, r+1, (n, m))

        norm_weights = weights / weights.sum(axis=1)[:, np.newaxis]
        interactions = rng.integers(0, 5, (n, n, m))
        mask = rng.integers(0, 2, (n, n, m))
        interactions = interactions * mask

        norm_interactions = interactions / np.maximum(np.abs(interactions).sum(axis=2)[:, :, np.newaxis], 1)
        for i in range(n):
            norm_interactions[i, i, :] = 0

        problem = KPG(norm_weights, norm_payoffs, norm_interactions / inter_factor, capacity)
        problem.solve(verbose)
        if problem.PNE:
            problems.append(problem)

    return problems


def inverse_weights(problems: list[KPG], eps=1e-3) -> np.ndarray:
    n = problems[0].n
    players = problems[0].players
    m = problems[0].m
    true_value = {(i, player): problem.solved_obj_value(player)
                  for i, problem in enumerate(problems) for player in players}

    model = gp.Model("Inverse KPG (Weights)")

    w = model.addMVar((n, m), name="w")

    model.setObjective(w.sum())

    model.addConstrs(w[j].sum() >= 1 for j in players)


    for problem in problems:
        for j in players:
            model.addConstr(problem.solution[j] @ w[j] <= problem.capacity[j])

    model.optimize()

    if model.Status == GRB.INFEASIBLE:
        raise ValueError("Problem is Infeasible!")

    while True:
        new_constraint = False

        for i, problem in enumerate(problems):

            for j in players:
                new_solution = problem.solve_player_weights(w.X, j)
                new_value = problem.solved_obj_value(j, new_solution)

                if new_value >= true_value[i, j] + eps:
                    model.addConstr(new_solution @ w[j] >= problem.capacity[j] + eps)
                    new_constraint = True

        if not new_constraint:
            break

        model.optimize()

        if model.Status == GRB.INFEASIBLE:
            raise ValueError("Problem is Infeasible!")

    return w.X


def inverse_payoffs(problems: list[KPG]) -> np.ndarray:
    # Uses the delta method
    n_problems = len(problems)
    n = problems[0].n
    players = problems[0].players
    m = problems[0].m
    
    model = gp.Model("Inverse KPG (Weights)")

    delta = model.addMVar((n_problems, n), name="delta")
    p = model.addMVar((n, m), name="p")

    model.setObjective(delta.sum())

    model.addConstrs(p[j].sum() == 1 for j in players)

    true_values = {(i, j): problem.solution[j] @ p[j] + 
                   sum(problem.solution[j] * problem.solution[o] @ problem.inter_coefs[j, o]
                               for o in problem.opps[j])
                   for i, problem in enumerate(problems) for j in players}

    model.optimize()

    if model.Status == GRB.INFEASIBLE:
        raise ValueError("Problem is Infeasible!")
    
    solutions = {(i, j): set() for i in range(n_problems) for j in players}
    
    while True:
        new_constraint = False

        for i, problem in enumerate(problems):
            for j in players:
                new_solution = problem.solve_player_payoffs(p.X, j)
                if tuple(new_solution) in solutions[i, j]:
                    continue
                new_value = new_solution @ p[j] + \
                    gp.quicksum(new_solution * problem.solution[o] @ problem.inter_coefs[j, o]
                                for o in problem.opps[j])
                model.addConstr(delta[i, j] >= new_value - true_values[i, j])
                new_constraint = True
                solutions[i, j].add(tuple(new_solution))

        if not new_constraint:
            break

        model.optimize()

        if model.Status == GRB.INFEASIBLE:
            raise ValueError("Problem is Infeasible!")
    
    return p.X


if __name__ == "__main__":
    start = time()
    rng = np.random.default_rng(1)

    pows = [2, 2.5, 3, 3.5, 4, 4.5, 5]

    approach = "payoff"

    match approach:
        case "weight":
            weight_problems = generate_weight_problems(50, 2, 20, 100, [0.5, 0.5], corr=True, rng=rng)
            print("Problem generation finished")

            values = weight_problems[0].weights

            inverse, eps = inverse_eps_opt(inverse_weights, weight_problems, values, pows)

        case "payoff":
            payoff_problems = generate_payoff_problems(100, 2, 10, 100, [0.5, 0.5], corr=False, rng=rng)

            print("Problem generation finished")

            values = payoff_problems[0].payoffs

            inverse = inverse_payoffs(payoff_problems)

    print(values)
    print(inverse)

    error = np.abs(inverse - values).sum()
    print(error)
    print(time() - start)