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

    problems = []

    weights = rng.integers(1, r+1, (n, m))

    if type(capacity) is float:
        capacity = [capacity * weights[i].sum() for i in range(n)]
    else:
        capacity = [cap * weights[i].sum() for i, cap in enumerate(capacity)]

    if corr:
        lower = np.maximum(weights - r/5, 1)
        upper = np.minimum(weights + r/5, r+1)


    while len(problems) < size:

        if corr:
            payoffs = rng.integers(lower, upper)
        else:
            payoffs = rng.integers(1, r+1, (n, m))

        interactions = rng.integers(1, int(r/inter_factor) + 1, (n, n, m))
        mask = rng.integers(0, 2, (n, n, m))
        interactions = interactions * mask

        for i in range(n):
            interactions[i, i, :] = 0

        problem = KPG(weights, payoffs, interactions, capacity)
        problem.solve(verbose)
        if problem.PNE:
            problems.append(problem)

    return problems


def generate_payoff_problems(size: int, n: int, m: int, r :int, capacity: float|list,
                             corr=True, inter_factor=3, rng=None, verbose=False) -> list[KPG]:
    if rng is None:
        rng = np.random.default_rng()

    

    problems = []

    payoffs = rng.integers(1, r+1, (n, m))

    if corr:
        lower = np.maximum(payoffs - r/5, 1)
        upper = np.minimum(payoffs + r/5, r+1)

    if type(capacity) is float:
        capacity = [capacity for _ in range(n)]

    while len(problems) < size:

        if corr:
            weights = rng.integers(lower, upper)
        else:
            weights = rng.integers(1, r+1, (n, m))

        interactions = rng.integers(0, int(r/inter_factor) + 1, (n, n, m))
        mask = rng.integers(0, 2, (n, n, m))
        interactions = interactions * mask

        for i in range(n):
            interactions[i, i, :] = 0

        problem_capacity = [cap * weights[i].sum() for i, cap in enumerate(capacity)]

        problem = KPG(weights, payoffs, interactions, problem_capacity)
        problem.solve(verbose)
        if problem.PNE:
            problems.append(problem)

    return problems


def inverse_weights(problems: list[KPG]) -> np.ndarray:
    n = problems[0].n
    players = problems[0].players
    m = problems[0].m
    true_value = {(i, player): problem.solved_obj_value(player)
                  for i, problem in enumerate(problems) for player in players}

    model = gp.Model("Inverse KPG (Weights)")

    w = model.addMVar((n, m), vtype=GRB.INTEGER, name="w")

    model.setObjective(w.sum())

    model.addConstrs(w[j].sum() >= problems[0].weights[j].sum() for j in players)


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

                if new_value >= true_value[i, j] + 0.1:
                    model.addConstr(new_solution @ w[j] >= problem.capacity[j] + 0.1)
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
    p = model.addMVar((n, m), vtype=GRB.INTEGER, name="p")

    model.setObjective(delta.sum())

    model.addConstrs(p[j].sum() <= problems[0].payoffs[j].sum() for j in players)

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
                    sum(new_solution * problem.solution[o] @ problem.inter_coefs[j, o]
                                for o in problem.opps[j])
                model.addConstr(delta[i, j] >= new_value - true_values[i, j])
                new_constraint = True
                solutions[i, j].add(tuple(new_solution))

                temp_new_value = new_solution @ p.X[j] + \
                    sum(new_solution * problem.solution[o] @ problem.inter_coefs[j, o]
                                for o in problem.opps[j])
                
                temp_true_value = problem.solution[j] @ p.X[j] + \
                    sum(problem.solution[j] * problem.solution[o] @ problem.inter_coefs[j, o]
                                for o in problem.opps[j])

                if temp_new_value >= temp_true_value + 0.1:
                    model.addConstr(new_value <= true_values[i, j])

        if not new_constraint:
            break

        model.optimize()

        if model.Status == GRB.INFEASIBLE:
            raise ValueError("Problem is Infeasible!")
    
    return p.X


if __name__ == "__main__":
    start = time()
    rng = np.random.default_rng(1)

    approach = "payoff"

    match approach:
        case "weight":
            weight_problems = generate_weight_problems(100, 2, 30, 100, [0.5, 0.5], corr=True, rng=rng)
            print("Problem generation finished")

            values = weight_problems[0].weights

            inverse = inverse_weights(weight_problems)

        case "payoff":
            payoff_problems = generate_payoff_problems(100, 2, 15, 100, [0.5, 0.5], corr=True, rng=rng)

            print("Problem generation finished")

            values = payoff_problems[0].payoffs

            inverse = inverse_payoffs(payoff_problems)

    print(values)
    print(inverse)

    error = np.abs(inverse - values).sum()
    print(error, error / values.sum())
    print(time() - start)