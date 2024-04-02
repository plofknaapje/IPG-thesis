import enum
import gurobipy as gp
from gurobipy import GRB
import numpy as np
from time import time
from kp import KP


def generate_weight_problems(size:int=50, m:int=10, r:int=100, capacity:float|list=0.5, corr=True, rng=None) -> list[KP]:
    if rng is None:
        rng = np.random.default_rng()

    problems = []
    weights = rng.integers(1, r+1, m)
    norm_weights = weights / weights.sum()

    if type(capacity) is float:
        capacity = [capacity for _ in range(size)]

    if corr:
        lower = np.maximum(weights - r / 10, 1)
        upper = np.minimum(weights + r / 10, r+1)

    for i in range(size):
        if corr:
            payoffs = rng.integers(lower, upper)
        else:
            payoffs = rng.integers(1, r+1, m)
        norm_payoffs = payoffs / payoffs.sum()
        problem = KP(norm_payoffs, norm_weights, capacity[i])
        problem.solve()
        problems.append(problem)

    return problems


def generate_payoff_problems(size:int, m:int=10, r:int=100, capacity:float|list=0.5, corr=True, rng=None) -> list[KP]:
    if rng is None:
        rng = np.random.default_rng()

    problems = []
    payoffs = rng.integers(1, r+1, m)
    norm_payoffs = payoffs / payoffs.sum()

    if type(capacity) is float:
        capacity = [capacity for _ in range(size)]

    if corr:
        lower = np.maximum(payoffs - r / 10, 1)
        upper = np.minimum(payoffs + r / 10, r+1)

    for i in range(size):
        if corr:
            weights = rng.integers(lower, upper)
        else:
            weights = rng.integers(1, r+1, m)
        norm_weights = weights / weights.sum()
        problem = KP(norm_payoffs, norm_weights, capacity[i])
        problem.solve()
        problems.append(problem)

    return problems


def inverse_weights(problems: list[KP], eps=1e-4) -> tuple[np.ndarray, int]:
    items = problems[0].n_items
    true_value = [problem.solution @ problem.payoffs for problem in problems]

    model = gp.Model("Inverse Knapsack (Weights)")
    model.Params.PoolSearchMode = 2
    model.Params.PoolSolutions = 10

    w = model.addMVar((items), name="w")

    model.setObjective(w.sum())

    model.addConstr(w.sum() >= 1)

    for problem in problems:
        model.addConstr(problem.solution @ w <= problem.capacity)

    model.optimize()

    if model.Status == GRB.INFEASIBLE:
        raise ValueError("Problem is Infeasible!")

    while True:
        new_constraint = False

        for i, problem in enumerate(problems):
            new_solution = problem.solve_weights(w.X)
            new_value = new_solution @ problem.payoffs

            if new_value >= true_value[i] + eps:
                model.addConstr(new_solution @ w >= problem.capacity + eps)
                new_constraint = True

        if not new_constraint:
            break

        model.optimize()

        if model.Status == GRB.INFEASIBLE:
            raise ValueError("Problem is Infeasible!")

    return w.X, model.SolCount


def inverse_payoffs_direct(problems: list[KP], eps = 1e-3) -> tuple[np.ndarray, int]:
    n_items = problems[0].n_items

    model = gp.Model("Inverse Knapsack (Payoffs)")
    model.Params.PoolSearchMode = 2
    model.Params.PoolSolutions = 10

    p = model.addMVar((n_items), name="p")

    model.setObjective(gp.quicksum(p @ problem.solution for problem in problems), GRB.MAXIMIZE)

    model.addConstr(p.sum() <= 1)

    model.optimize()

    if model.Status == GRB.INFEASIBLE:
        raise ValueError("Problem is Infeasible!")

    while True:
        new_constraint = False

        for _, problem in enumerate(problems):
            new_solution = problem.solve_payoffs(p.X)
            new_value = new_solution @ p.X
            true_value = problem.solution @ p.X

            if new_value >= true_value + eps:
                model.addConstr(problem.solution @ p >= new_solution @ p)
                new_constraint = True

        if not new_constraint:
            break

        model.optimize()

        if model.Status == GRB.INFEASIBLE:
            raise ValueError("Problem is Infeasible!")

    return p.X, model.SolCount


def inverse_payoffs_delta(problems: list[KP]) -> tuple[np.ndarray, int]:
    n_items = problems[0].n_items
    n_problems = len(problems)

    model = gp.Model("Inverse Knapsack (Payoffs)")
    model.Params.PoolSearchMode = 2
    model.Params.PoolSolutions = 10

    delta = model.addMVar((n_problems), name="delta")
    p = model.addMVar((n_items), name="p")

    model.setObjective(delta.sum())

    model.addConstr(p.sum() == 1)

    model.optimize()

    if model.Status == GRB.INFEASIBLE:
        raise ValueError("Problem is Infeasible!")

    solutions = [set() for _ in problems]

    while True:
        new_constraint = False

        for i, problem in enumerate(problems):
            new_solution = problem.solve_payoffs(p.X)
            if tuple(new_solution) in solutions[i]:
                continue
            model.addConstr(delta[i] >= new_solution @ p - problem.solution @ p)
            new_constraint = True
            solutions[i].add(tuple(new_solution))

        if not new_constraint:
            break

        model.optimize()

        if model.Status == GRB.INFEASIBLE:
            raise ValueError("Problem is Infeasible!")

    return p.X, model.SolCount


def inverse_eps_opt(inverse_function, problems: list, truth: np.ndarray, pows: list|np.ndarray, complete=False):

    for pow in pows:
        eps = 10 ** (-pow)
        try:
            output = inverse_function(problems, eps)
            if type(output) is tuple:
                values = output[0]
            else:
                values = output
            print(f"Problem solved with eps={eps}")
            if not complete:
                print(np.abs(values - truth).sum())
                return values, eps
            else:
                print(np.abs(values - truth).sum())
        except ValueError:
            print(f"Problem not possible with eps={eps}")
    return values, eps

if __name__ == "__main__":

    start = time()
    rng = np.random.default_rng(0)
    pows = [2, 2.5, 3, 3.5, 4, 4.5, 5]

    approach = "payoff"

    match approach:
        case "weight":
            weight_problems = generate_weight_problems(size=150, m=50, rng=rng, corr=False)

            values = weight_problems[0].weights
            inverse, eps = inverse_eps_opt(inverse_weights, weight_problems, values, pows)

        case "payoff":
            payoff_problems = generate_payoff_problems(size=100, m=25, rng=rng)

            values = payoff_problems[0].payoffs
            inverse, _ = inverse_eps_opt(inverse_payoffs_direct, payoff_problems, values, pows, True)
            # inverse, _ = inverse_payoffs_delta(payoff_problems)


    error = np.abs(values - inverse).sum()
    print(error)
    print(time() - start)