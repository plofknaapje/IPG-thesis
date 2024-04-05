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

    if type(capacity) is float:
        capacity = [capacity * weights.sum() for _ in range(size)]
    else:
        capacity = [fraction * weights.sum() for fraction in capacity]

    if corr:
        lower = np.maximum(weights - r / 10, 1)
        upper = np.minimum(weights + r / 10, r+1)

    for i in range(size):
        if corr:
            payoffs = rng.integers(lower, upper)
        else:
            payoffs = rng.integers(1, r+1, m)
        problem = KP(payoffs, weights, capacity[i])
        problem.solve()
        problems.append(problem)

    return problems


def generate_payoff_problems(size:int, m:int=10, r:int=100, capacity:float|list=0.5, corr=True, rng=None) -> list[KP]:
    if rng is None:
        rng = np.random.default_rng()

    problems = []
    payoffs = rng.integers(1, r+1, m)

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
        problem = KP(payoffs, weights, capacity[i] * weights.sum())
        problem.solve()
        problems.append(problem)

    return problems


def inverse_weights(problems: list[KP]) -> np.ndarray:
    items = problems[0].n_items
    true_value = [problem.solution @ problem.payoffs for problem in problems]

    model = gp.Model("Inverse Knapsack (Weights)")

    w = model.addMVar((items), vtype=GRB.INTEGER, name="w")

    model.setObjective(w.sum())

    model.addConstr(w.sum() >= problems[0].weights.sum())

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

            if new_value >= true_value[i] + 0.1:
                model.addConstr(new_solution @ w >= problem.capacity + 0.1)
                new_constraint = True

        if not new_constraint:
            break

        model.optimize()

        if model.Status == GRB.INFEASIBLE:
            raise ValueError("Problem is Infeasible!")

    inverse = w.X

    model.close()

    return inverse


def inverse_payoffs_direct(problems: list[KP]) -> np.ndarray:
    n_items = problems[0].n_items

    model = gp.Model("Inverse Knapsack (Payoffs)")

    p = model.addMVar((n_items), vtype=GRB.INTEGER, name="p")

    model.setObjective(gp.quicksum(p @ problem.solution for problem in problems), GRB.MAXIMIZE)

    model.addConstr(p.sum() <= problems[0].payoffs.sum())

    model.optimize()

    if model.Status == GRB.INFEASIBLE:
        raise ValueError("Problem is Infeasible!")

    while True:
        new_constraint = False

        for _, problem in enumerate(problems):
            new_solution = problem.solve_payoffs(p.X)
            new_value = new_solution @ p.X
            true_value = problem.solution @ p.X

            if new_value >= true_value + 0.1:
                model.addConstr(problem.solution @ p >= new_solution @ p)
                new_constraint = True

        if not new_constraint:
            break

        model.optimize()

        if model.Status == GRB.INFEASIBLE:
            raise ValueError("Problem is Infeasible!")

    inverse = p.X

    model.close()

    return inverse


def inverse_payoffs_delta(problems: list[KP]) -> np.ndarray:
    n_items = problems[0].n_items
    n_problems = len(problems)

    model = gp.Model("Inverse Knapsack (Payoffs)")

    delta = model.addMVar((n_problems), name="delta")
    p = model.addMVar((n_items), vtype=GRB.INTEGER, name="p")

    model.setObjective(delta.sum())

    model.addConstr(p.sum() == problems[0].payoffs.sum())

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
        
    inverse = p.X

    model.close()

    return inverse


def inverse_payoffs_hybrid(problems: list[KP]) -> np.ndarray:
    n_items = problems[0].n_items
    n_problems = len(problems)

    model = gp.Model("Inverse Knapsack (Payoffs)")

    delta = model.addMVar((n_problems), name="delta")
    p = model.addMVar((n_items), vtype=GRB.INTEGER, name="p")

    model.setObjective(delta.sum())

    model.addConstr(p.sum() == problems[0].payoffs.sum())

    model.optimize()

    if model.Status == GRB.INFEASIBLE:
        raise ValueError("Problem is Infeasible!")

    solutions = [set() for _ in problems]

    while True:
        new_constraint = False

        for i, problem in enumerate(problems):
            new_solution = problem.solve_payoffs(p.X)

            new_value = new_solution @ p.X
            true_value = problem.solution @ p.X

            if new_value >= true_value + 0.1:
                model.addConstr(problem.solution @ p >= new_solution @ p)

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

    inverse = p.X

    model.close()

    return inverse


if __name__ == "__main__":

    start = time()
    rng = np.random.default_rng(0)

    approach = "weight"

    match approach:
        case "weight":
            weight_problems = generate_weight_problems(size=100, m=50, rng=rng, corr=True)
            print("Finished generating problems")

            values = weight_problems[0].weights
            inverse = inverse_weights(weight_problems)

        case "payoff":
            payoff_problems = generate_payoff_problems(size=100, m=25, rng=rng)
            print("Finished generating problems")

            values = payoff_problems[0].payoffs
            # inverse = inverse_payoffs_direct(payoff_problems)
            inverse = inverse_payoffs_delta(payoff_problems)
            # inverse = inverse_payoffs_hybrid(payoff_problems)

    print(values.sum())

    error = np.abs(values - inverse).sum()
    print(error, error / values.sum())
    print(time() - start)