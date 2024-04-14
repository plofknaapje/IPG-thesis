import gurobipy as gp
from gurobipy import GRB
import numpy as np
from numpy.random import Generator

from problems.knapsack_problem import KnapsackProblem

eps = 0.001


def generate_weight_problems(
    size: int,
    n: int,
    r: int = 100,
    capacity: float | list[float] | None = None,
    corr=True,
    rng: Generator | None = None,
) -> list[KnapsackProblem]:
    """
    Generates KnapsackProblem instances with the same weights vector.

    Args:
        size (int, optional): Number of instances.
        n (int, optional): Number of items.
        r (int, optional): Range of payoff and weight values. Defaults to 100.
        capacity (float | list[float] | None, optional): Fractional capacity of instances. Defaults to None.
        corr (bool, optional): Should weights and payoffs be correlated?. Defaults to True.
        rng (Generator | None, optional): Random number generator. Defaults to None.

    Returns:
        list[KnapsackProblem]: KnapsackProblem instances with the same weights vector.
    """
    if rng is None:
        rng = np.random.default_rng()

    problems = []
    weights = rng.integers(1, r + 1, n)
    weight_sum = weights.sum()

    if capacity is None:
        capacity: list[float] = list(rng.uniform(0.2, 0.8, size))
    elif isinstance(capacity, float):
        capacity: list[float] = [capacity for _ in range(size)]
    else:
        capacity: list[float] = capacity

    if corr:
        lower = np.maximum(weights - r / 10, 1)
        upper = np.minimum(weights + r / 10, r + 1)

    for i in range(size):
        if corr:
            payoffs = rng.integers(lower, upper)
        else:
            payoffs = rng.integers(1, r + 1, n)
        problem = KnapsackProblem(payoffs, weights, capacity[i] * weight_sum)
        problem.solve()
        problems.append(problem)

    return problems


def generate_payoff_problems(
    size: int,
    n: int = 10,
    r: int = 100,
    capacity: float | list[float] | None = None,
    corr=True,
    rng: Generator | None = None,
) -> list[KnapsackProblem]:
    """
    Generates KnapsackProblem instances with the same payoffs vector.

    Args:
        size (int): Number of instances
        n (int, optional): Number of items. Defaults to 10.
        r (int, optional): Range of payoff and weight values. Defaults to 100.
        capacity (float | list[float] | None, optional): Fractional capacity of instances. Defaults to None.
        corr (bool, optional): Should weights and payoffs be correlated?. Defaults to True.
        rng (Generator | None, optional): Random number generator. Defaults to None.

    Returns:
        list[KnapsackProblem]: KnapsackProblem instances with the same payoffs vector.
    """
    if rng is None:
        rng = np.random.default_rng()

    problems = []
    payoffs = rng.integers(1, r + 1, n)

    if capacity is None:
        capacity: list[float] = list(rng.uniform(0.2, 0.8, size))
    elif isinstance(capacity, float):
        capacity: list[float] = [capacity for _ in range(size)]

    if corr:
        lower = np.maximum(payoffs - r / 10, 1)
        upper = np.minimum(payoffs + r / 10, r + 1)

    for i in range(size):
        if corr:
            weights = rng.integers(lower, upper)
        else:
            weights = rng.integers(1, r + 1, n)
        problem = KnapsackProblem(payoffs, weights, capacity[i] * weights.sum())
        problem.solve()
        problems.append(problem)

    return problems


def inverse_weights(problems: list[KnapsackProblem], verbose=False) -> np.ndarray:
    """
    Determine the shared weights vector of the problems using inverse optimization.

    Args:
        problems (list[KnapsackProblem]): KnapsackProblems with the same weights vector.
        verbose (bool, optional): Verbose outputs with progress details. Defaults to False.

    Raises:
        ValueError: The problem is infeasible.

    Returns:
        np.ndarray: The inversed weights vector.
    """
    n = problems[0].n
    true_value = [problem.solution @ problem.payoffs for problem in problems]

    model = gp.Model("Inverse Knapsack (Weights)")

    w = model.addMVar((n), vtype=GRB.INTEGER, name="w")

    model.setObjective(w.sum())

    model.addConstr(w.sum() >= problems[0].weights.sum())

    for problem in problems:
        model.addConstr(problem.solution @ w <= problem.capacity)

    model.optimize()

    if model.Status == GRB.INFEASIBLE:
        raise ValueError("Problem is Infeasible!")

    while True:
        new_constraint = False
        current_w = w.X

        for i, problem in enumerate(problems):
            new_solution = problem.solve_weights(w.X)
            new_value = new_solution @ problem.payoffs

            if new_value >= true_value[i] + eps:
                model.addConstr(new_solution @ w >= problem.capacity + eps)
                new_constraint = True

        if not new_constraint:
            break

        model.optimize()

        if np.array_equal(current_w, w.X):
            break

        if model.Status == GRB.INFEASIBLE:
            raise ValueError("Problem is Infeasible!")

        if verbose:
            error = np.abs(problems[0].weights - w.X).sum()
            print(error, error / problems[0].weights.sum())

    inverse = w.X

    model.close()

    return inverse.astype(int)


def inverse_payoffs_direct(
    problems: list[KnapsackProblem], verbose=False
) -> np.ndarray:
    """
    Determine the shared payoffs vector of the problems using inverse optimization.
    This method optimises the payoff of the given solutions and adds constraints to the feasible set.

    Args:
        problems (list[KnapsackProblem]): KnapsackProblems with a shared payoffs vector.
        verbose (bool, optional): Verbose outputs with progress details. Defaults to False.

    Raises:
        ValueError: The problem is infeasible.

    Returns:
        np.ndarray: The inverse payoffs vector.
    """
    n = problems[0].n

    model = gp.Model("Inverse Knapsack (Payoffs)")

    p = model.addMVar((n), vtype=GRB.INTEGER, name="p")

    model.setObjective(
        gp.quicksum(p @ problem.solution for problem in problems), GRB.MAXIMIZE
    )

    model.addConstr(p.sum() <= problems[0].payoffs.sum())

    model.optimize()

    if model.Status == GRB.INFEASIBLE:
        raise ValueError("Problem is Infeasible!")

    while True:
        new_constraint = False
        current_p = p.X

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

        if np.array_equal(current_p, p.X):
            break

        if model.Status == GRB.INFEASIBLE:
            raise ValueError("Problem is Infeasible!")

        if verbose:
            error = np.abs(problems[0].payoffs - p.X).sum()
            print(error, error / problems[0].payoffs.sum())

    inverse = p.X

    model.close()

    return inverse.astype(int)


def inverse_payoffs_delta(problems: list[KnapsackProblem], verbose=False) -> np.ndarray:
    """
    Determine the shared payoffs vector of the problems using inverse optimization.
    This method uses the delta method which maximises the difference between the
    objective value of the true solution and candidate alternatives.

    Args:
        problems (list[KnapsackProblem]): KnapsackProblems with a shared payoffs vector.
        verbose (bool, Optional): Report progress. Defaults to False.

    Raises:
        ValueError: The problem is infeasible.

    Returns:
        np.ndarray: The inverse payoffs vector.
    """
    n = problems[0].n
    n_problems = len(problems)

    model = gp.Model("Inverse Knapsack (Payoffs)")

    delta = model.addMVar((n_problems), name="delta")
    p = model.addMVar((n), vtype=GRB.INTEGER, name="p")

    model.setObjective(delta.sum())

    model.addConstr(p.sum() == problems[0].payoffs.sum())

    model.optimize()

    if model.Status == GRB.INFEASIBLE:
        raise ValueError("Problem is Infeasible!")

    solutions = [set() for _ in problems]

    while True:
        new_constraint = False
        current_p = p.X

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

        if np.array_equal(current_p, p.X):
            break

        if model.Status == GRB.INFEASIBLE:
            raise ValueError("Problem is Infeasible!")

        if verbose:
            error = np.abs(problems[0].payoffs - p.X).sum()
            print(error, error / problems[0].payoffs.sum())

    inverse = p.X

    model.close()

    return inverse.astype(int)


def inverse_payoffs_hybrid(
    problems: list[KnapsackProblem], verbose=False
) -> np.ndarray:
    """
    Determine the shared payoffs vector of the problems using inverse optimization.
    Combines the approach of inverse_payoffs_direct() and inverse_payoffs_delta().

    Args:
        problems (list[KnapsackProblem]): KnapsackProblems with a shared payoffs vector.
        verbose (bool, Optional): Report progress. Defaults to False.

    Raises:
        ValueError: The problem is infeasible.

    Returns:
        np.ndarray: The inverse payoffs vector.
    """
    n = problems[0].n
    n_problems = len(problems)

    model = gp.Model("Inverse Knapsack (Payoffs)")

    delta = model.addMVar((n_problems), name="delta")
    p = model.addMVar((n), vtype=GRB.INTEGER, name="p")

    model.setObjective(delta.sum())

    model.addConstr(p.sum() == problems[0].payoffs.sum())

    model.optimize()

    if model.Status == GRB.INFEASIBLE:
        raise ValueError("Problem is Infeasible!")

    solutions = [set() for _ in problems]

    while True:
        new_constraint = False
        current_p = p.X

        for i, problem in enumerate(problems):
            new_solution = problem.solve_payoffs(p.X)

            new_value = new_solution @ p.X
            true_value = problem.solution @ p.X

            if new_value >= true_value + eps:
                model.addConstr(problem.solution @ p >= new_solution @ p)

            if tuple(new_solution) in solutions[i]:
                continue

            model.addConstr(delta[i] >= new_solution @ p - problem.solution @ p)
            new_constraint = True
            solutions[i].add(tuple(new_solution))

        if not new_constraint:
            break

        model.optimize()

        if np.array_equal(current_p, p.X):
            break

        if model.Status == GRB.INFEASIBLE:
            raise ValueError("Problem is Infeasible!")

        if verbose:
            error = np.abs(problems[0].payoffs - p.X).sum()
            print(error, error / problems[0].payoffs.sum())

    inverse = p.X

    model.close()

    return inverse.astype(int)
