from typing import List, Optional
from time import time

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
    capacity: float | List[float] = None,
    corr=True,
    rng: Optional[Generator] = None,
) -> List[KnapsackProblem]:
    """
    Generates KnapsackProblem instances with the same weights vector.

    Args:
        size (int, optional): Number of instances.
        n (int, optional): Number of items.
        r (int, optional): Range of payoff and weight values. Defaults to 100.
        capacity (float | List[float], optional): Fractional capacity of instances. Defaults to None.
        corr (bool, optional): Should weights and payoffs be correlated?. Defaults to True.
        rng (Generator, optional): Random number generator. Defaults to None.

    Returns:
        List[KnapsackProblem]: KnapsackProblem instances with the same weights vector.
    """
    if rng is None:
        rng = np.random.default_rng()

    problems = []
    weights = rng.integers(1, r + 1, n)
    weight_sum = weights.sum()

    if capacity is None:
        capacity = rng.uniform(0.2, 0.8, size).tolist()
    elif isinstance(capacity, float):
        capacity = [capacity for _ in range(size)]

    if corr:
        lower = np.maximum(weights - r / 5, 1)
        upper = np.minimum(weights + r / 5, r + 1)

    for i in range(size):
        if corr:
            payoffs = rng.integers(lower, upper)
        else:
            payoffs = rng.integers(1, r + 1, n)
        problem = KnapsackProblem(payoffs, weights, capacity[i])
        problem.solve()
        problems.append(problem)

    return problems


def generate_payoff_problems(
    size: int,
    n: int = 10,
    r: int = 100,
    capacity: float | List[float] = None,
    corr=True,
    rng: Optional[Generator] = None,
) -> List[KnapsackProblem]:
    """
    Generates KnapsackProblem instances with the same payoffs vector.

    Args:
        size (int): Number of instances
        n (int, optional): Number of items. Defaults to 10.
        r (int, optional): Range of payoff and weight values. Defaults to 100.
        capacity (float | List[float], optional): Fractional capacity of instances. Defaults to None.
        corr (bool, optional): Should weights and payoffs be correlated?. Defaults to True.
        rng (Generator, optional): Random number generator. Defaults to None.

    Returns:
        List[KnapsackProblem]: KnapsackProblem instances with the same payoffs vector.
    """
    if rng is None:
        rng = np.random.default_rng()

    problems = []
    payoffs = rng.integers(1, r + 1, n)

    if capacity is None:
        capacity = list(rng.uniform(0.2, 0.8, size))
    elif isinstance(capacity, float):
        capacity = [capacity for _ in range(size)]

    if corr:
        lower = np.maximum(payoffs - np.ceil(payoffs - r / 5), 1)
        upper = np.minimum(payoffs + np.floor(payoffs + r / 5), r + 1)

    for i in range(size):
        if corr:
            weights = rng.integers(lower, upper, n)
        else:
            weights = rng.integers(1, r + 1, n)
        problem = KnapsackProblem(payoffs, weights, capacity[i])
        problem.solve()
        problems.append(problem)

    return problems


def inverse_weights(
    problems: List[KnapsackProblem], timelimit: Optional[float] = None, verbose=False
) -> np.ndarray:
    """
    Determine the shared weights vector of the problems using inverse optimization.

    Args:
        problems (List[KnapsackProblem]): KnapsackProblems with the same weights vector.
        verbose (bool, optional): Verbose outputs with progress details. Defaults to False.

    Raises:
        ValueError: The problem is infeasible.

    Returns:
        np.ndarray: The inversed weights vector.
    """
    start = time()
    n = problems[0].n
    true_obj = [problem.solution @ problem.payoffs for problem in problems]
    weights = problems[0].weights

    model = gp.Model("Inverse Knapsack (Weights)")
    if timelimit is not None:
        model.params.TimeLimit = timelimit / 8

    w = model.addMVar((n), vtype=GRB.INTEGER, lb=1, name="w")

    model.setObjective(1)

    model.addConstr(w.sum() == weights.sum())

    for problem in problems:
        model.addConstr(problem.solution @ w <= problem.capacity)

    new_constraint = True
    current_w = np.zeros_like(weights)

    while new_constraint:
        model.optimize()

        while model.SolCount == 0:
            if timelimit is not None and time() - start >= timelimit:
                return current_w.astype(int)

            model.optimize()

        if model.Status == GRB.INFEASIBLE:
            raise ValueError("Problem is Infeasible!")
        
        if np.array_equal(current_w, w.X):
            break

        if timelimit is not None and time() - start >= timelimit:
            break

        new_constraint = False
        current_w = w.X

        if verbose:
            error = np.abs(weights - w.X).sum()
            print(error, error / weights.sum())

        for i, problem in enumerate(problems):
            new_x = problem.solve(weights=w.X)
            new_obj = new_x @ problem.payoffs

            if new_obj >= true_obj[i] + eps:
                model.addConstr(new_x @ w >= problem.capacity + eps)
                new_constraint = True

    inverse = w.X

    model.close()

    return inverse.astype(int)


def inverse_payoffs_delta(
    problems: List[KnapsackProblem], timelimit: Optional[float] = None, verbose=False
) -> np.ndarray:
    """
    Determine the shared payoffs vector of the problems using inverse optimization.
    This method uses the delta method which maximises the difference between the
    objective value of the true solution and candidate alternatives.

    Args:
        problems (List[KnapsackProblem]): KnapsackProblems with a shared payoffs vector.
        verbose (bool, Optional): Report progress. Defaults to False.

    Raises:
        ValueError: The problem is infeasible.

    Returns:
        np.ndarray: The inverse payoffs vector.
    """
    start = time()
    n = problems[0].n
    n_problems = len(problems)
    payoffs = problems[0].payoffs

    model = gp.Model("Inverse Knapsack (Payoffs)")
    if timelimit is not None:
        model.params.TimeLimit = timelimit / 2

    delta = model.addMVar((n_problems), name="delta")
    p = model.addMVar((n), vtype=GRB.INTEGER, lb=1, ub=np.max(payoffs), name="p")

    model.setObjective(delta.sum())

    model.addConstr(p.sum() == payoffs.sum())

    new_constraint = True
    current_p = np.zeros_like(payoffs)


    solutions = [set() for _ in problems]

    while new_constraint:
        model.optimize()

        while model.SolCount == 0:
            if timelimit is not None and time() - start >= timelimit:
                return current_p.astype(int)

            model.optimize()

        if model.Status == GRB.INFEASIBLE:
            raise ValueError("Problem is Infeasible!")
        
        if np.array_equal(current_p, p.X):
            break

        if timelimit is not None and time() - start >= timelimit:
            break

        new_constraint = False
        current_p = p.X

        if verbose:
            error = np.abs(payoffs - p.X).sum()
            print(error, error / payoffs.sum())

        for i, problem in enumerate(problems):
            new_x = problem.solve(payoffs=p.X)
            if tuple(new_x) in solutions[i]:
                continue
            model.addConstr(delta[i] >= new_x @ p - problem.solution @ p)
            new_constraint = True
            solutions[i].add(tuple(new_x))

    inverse = p.X

    model.close()

    return inverse.astype(int)
