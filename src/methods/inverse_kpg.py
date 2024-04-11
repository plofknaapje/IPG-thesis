from time import time

import gurobipy as gp
from gurobipy import GRB
import numpy as np
from numpy.random import Generator


from problems.knapsack_packing_game import KnapsackPackingGame

eps = 0.001


def generate_weight_problems(
    size: int,
    n: int,
    m: int,
    r: int = 100,
    capacity: float | list[float] | list[list[float]] | None = None,
    corr=True,
    inter_factor=3,
    rng: Generator | None = None,
    verbose=False,
) -> list[KnapsackPackingGame]:
    """
    Generate KnapsackPackingGame instances with a shared weights matrix.

    Args:
        size (int): Number of instances.
        n (int): Number of players.
        m (int): Number of items.
        r (int, optional): Range of payoff, weight and interaction values. Defaults to 100.
        capacity (float | list[float] | list[list[float]] | None, optional): Fractional capacity of instances. Defaults to None.
        corr (bool, optional): Should weights and payoffs be correlated?. Defaults to True.
        inter_factor (int, optional): Denominator to limit the influence of interaction. Defaults to 3.
        rng (Generator | None, optional): Random number generator. Defaults to None.
        verbose (bool, optional): Verbose outputs with progress details. Defaults to False.

    Returns:
        list[KnapsackPackingGame]: KnapsackPackingGame instances with the same weights vector.
    """
    if rng is None:
        rng = np.random.default_rng()

    problems = []

    weights = rng.integers(1, r + 1, (n, m))
    weight_sum = weights.sum(axis=1)

    if capacity is None:
        capacity: np.ndarray = rng.uniform(0.2, 0.8, (size, n))
    elif isinstance(capacity, float):
        capacity: np.ndarray = np.ones((size, n)) * capacity
    elif isinstance(capacity[0], float):
        capacity: np.ndarray = np.ones((size, n)) * np.array(capacity)
    else:
        capacity: np.ndarray = np.array(capacity)

    if corr:
        lower = np.maximum(weights - r / 5, 1)
        upper = np.minimum(weights + r / 5, r + 1)

    while len(problems) < size:
        if corr:
            payoffs = rng.integers(lower, upper)
        else:
            payoffs = rng.integers(1, r + 1, (n, m))

        interactions = rng.integers(1, int(r / inter_factor) + 1, (n, n, m))
        mask = rng.integers(0, 2, (n, n, m))
        interactions = interactions * mask

        for i in range(n):
            interactions[i, i, :] = 0

        problem = KnapsackPackingGame(
            weights, payoffs, interactions, list(capacity[len(problems)] * weight_sum)
        )
        problem.solve(verbose)
        if problem is not None:
            problems.append(problem)

    return problems


def generate_payoff_problems(
    size: int,
    n: int,
    m: int,
    r: int = 100,
    capacity: float | list[float] | list[list[float]] | None = None,
    corr=True,
    inter_factor=3,
    rng=None,
    verbose=False,
) -> list[KnapsackPackingGame]:
    """
    Generate KnapsackPackingGame instances with a shared weights matrix.

    Args:
        size (int): Number of instances.
        n (int): Number of players.
        m (int): Number of items.
        r (int, optional): Range of payoff, weight and interaction values. Defaults to 100.
        capacity (float | list[float] | list[list[float]] | None, optional): Fractional capacity of instances. Defaults to None.
        corr (bool, optional): Should weights and payoffs be correlated?. Defaults to True.
        inter_factor (int, optional): Denominator to limit the influence of interaction. Defaults to 3.
        rng (Generator | None, optional): Random number generator. Defaults to None.
        verbose (bool, optional): Verbose outputs with progress details. Defaults to False.

    Returns:
        list[KnapsackPackingGame]: KnapsackPackingGame instances with the same weights vector.
    """
    if rng is None:
        rng = np.random.default_rng()

    problems = []

    payoffs = rng.integers(1, r + 1, (n, m))

    if capacity is None:
        capacity: np.ndarray = rng.uniform(0.2, 0.8, (size, n))
    elif isinstance(capacity, float):
        capacity: np.ndarray = np.ones((size, n)) * capacity
    elif isinstance(capacity[0], float):
        capacity: np.ndarray = np.ones((size, n)) * np.array(capacity)
    else:
        capacity: np.ndarray = np.array(capacity)

    if corr:
        lower = np.maximum(payoffs - r / 5, 1)
        upper = np.minimum(payoffs + r / 5, r + 1)

    while len(problems) < size:
        if corr:
            weights = rng.integers(lower, upper)
        else:
            weights = rng.integers(1, r + 1, (n, m))

        interactions = rng.integers(0, int(r / inter_factor) + 1, (n, n, m))
        mask = rng.integers(0, 2, (n, n, m))
        interactions = interactions * mask

        for i in range(n):
            interactions[i, i, :] = 0

        problem = KnapsackPackingGame(
            weights,
            payoffs,
            interactions,
            list(capacity[len(problems)] * weights.sum(axis=1)),
        )
        problem.solve(verbose)
        if problem is not None:
            problems.append(problem)

    return problems


def inverse_weights(problems: list[KnapsackPackingGame], verbose=False) -> np.ndarray:
    """
    Determine the shared weights matrix of the problems using inverse optimization.

    Args:
        problems (list[KnapsackPackingGame]): KnapsackPackingGames with the same weights matrix.
        verbose (bool, optional): Verbose outputs with progress details. Defaults to False.

    Raises:
        ValueError: Problem is infeasible.

    Returns:
        np.ndarray: The inversed weights matrix.
    """
    n = problems[0].n
    players = problems[0].players
    m = problems[0].m
    true_value = {
        (i, player): problem.solved_obj_value(player)
        for i, problem in enumerate(problems)
        for player in players
    }

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
        current_w = w.X

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

        if np.array_equal(current_w, w.X):
            break

        if model.Status == GRB.INFEASIBLE:
            raise ValueError("Problem is Infeasible!")

        if verbose:
            error = np.abs(inverse - problems[0].weights).sum()
            print(error, error / problems[0].weights.sum())

    inverse = w.X

    model.close()

    return inverse.astype(int)


def inverse_payoffs(problems: list[KnapsackPackingGame], verbose=False) -> np.ndarray:
    """
    Determine the shared payoffs matrix of the problems using inverse optimization.
    Combines the delta and direct method.

    Args:
        problems (list[KnapsackPackingGame]): KnapsackPackingGames with the same payoffs matrix.
        verbose (bool, optional): Verbose outputs with progress details. Defaults to False.

    Raises:
        ValueError: Problem is infeasible.

    Returns:
        np.ndarray: The inversed payoffs matrix.
    """
    n_problems = len(problems)
    n = problems[0].n
    players = problems[0].players
    m = problems[0].m

    model = gp.Model("Inverse KPG (Weights)")

    delta = model.addMVar((n_problems, n), name="delta")
    p = model.addMVar((n, m), vtype=GRB.INTEGER, name="p")

    model.setObjective(delta.sum())

    model.addConstrs(p[j].sum() <= problems[0].payoffs[j].sum() for j in players)

    true_values = {
        (i, j): problem.solution[j] @ p[j]
        + sum(
            problem.solution[j] * problem.solution[o] @ problem.inter_coefs[j, o]
            for o in problem.opps[j]
        )
        for i, problem in enumerate(problems)
        for j in players
    }

    model.optimize()

    if model.Status == GRB.INFEASIBLE:
        raise ValueError("Problem is Infeasible!")

    solutions = {(i, j): set() for i in range(n_problems) for j in players}

    while True:
        new_constraint = False
        current_p = p.X  # for comparison after optimization

        for i, problem in enumerate(problems):
            for j in players:
                new_solution = problem.solve_player_payoffs(p.X, j)
                if tuple(new_solution) in solutions[i, j]:
                    continue

                new_value = new_solution @ p[j] + sum(
                    new_solution * problem.solution[o] @ problem.inter_coefs[j, o]
                    for o in problem.opps[j]
                )
                model.addConstr(delta[i, j] >= new_value - true_values[i, j])
                new_constraint = True
                solutions[i, j].add(tuple(new_solution))

                temp_new_value = new_solution @ p.X[j] + sum(
                    new_solution * problem.solution[o] @ problem.inter_coefs[j, o]
                    for o in problem.opps[j]
                )

                temp_true_value = problem.solution[j] @ p.X[j] + sum(
                    problem.solution[j]
                    * problem.solution[o]
                    @ problem.inter_coefs[j, o]
                    for o in problem.opps[j]
                )

                if temp_new_value >= temp_true_value + eps:
                    model.addConstr(new_value <= true_values[i, j])

        if not new_constraint:
            break

        model.optimize()

        if np.array_equal(p.X, current_p):
            break

        if model.Status == GRB.INFEASIBLE:
            raise ValueError("Problem is Infeasible!")

        if verbose:
            error = np.abs(inverse - problems[0].payoffs).sum()
            print(error, error / problems[0].payoffs.sum())

    inverse = p.X

    model.close()

    return inverse.astype(int)
