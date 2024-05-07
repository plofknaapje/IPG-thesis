import gurobipy as gp
from gurobipy import GRB
import numpy as np
from numpy.random import Generator

from problems.knapsack_packing_game import KnapsackPackingGame
from problems.base import ApproxOptions

eps = 0.001


def generate_weight_problems(
    size: int,
    n: int,
    m: int,
    r: int = 100,
    capacity: float | list[float] | list[list[float]] | None = None,
    corr=True,
    inter_factor=3,
    neg_inter=False,
    approx_options: ApproxOptions | None = None,
    rng: Generator | None = None,
    verbose=False,
    solve=True,
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
        neg_inter (bool, optional): Allow for negative interactions. Defaults to False.
        approx_options (ApproxOptions | None, optional): How to deal with approximate solutions?. Defaults to None.
        rng (Generator | None, optional): Random number generator. Defaults to None.
        verbose (bool, optional): Verbose outputs with progress details. Defaults to False.
        solve (bool, optional): Solve all problems and check solvability. Defaults to True.

    Returns:
        list[KnapsackPackingGame]: KnapsackPackingGame instances with the same weights vector.
    """
    if rng is None:
        rng = np.random.default_rng()
    if approx_options is None:
        approx_options = ApproxOptions()

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

        if neg_inter:
            interactions = rng.integers(- np.floor(r / inter_factor), np.floor(r / inter_factor) + 1, (n, n, m))
        else:
            interactions = rng.integers(0, np.floor(r / inter_factor) + 1, (n, n, m))

        for i in range(n):
            interactions[i, i, :] = 0

        problem = KnapsackPackingGame(
            weights, payoffs, interactions, list(capacity[len(problems)] * weight_sum)
        )

        if solve:
            result = problem.solve(verbose, approx_options.timelimit)

            if approx_options.valid_problem(result):
                problems.append(problem)

            if len(problems) != 0 and len(problems) % 10 == 0:
                print(f"{len(problems)} problems generated.")
        else:
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
    neg_inter=False,
    approx_options: ApproxOptions | None = None,
    rng=None,
    verbose=False,
    solve=True,
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
        neg_inter (bool, optional): Allow for negative interactions. Defaults to False.
        approx_options (ApproxOptions | None, optional): How to deal with approximate solutions?. Defaults to None.
        rng (Generator | None, optional): Random number generator. Defaults to None.
        verbose (bool, optional): Verbose outputs with progress details. Defaults to False.
        solve (bool, optional): Solve all problems and check solvability. Defaults to True.

    Returns:
        list[KnapsackPackingGame]: KnapsackPackingGame instances with the same weights vector.
    """
    if rng is None:
        rng = np.random.default_rng()
    if approx_options is None:
        approx_options = ApproxOptions()

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
        if neg_inter:
            mask = rng.integers(-1, 2, (n, n, m))
        else:
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

        if solve:
            result = problem.solve(verbose, approx_options.timelimit)

            if approx_options.valid_problem(result):
                problems.append(problem)

            if len(problems) != 0 and len(problems) % 10 == 0:
                print(f"{len(problems)} problems generated.")
        else:
            problems.append(problem)

    return problems


def inverse_weights(
    problems: list[KnapsackPackingGame], sub_timelimit: int | None = None, verbose=False
) -> np.ndarray:
    """
    Determine the shared weights matrix of the problems using inverse optimization.

    Args:
        problems (list[KnapsackPackingGame]): KnapsackPackingGames with the same weights matrix.
        sub_timelimit (int | Non e, optional): Soft timelimit for solving player problems. Defaults to None.
        verbose (bool, optional): Verbose outputs with progress details. Defaults to False.

    Raises:
        ValueError: Problem is infeasible.

    Returns:
        np.ndarray: The inversed weights matrix.
    """
    n = problems[0].n
    players = problems[0].players
    m = problems[0].m
    true_objs = {
        (i, player): problem.obj_value(player)
        for i, problem in enumerate(problems)
        for player in players
    }

    model = gp.Model("Inverse KPG (Weights)")

    w = model.addMVar((n, m), vtype=GRB.INTEGER, lb=1, name="w")

    model.setObjective(w.sum())

    # model.addConstrs(w[j].sum() >= problems[0].weights[j].sum() for j in players)

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
                new_player_x = problem.solve_player(
                    j, weights=w.X, timelimit=sub_timelimit
                )
                new_player_obj = problem.obj_value(j, player_solution=new_player_x)

                if new_player_obj >= true_objs[i, j] + eps:
                    model.addConstr(new_player_x @ w[j] >= problem.capacity[j] + eps)
                    new_constraint = True

        if not new_constraint:
            break

        model.optimize()

        if model.Status == GRB.INFEASIBLE:
            raise ValueError("Problem is Infeasible!")

        if np.array_equal(current_w, w.X):
            break

        if verbose:
            error = np.abs(current_w - problems[0].weights).sum()
            print(error, error / problems[0].weights.sum())

    inverse = w.X

    model.close()

    return inverse.astype(int)


def inverse_payoffs(
    problems: list[KnapsackPackingGame], sub_timelimit: int | None = None, verbose=False
) -> np.ndarray:
    """
    Determine the shared payoffs matrix of the problems using inverse optimization.
    Combines the delta and direct method.

    Args:
        problems (list[KnapsackPackingGame]): KnapsackPackingGames with the same payoffs matrix.
        sub_timelimit (int | Non e, optional): Soft timelimit for solving player problems. Defaults to None.
        verbose (bool, optional): Verbose outputs with progress details. Defaults to False.

    Raises:
        ValueError: Problem is infeasible.

    Returns:
        np.ndarray: The inversed payoffs matrix.
    """
    n_problems = len(problems)
    n_players = problems[0].n
    players = problems[0].players
    n_items = problems[0].m

    model = gp.Model("Inverse KPG (Weights)")

    delta = model.addMVar((n_problems, n_players), name="delta")
    p = model.addMVar((n_players, n_items), vtype=GRB.INTEGER, lb=1, name="p")

    model.setObjective(delta.sum())

    # model.addConstrs(p[j].sum() <= problems[0].payoffs[j].sum() for j in players)

    true_objs = {
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
                new_player_x = problem.solve_player(
                    j, payoffs=p.X, timelimit=sub_timelimit
                )
                if tuple(new_player_x) in solutions[i, j]:
                    continue

                new_player_obj = new_player_x @ p[j] + sum(
                    new_player_x * problem.solution[o] @ problem.inter_coefs[j, o]
                    for o in problem.opps[j]
                )
                model.addConstr(delta[i, j] >= new_player_obj - true_objs[i, j])
                new_constraint = True
                solutions[i, j].add(tuple(new_player_x))

        if not new_constraint:
            break

        model.optimize()

        if model.Status == GRB.INFEASIBLE:
            raise ValueError("Problem is Infeasible!")

        if np.array_equal(p.X, current_p):
            break

        if verbose:
            error = np.abs(current_p - problems[0].payoffs).sum()
            print(error, error / problems[0].payoffs.sum())

    inverse = p.X

    model.close()

    return inverse.astype(int)
