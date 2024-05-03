import numpy as np
from numpy.random import Generator
import gurobipy as gp
from gurobipy import GRB

from problems.knapsack_problem import KnapsackProblem

eps = 0.001


def generate_problem(
    n: int,
    r: int = 100,
    capacity: float | None = None,
    corr: bool = True,
    rng: Generator | None = None,
) -> KnapsackProblem:
    """
    Generates a single KnapsackProblem.

    Args:
        n (int): Number of items
        r (int, optional): Range of payoff and weight values. Defaults to 100.
        capacity (float | None, optional): Fractional capacity of instances. Defaults to None.
        corr (bool, optional): Should weights and payoffs be correlated?. Defaults to True.
        rng (Generator, optional): Random number generator. Defaults to None.

    Returns:
        KnapsackProblem: Generated KnapsackProblem instance.
    """
    if rng is None:
        rng = np.random.default_rng()

    payoffs = rng.integers(1, r + 1, n)

    if capacity is None:
        capacity = float(rng.uniform(0.2, 0.8))

    if corr:
        weights = rng.integers(
            np.maximum(payoffs - r / 5, 1), np.minimum(payoffs + r / 5, r + 1), n
        )
    else:
        weights = rng.integers(1, r + 1, n)

    return KnapsackProblem(payoffs, weights, capacity * weights.sum())


def local_inverse_weights(problem: KnapsackProblem) -> np.ndarray:
    """
    Inverse the problem such that the greedy solution becomes the optimal solution.
    This is done by minimally adjusting the weights vector.

    Args:
        problem (KnapsackProblem): Problem instance.

    Raises:
        ValueError: The problem is infeasible.

    Returns:
        np.ndarray: The inverse payoffs vector.
    """
    greedy_solution = problem.solve_greedy()
    i_range = list(range(problem.n))

    model = gp.Model("Local Inverse Weights")

    w = model.addMVar((problem.n), vtype=GRB.INTEGER, lb=1)
    delta = model.addMVar((problem.n))

    model.setObjective(delta.sum())

    model.addConstr(w @ greedy_solution <= problem.capacity)

    model.addConstrs(delta[i] >= w[i] - problem.weights[i] for i in i_range)
    model.addConstrs(delta[i] >= problem.weights[i] - w[i] for i in i_range)

    model.optimize()
    if model.Status == GRB.INFEASIBLE:
        raise ValueError("Problem is Infeasible!")

    while True:
        new_solution = problem.solve(weights=w.X)

        if new_solution @ problem.payoffs >= greedy_solution @ problem.payoffs + eps:
            model.addConstr(new_solution @ w >= problem.capacity + eps)
        elif new_solution @ problem.payoffs <= greedy_solution @ problem.payoffs:
            print("Solution is now optimal")
            break

        model.optimize()
        if model.Status == GRB.INFEASIBLE:
            raise ValueError("Problem is Infeasible!")

    return w.X.astype(int)


def local_inverse_payoffs(problem: KnapsackProblem) -> np.ndarray:
    """
    Inverse the problem such that the greedy solution becomes the optimal solution.
    This is done by minimally adjusting the payoffs vector.

    Args:
        problem (KnapsackProblem): Problem instance.

    Raises:
        ValueError: The problem is infeasible.

    Returns:
        np.ndarray: The inverse payoffs vector.
    """
    greedy_solution = problem.solve_greedy()
    i_range = list(range(problem.n))

    model = gp.Model("Local Inverse Payoffs")

    y = model.addMVar((problem.n), lb=0, vtype=GRB.INTEGER)
    p = model.addMVar((problem.n), vtype=GRB.INTEGER, lb=1)
    delta = model.addMVar((problem.n))

    model.setObjective(delta.sum())

    model.addConstrs(y[i] * problem.weights[i] >= p[i] for i in i_range)
    model.addConstrs(delta[i] >= p[i] - problem.payoffs[i] for i in i_range)
    model.addConstrs(delta[i] >= problem.payoffs[i] - p[i] for i in i_range)

    model.optimize()

    if model.Status == GRB.INFEASIBLE:
        raise ValueError("Problem is Infeasible!")

    solutions = set()

    while True:
        # new_payoffs = problem.payoffs - e.X + f.X
        new_payoffs = p.X
        new_solution = problem.solve(payoffs=new_payoffs)

        if new_payoffs @ new_solution <= new_payoffs @ greedy_solution:
            print("Solution is now optimal")
            break

        elif tuple(new_solution) in solutions:
            print("Duplicate solution")
            break

        solutions.add(tuple(new_solution))

        model.addConstr(p @ greedy_solution >= p @ new_solution)

        model.optimize()

        if model.Status == GRB.INFEASIBLE:
            raise ValueError("Problem is Infeasible!")

    return p.X.astype(int)
    # return (problem.payoffs - e.X + f.X).astype(int)
