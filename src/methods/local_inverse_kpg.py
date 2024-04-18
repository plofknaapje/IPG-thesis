import numpy as np
from numpy.random import Generator
import gurobipy as gp
from gurobipy import GRB

from problems.knapsack_packing_game import KnapsackPackingGame


eps = 0.001


def local_inverse_weights(problem: KnapsackPackingGame) -> np.ndarray:
    greedy_solution = problem.solve_greedy()
    i_range = list(range(problem.m))

    model = gp.Model("Local Inverse Weights")

    w = model.addMVar((problem.n, problem.m), vtype=GRB.INTEGER, lb=0)
    delta = model.addMVar((problem.n, problem.m), lb=0)

    model.setObjective(delta.sum())

    model.addConstrs(
        w[p] @ greedy_solution[p] <= problem.capacity[p] for p in problem.players
    )

    model.addConstrs(
        delta[p, i] >= w[p, i] - problem.weights[p, i]
        for i in i_range
        for p in problem.players
    )
    model.addConstrs(
        delta[p, i] >= problem.weights[p, i] - w[p, i]
        for i in i_range
        for p in problem.players
    )

    model.optimize()
    if model.Status == GRB.INFEASIBLE:
        raise ValueError("Problem is initially Infeasible")

    while True:
        new_constraint = False

        for p in problem.players:
            new_player_x = problem.solve_player(
                p, solution=greedy_solution, weights=w.X
            )

            if (
                problem.obj_value(
                    p, solution=greedy_solution, player_solution=new_player_x
                )
                >= problem.obj_value(p, solution=greedy_solution) + eps
            ):
                model.addConstr(new_player_x @ w[p] >= problem.capacity[p] + eps)
                new_constraint = True

        if not new_constraint:
            break

        model.optimize()
        if model.Status == GRB.INFEASIBLE:
            raise ValueError("Problem is Infeasible")

    return w.X.astype(int)


def local_inverse_payoffs(problem: KnapsackPackingGame) -> np.ndarray:
    greedy_solution = problem.solve_greedy()
    i_range = list(range(problem.m))

    model = gp.Model("Local Inverse Weights")

    p = model.addMVar((problem.n, problem.m), vtype=GRB.INTEGER, lb=0)
    delta = model.addMVar((problem.n, problem.m), lb=0)

    model.setObjective(delta.sum())

    model.addConstrs(
        delta[j, i] >= p[j, i] - problem.payoffs[j, i]
        for i in i_range
        for j in problem.players
    )
    model.addConstrs(
        delta[j, i] >= problem.payoffs[j, i] - p[j, i]
        for i in i_range
        for j in problem.players
    )

    true_values = [
        greedy_solution[j] @ p[j]
        + sum(
            greedy_solution[j] * greedy_solution[o] @ problem.inter_coefs[j, o]
            for o in problem.opps[j]
        )
        for j in problem.players
    ]

    model.optimize()
    if model.Status == GRB.INFEASIBLE:
        raise ValueError("Problem is initially Infeasible")

    # solutions = [set() for _ in problem.players]

    while True:
        new_constraint = False

        for j in problem.players:
            new_player_x = problem.solve_player(
                j, solution=greedy_solution, payoffs=p.X
            )

            new_value = new_player_x @ p[j] + sum(
                new_player_x * greedy_solution[o] @ problem.inter_coefs[j, o]
                for o in problem.opps[j]
            )

            if new_value.getValue() >= true_values[j].getValue() + eps:
                model.addConstr(new_value <= true_values[j])
                new_constraint = True

        if not new_constraint:
            break

        model.optimize()
        if model.Status == GRB.INFEASIBLE:
            raise ValueError("Problem is Infeasible")

    return p.X.astype(int)
