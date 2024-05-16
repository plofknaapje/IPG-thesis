from time import time

import numpy as np
import gurobipy as gp
from gurobipy import GRB

from problems.knapsack_packing_game import KnapsackPackingGame


eps = 0.001


def local_inverse_weights(problem: KnapsackPackingGame, timelimit=None) -> np.ndarray:
    start = time()

    greedy_solution = problem.solve_greedy()
    i_range = list(range(problem.m))
    obj_values = [problem.obj_value(p, greedy_solution) for p in problem.players]

    model = gp.Model("Local Inverse Weights")

    w = model.addMVar((problem.n, problem.m), vtype=GRB.INTEGER, lb=1)

    delta = model.addMVar((problem.n, problem.m))

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
        current_w = w.X

        for p in problem.players:
            new_player_x = problem.solve_player(
                p, current_sol=greedy_solution, weights=current_w
            )

            if (
                problem.obj_value(
                    p, solution=greedy_solution, player_solution=new_player_x
                )
                >= obj_values[p] + eps
            ):
                model.addConstr(new_player_x @ w[p] >= problem.capacity[p] + eps)
                new_constraint = True

        if not new_constraint:
            break

        model.optimize()

        if model.Status == GRB.INFEASIBLE:
            raise ValueError("Problem is Infeasible")

        if np.array_equal(current_w, w.X):
            break

        if timelimit is not None and time() - start >= timelimit:
            raise ValueError("Time limit reached!")

    inverse = w.X

    model.close()

    return inverse.astype(int)


def local_inverse_payoffs(problem: KnapsackPackingGame, timelimit=None) -> tuple[np.ndarray, np.ndarray]:
    start = time()

    greedy_solution = problem.solve_greedy()
    i_range = list(range(problem.m))

    model = gp.Model("Local Inverse Weights")

    p = model.addMVar((problem.n, problem.m), vtype=GRB.INTEGER, lb=1)
    inter = model.addMVar((problem.n, problem.n, problem.m), vtype=GRB.INTEGER)

    for j in problem.players:
        for i in i_range:
            inter[j, j, i].lb = 0
            inter[j, j, i].ub = 0

    if problem.inter_coefs.min() >= 0:
        inter.lb = 0

    delta_p = model.addMVar((problem.n, problem.m))
    delta_i = model.addMVar((problem.n, problem.n, problem.m))

    model.setObjective(delta_p.sum() + delta_i.sum())

    model.addConstrs(
        delta_p[j, i] >= p[j, i] - problem.payoffs[j, i]
        for i in i_range
        for j in problem.players
    )
    model.addConstrs(
        delta_p[j, i] >= problem.payoffs[j, i] - p[j, i]
        for i in i_range
        for j in problem.players
    )

    model.addConstrs(
        delta_i[j, k, i] >= problem.inter_coefs[j, k, i] - inter[j, k, i]
        for i in i_range
        for j in problem.players
        for k in problem.players
    )

    model.addConstrs(
        delta_i[j, k, i] >= inter[j, k, i] - problem.inter_coefs[j, k, i]
        for i in i_range
        for j in problem.players
        for k in problem.players
    )

    true_values = [
        greedy_solution[j] @ p[j]
        + sum(
            greedy_solution[j] * greedy_solution[o] @ inter[j, o]
            for o in problem.opps[j]
        )
        for j in problem.players
    ]

    model.optimize()
    if model.Status == GRB.INFEASIBLE:
        raise ValueError("Problem is initially Infeasible")

    solutions = [set() for _ in problem.players]

    while True:
        new_constraint = False
        current_p = p.X
        current_i = inter.X

        for j in problem.players:
            new_player_x = problem.solve_player(
                j, current_sol=greedy_solution, payoffs=current_p, inter_coefs=current_i
            )

            if tuple(new_player_x) in solutions[j]:
                continue

            new_value = new_player_x @ p[j] + sum(
                new_player_x * greedy_solution[o] @ inter[j, o]
                for o in problem.opps[j]
            )

            if new_value.getValue() >= true_values[j].getValue():
                model.addConstr(new_value <= true_values[j])
                new_constraint = True
                solutions[j].add(tuple(new_player_x))

        if not new_constraint:
            break

        model.optimize()

        if model.Status == GRB.INFEASIBLE:
            raise ValueError("Problem is Infeasible")

        if np.array_equal(current_p, p.X) and np.array_equal(current_i, inter.X):
            break

        if timelimit is not None and time() - start >= timelimit:
            raise ValueError("Time limit reached!")

    inverse_p = p.X
    inverse_i = inter.X

    model.close()

    return inverse_p.astype(int), inverse_i.astype(int)
