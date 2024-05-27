from time import time

import numpy as np
import gurobipy as gp
from gurobipy import GRB

from problems.critical_node_game import CriticalNodeGame

eps = 0.00001


def local_inverse_weights(problem: CriticalNodeGame, defender=True, phi=0, timelimit=None) -> np.ndarray:
    start = time()
    if timelimit is not None:
        local_timelimit = timelimit

    i_range = list(range(problem.n))
    if defender:
        solution = problem.solution[0]
    else:
        solution = problem.solution[1]

    obj_values = [problem.obj_value(True, solution[0], solution[1]),
                  problem.obj_value(False, solution[0], solution[1])]

    model = gp.Model("Local Inverse Weights")
    if timelimit is not None and timelimit <= 0:
        raise UserWarning("No time")
    else:    
        model.params.TimeLimit = max(1, timelimit)
    

    w = model.addMVar((2, problem.n), vtype=GRB.INTEGER, lb=1)

    delta = model.addMVar((2, problem.n))

    model.setObjective(delta.sum())

    model.addConstrs(
        w[j] @ solution[j] <= problem.capacity[j] for j in [0, 1]
    )

    model.addConstrs(
        delta[j, i] >= w[j, i] - problem.weights[j, i]
        for i in i_range
        for j in [0, 1]
    )

    model.addConstrs(
        delta[j, i] >= problem.weights[j, i] - w[j, i]
        for i in i_range
        for j in [0, 1]
    )

    model.optimize()
    if timelimit is not None:
        local_timelimit -= model.Runtime
        model.params.TimeLimit = max(1, local_timelimit)

    if model.Status == GRB.INFEASIBLE:
        raise ValueError("Problem is initially infeasible")

    while True:
        new_constraint = False
        current_w = w.X

        new_def_x = problem.solve_player(True, solution, current_w)
        new_def_obj = problem.obj_value(True, new_def_x, solution[1])

        if new_def_obj >= obj_values[0] + phi:
            model.addConstr(new_def_x @ w[0] >= problem.capacity[0] + eps)
            new_constraint = True

        new_att_x = problem.solve_player(False, solution, current_w)
        new_att_obj = problem.obj_value(False, solution[0], new_att_x)

        if new_att_obj >= obj_values[1] + phi:
            model.addConstr(new_att_x @ w[1] >= problem.capacity[1] + eps)
            new_constraint = True

        if not new_constraint:
            break

        model.optimize()
        if timelimit is not None:
            local_timelimit -= model.Runtime
            model.params.TimeLimit = max(1, local_timelimit)

        if model.Status == GRB.TIME_LIMIT or \
            (timelimit is not None and time() - start >= timelimit):
            print("Timelimit reached")
            raise UserWarning("Timelimit reached")
        
        if model.Status == GRB.INFEASIBLE:
            raise ValueError("Problem is Infeasible")

        if np.array_equal(current_w, w.X):
            break

    inverse = w.X

    model.close()

    return inverse.astype(int)


def local_inverse_payoffs(problem: CriticalNodeGame, defender=True, max_phi: int | None = None, timelimit=None) -> tuple[np.ndarray, int]:
    start = time()
    if timelimit is not None:
        local_timelimit = timelimit

    i_range = list(range(problem.n))
    if defender:
        solution = problem.solution[0]
    else:
        solution = problem.solution[1]

    defence = solution[0]
    attack = solution[1]

    model = gp.Model("Local Inverse Payoffs")

    p = model.addMVar((2, problem.n), vtype=GRB.INTEGER, lb=1)

    delta = model.addMVar((2, problem.n))

    phi_ub = 0
    phi = model.addVar(lb=0, ub=0)

    model.setObjective(delta.sum())

    model.addConstrs(
        delta[j, i] >= p[j, i] - problem.payoffs[j, i]
        for i in i_range
        for j in [0, 1]
    )

    model.addConstrs(
        delta[j, i] >= problem.payoffs[j, i] - p[j, i]
        for i in i_range
        for j in [0, 1]
    )

    true_objs = [0, 0]
    true_objs[0] = p[0] @ (
        (1 - defence) * (1 - attack)
        + problem.mitigated * defence * attack
        + problem.overcommit * defence * (1 - attack)
        + problem.success * (1 - defence) * attack
    )

    true_objs[1] = p[1] @ (
        -problem.normal * (1 - defence) * (1 - attack)
        + (1 - defence) * attack
        + (1 - problem.mitigated) * defence * attack
    )

    model.optimize()
    if timelimit is not None:
        local_timelimit -= model.Runtime
        model.params.TimeLimit = max(1, local_timelimit)

    if model.Status == GRB.INFEASIBLE:
        raise ValueError("Problem is initially Infeasible")

    solutions = [set(), set()]

    while True:
        new_constraint = False
        current_p = p.X

        new_def_x = problem.solve_player(
            True, solution, payoffs=current_p)
        new_def_obj = p[0] @ (
            (1 - new_def_x) * (1 - attack)
            + problem.mitigated * new_def_x * attack
            + problem.overcommit * new_def_x * (1 - attack)
            + problem.success * (1 - new_def_x) * attack
        )

        if tuple(new_def_x) not in solutions[0] and new_def_obj.getValue() >= true_objs[0].getValue() + phi_ub:
            model.addConstr(new_def_obj <= true_objs[0] + phi)
            solutions[0].add(tuple(new_def_x))
            new_constraint = True

        new_att_x = problem.solve_player(
            False, solution, payoffs=current_p)
        new_att_obj = p[1] @ (
            -problem.normal * (1 - defence) * (1 - new_att_x)
            + (1 - defence) * new_att_x
            + (1 - problem.mitigated) * defence * new_att_x
        )

        if tuple(new_att_x) not in solutions[1] and new_att_obj.getValue() >= true_objs[1].getValue() + phi_ub:
            model.addConstr(new_att_obj <= true_objs[1] + phi)
            solutions[1].add(tuple(new_att_x))
            new_constraint = True

        if not new_constraint:
            break

        model.optimize()
        if timelimit is not None:
            local_timelimit -= model.Runtime
            model.params.TimeLimit = max(1, local_timelimit)

        while model.Status == GRB.INFEASIBLE:
            phi_ub += 1
            phi.ub = phi_ub

            if max_phi is not None and phi_ub > max_phi:
                raise ValueError("Problem is Infeasible")
            model.optimize()

            if model.Status == GRB.TIME_LIMIT or \
                (timelimit is not None and time() - start >= timelimit):
                break
            
        if model.Status == GRB.TIME_LIMIT or \
            (timelimit is not None and time() - start >= timelimit):
            print("Timelimit reached")
            raise UserWarning("Timelimit reached")


    inverse = p.X

    model.close()

    return inverse.astype(int), phi_ub
