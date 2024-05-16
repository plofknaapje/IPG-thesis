import numpy as np
import gurobipy as gp
from gurobipy import GRB

from problems.critical_node_game import CriticalNodeGame

eps = 0.00001

def local_inverse_weights(problem: CriticalNodeGame) -> np.ndarray:
    i_range = list(range(problem.n))
    solution = problem.solution
    obj_values = [problem.obj_value(True, solution[0], solution[1]),
                  problem.obj_value(False, solution[0], solution[1])]

    model = gp.Model("Local Inverse Weights")

    w = model.addMVar((2, problem.n), vtype=GRB.INTEGER, lb=1)

    delta = model.addMVar((2, problem.n))

    model.setObjective(delta.sum())

    model.addConstrs(
        w[j] @ solution[j] <= problem.capacity[j] for j in [0,1]
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
    if model.Status == GRB.INFEASIBLE:
        raise ValueError("Problem is initially infeasible")

    while True:
        new_constraint = False
        current_w = w.X

        new_def_x = problem.solve_player(True, solution, current_w)
        new_def_obj = problem.obj_value(True, new_def_x, solution[1])

        if new_def_obj >= obj_values[0] + eps:
            model.addConstr(new_def_x @ w[0] >= problem.capacity[0] + eps)
            new_constraint = True

        new_att_x = problem.solve_player(False, solution, current_w)
        new_att_obj = problem.obj_value(False, solution[0], new_att_x)

        if new_att_obj >= obj_values[1] + eps:
            model.addConstr(new_att_x @ w[1] >= problem.capacity[1] + eps)
            new_constraint = True

        if not new_constraint:
            break

        model.optimize()

        if model.Status == GRB.INFEASIBLE:
            raise ValueError("Problem is Infeasible")

        if np.array_equal(current_w, w.X):
            break

    inverse = w.X

    model.close()

    return inverse.astype(int)


def local_inverse_payoffs(problem: CriticalNodeGame, sub_timelimit: int | None = None) -> np.ndarray:
    i_range = list(range(problem.n))
    solution = problem.solution
    defence = solution[0]
    attack = solution[1]

    model = gp.Model("Local Inverse Payoffs")

    p = model.addMVar((2, problem.n), vtype=GRB.INTEGER, lb=1)

    delta = model.addMVar((2, problem.n))

    model.setObjective(delta.sum())

    model.addConstrs(
        delta[j, i] >= p[j, i] - problem.weights[j, i]
        for i in i_range
        for j in [0, 1]
    )

    model.addConstrs(
        delta[j, i] >= problem.weights[j, i] - p[j, i]
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
    if model.Status == GRB.INFEASIBLE:
        raise ValueError("Problem is initially Infeasible")


    while True:
        new_constraint = False
        current_p = p.X
        print(current_p)

        new_def_x = problem.solve_player(True, solution, payoffs=current_p, timelimit=sub_timelimit)
        new_def_obj = p[0] @ (
                    (1 - new_def_x) * (1 - attack)
                    + problem.mitigated * new_def_x * attack
                    + problem.overcommit * new_def_x * (1 - attack)
                    + problem.success * (1 - new_def_x) * attack
                )
        if new_def_obj.getValue() >= true_objs[0].getValue():
            model.addConstr(new_def_obj <= true_objs[0])
            new_constraint = True

        new_att_x = problem.solve_player(False, solution, payoffs=current_p, timelimit=sub_timelimit)
        new_att_obj = p[1] @ (
                    -problem.normal * (1 - defence) * (1 - new_att_x)
                    + (1 - defence) * new_att_x
                    + (1 - problem.mitigated) * defence * new_att_x
                )

        if new_att_obj.getValue() >= true_objs[1].getValue():
            model.addConstr(new_att_obj <= true_objs[1])
            new_constraint = True

        if not new_constraint:
            break

        model.optimize()

        if model.Status == GRB.INFEASIBLE:
            raise ValueError("Problem is Infeasible")

    inverse = p.X

    model.close()

    return inverse.astype(int)