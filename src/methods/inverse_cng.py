from typing import List, Optional

import numpy as np
from numpy.random import Generator
import gurobipy as gp
from gurobipy import GRB

from problems.critical_node_game import CriticalNodeGame, CNGParams
from problems.base import ApproxOptions


def generate_weight_problems(
    size: int,
    n: int,
    r=25,
    parameters: CNGParams | List[CNGParams] = None,
    approx_options: Optional[ApproxOptions] = None,
    rng: Optional[Generator] = None,
    verbose=False,
) -> List[CriticalNodeGame]:
    if rng is None:
        rng = np.random.default_rng()

    if approx_options is None:
        approx_options = ApproxOptions()

    if parameters is None:
        parameters = [CNGParams(rng=rng) for _ in range(size)]
    elif isinstance(parameters, CNGParams):
        parameters = [parameters for _ in range(size)]

    problems = []

    weights = rng.integers(1, r + 1, (2, n))

    while len(problems) < size:
        payoffs = weights + rng.integers(1, r + 1, (2, n))

        problem = CriticalNodeGame(weights, payoffs, parameters[len(problems)], rng)

        try:
            problem.solve(approx_options.timelimit, verbose)
        except UserWarning:
            print("Timelimit reached without useful solution")
            continue

        if approx_options.valid_problem(
            problem.result[0]
        ) and approx_options.valid_problem(problem.result[1]):
            problems.append(problem)
        else:
            print(
                f"Problem rejected, {approx_options.valid_problem(problem.result[0])}, {approx_options.valid_problem(problem.result[1])}"
            )
            print(problem.result[0])
            print(problem.result[1])

        if len(problems) != 0 and len(problems) % 10 == 0 and verbose:
            print(f"{len(problems)} problems generated.")

    return problems


def generate_payoff_problems(
    size: int,
    n: int,
    r=25,
    parameters: CNGParams | List[CNGParams] = None,
    approx_options: Optional[ApproxOptions] = None,
    rng: Optional[Generator] = None,
    verbose=False,
) -> List[CriticalNodeGame]:
    if rng is None:
        rng = np.random.default_rng()

    if approx_options is None:
        approx_options = ApproxOptions()

    if parameters is None:
        parameters = [CNGParams(rng=rng) for _ in range(size)]
    elif isinstance(parameters, CNGParams):
        parameters = [parameters for _ in range(size)]

    problems = []

    payoffs = rng.integers(1, r + 1, (2, n))

    while len(problems) < size:
        weights = payoffs + rng.integers(1, r + 1, (2, n))

        problem = CriticalNodeGame(weights, payoffs, parameters[len(problems)], rng)

        result = problem.solve(approx_options.timelimit, verbose)
        if approx_options.valid_problem(result[0]) and approx_options.valid_problem(
            result[1]
        ):
            problems.append(problem)

        if len(problems) != 0 and len(problems) % 10 == 0:
            print(f"{len(problems)} problems generated.")

    return problems


def generate_param_problems(
    size: int,
    n: int,
    parameters: CNGParams,
    r=25,
    approx_options: Optional[ApproxOptions] = None,
    rng: Optional[Generator] = None,
    verbose=False,
) -> List[CriticalNodeGame]:
    if rng is None:
        rng = np.random.default_rng()

    if approx_options is None:
        approx_options = ApproxOptions()

    if parameters is None:
        parameters = [CNGParams(rng=rng) for _ in range(size)]
    elif isinstance(parameters, CNGParams):
        parameters = [parameters for _ in range(size)]

    problems = []

    while len(problems) < size:
        weights = rng.integers(1, r + 1, (2, n))
        payoffs = weights + rng.integers(1, r + 1, (2, n))

        problem = CriticalNodeGame(weights, payoffs, parameters[len(problems)], rng)

        result = problem.solve(approx_options.timelimit, verbose)
        if approx_options.valid_problem(result[0]) and approx_options.valid_problem(
            result[1]
        ):
            problems.append(problem)

        if len(problems) != 0 and len(problems) % 10 == 0:
            print(f"{len(problems)} problems generated.")

    return problems


def inverse_payoffs(
    problems: List[CriticalNodeGame],
    sub_timelimit: Optional[int] = None,
    verbose=False,
) -> np.ndarray:
    n_problems = len(problems)
    n_items = problems[0].n  # number of items
    payoffs = problems[0].payoffs

    model = gp.Model("Inverse CNG (Payoffs)")

    delta = model.addMVar((n_problems, 2), name="delta")
    p = model.addMVar((2, n_items), vtype=GRB.INTEGER, lb=1, name="p")

    model.setObjective(delta.sum())

    model.addConstrs(p[j].sum() == payoffs[j].sum() for j in [0, 1])

    true_objs = {}

    for i, problem in enumerate(problems):
        defence = problem.solution[0][0]
        attack = problem.solution[0][1]
        true_objs[i, 0] = p[0] @ (
            (1 - defence) * (1 - attack)
            + problem.mitigated * defence * attack
            + problem.overcommit * defence * (1 - attack)
            + problem.success * (1 - defence) * attack
        )

        defence = problem.solution[1][0]
        attack = problem.solution[1][1]
        true_objs[i, 1] = p[1] @ (
            -problem.normal * (1 - defence) * (1 - attack)
            + (1 - defence) * attack
            + (1 - problem.mitigated) * defence * attack
        )

    new_constraint = True
    current_p = np.zeros_like(p)

    solutions = {(i, j): set() for i in range(n_problems) for j in [0, 1]}

    while new_constraint:
        model.optimize()

        if model.Status == GRB.INFEASIBLE:
            raise ValueError("Problem is Infeasible!")
        
        if np.array_equal(current_p, p.X):
            break

        new_constraint = False
        current_p = p.X

        if verbose:
            error = np.abs(current_p - payoffs).sum()
            print(error, error / payoffs.sum())

        for i, problem in enumerate(problems):
            # Defender
            attack = problem.solution[0][1]
            new_def_x = problem.solve_player(
                True, payoffs=current_p, timelimit=sub_timelimit
            )
            if tuple(new_def_x) not in solutions[i, 0]:
                new_def_obj = p[0] @ (
                    (1 - new_def_x) * (1 - attack)
                    + problem.mitigated * new_def_x * attack
                    + problem.overcommit * new_def_x * (1 - attack)
                    + problem.success * (1 - new_def_x) * attack
                )

                model.addConstr(delta[i, 0] >= new_def_obj - true_objs[i, 0])
                new_constraint = True

                solutions[i, 0].add(tuple(new_def_x))

            # Attacker
            defence = problem.solution[1][0]
            new_att_x = problem.solve_player(
                False, payoffs=p.X, timelimit=sub_timelimit
            )
            if tuple(new_att_x) not in solutions[i, 1]:
                new_att_obj = p[1] @ (
                    -problem.normal * (1 - defence) * (1 - new_att_x)
                    + (1 - defence) * new_att_x
                    + (1 - problem.mitigated) * defence * new_att_x
                )

                model.addConstr(delta[i, 1] >= new_att_obj - true_objs[i, 1])
                new_constraint = True

                solutions[i, 1].add(tuple(new_att_x))

    inverse = p.X

    model.close()

    return inverse.astype(int)


def inverse_weights(problems: List[CriticalNodeGame], verbose=True) -> np.ndarray:
    weights = problems[0].weights
    true_objs = [
        [problem.obj_value(True), problem.obj_value(False)] for problem in problems
    ]

    model = gp.Model("Inverse CNG (Weights)")

    w = model.addMVar((2, problems[0].n), vtype=GRB.INTEGER, lb=1)

    model.addConstrs(w[i].sum() == weights[i].sum() for i in [0, 1])

    for problem in problems:
        for i in [0, 1]:
            model.addConstr(w[i] @ problem.solution[0][i] <= problem.capacity[i])
            model.addConstr(w[i] @ problem.solution[1][i] <= problem.capacity[i])
    new_constraint = True

    current_w = np.zeros_like(w)

    while new_constraint:
        model.optimize()

        if model.Status == GRB.INFEASIBLE:
            raise ValueError("Problem is Infeasible!")

        if np.array_equal(current_w, w.X):
            break

        new_constraint = False
        current_w = w.X

        if verbose:
            error = np.abs(current_w - weights).sum()
            print(error, error / weights.sum())

        for i, problem in enumerate(problems):
            phi_def = problem.result[0].phi
            phi_att = problem.result[1].phi
            if phi_def != phi_att:
                print("Phis not the same")

            new_def_x = problem.solve_player(True, weights=current_w)
            if true_objs[i][0] + phi_def < problem.obj_value(True, def_sol=new_def_x):
                model.addConstr(w[0] @ new_def_x >= problem.capacity[0] + 0.5)
                new_constraint = True

            new_att_x = problem.solve_player(False, weights=current_w)
            if true_objs[i][0] + phi_att < problem.obj_value(False, att_sol=new_att_x):
                model.addConstr(w[1] @ new_att_x >= problem.capacity[1] + 0.5)
                new_constraint = True

    inverse = w.X

    model.close()

    return inverse.astype(int)


def inverse_params(problems: List[CriticalNodeGame], verbose=True) -> np.ndarray:
    n_problems = len(problems)
    true_params = problems[0].params()

    model = gp.Model("Inverse CNG (Payoffs)")

    delta = model.addMVar((n_problems, 2), name="delta")
    params = model.addMVar((4), lb=0.01, ub=0.99)
    success = params[0]
    mitigated = params[1]
    unchallanged = params[2]
    normal = params[3]

    model.setObjective(delta.sum())

    true_objs = {}

    for i, problem in enumerate(problems):
        defence = problem.solution[0][0]
        attack = problem.solution[0][1]
        true_objs[i, 0] = problem.payoffs @ (
            (1 - defence) * (1 - attack)
            + mitigated * defence * attack
            + unchallanged * defence * (1 - attack)
            + success * (1 - defence) * attack
        )

        defence = problem.solution[1][0]
        attack = problem.solution[1][1]
        true_objs[i, 1] = problem.payoffs @ (
            -normal * (1 - defence) * (1 - attack)
            + (1 - defence) * attack
            + (1 - mitigated) * defence * attack
        )

    model.addConstr(success * 1.001 <= mitigated)
    model.addConstr(mitigated * 1.001 <= unchallanged)

    new_constraint = True

    current_params = np.zeros_like(params)

    solutions = {(i, j): set() for i in range(n_problems) for j in [0, 1]}

    while new_constraint:
        model.optimize()

        if model.Status == GRB.INFEASIBLE:
            raise ValueError("Problem is Infeasible!")
        
        if np.array_equal(current_params, params.X):
            break

        current_params = params.X

        if verbose:
            print(params.X)
            error = np.abs(params.X - true_params).sum()
            print(error, error / true_params.sum())

        for i, problem in enumerate(problems):
            # Defender
            attack = problem.solution[0][1]
            new_def_x = problem.solve_player(True, params=current_params, timelimit=1)
            if tuple(new_def_x) not in solutions[i, 0]:
                new_def_obj = problem.payoffs @ (
                    (1 - new_def_x) * (1 - attack)
                    + mitigated * new_def_x * attack
                    + unchallanged * new_def_x * (1 - attack)
                    + success * (1 - new_def_x) * attack
                )

                model.addConstr(delta[i, 0] >= new_def_obj - true_objs[i, 0])
                new_constraint = True

                solutions[i, 0].add(tuple(new_def_x))

            # Attacker
            defence = problem.solution[1][0]
            new_att_x = problem.solve_player(False, params=current_params, timelimit=1)
            if tuple(new_att_x) not in solutions[i, 1]:
                new_att_obj = problem.payoffs @ (
                    -normal * (1 - defence) * (1 - new_att_x)
                    + (1 - defence) * new_att_x
                    + (1 - mitigated) * defence * new_att_x
                )

                model.addConstr(delta[i, 1] >= new_att_obj - true_objs[i, 1])
                new_constraint = True

                solutions[i, 1].add(tuple(new_att_x))

    inverse = params.X

    model.close()

    return inverse
