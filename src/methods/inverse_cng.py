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
    parameters: CNGParams | list[CNGParams] | None = None,
    approx_options: ApproxOptions | None = None,
    rng: Generator | None = None,
    verbose=False,
) -> list[CriticalNodeGame]:
    if rng is None:
        rng = np.random.default_rng()

    if approx_options is None:
        approx_options = ApproxOptions()

    if parameters is None:
        parameters: list[CNGParams] = [
            CNGParams(None, None, None, None, None, rng) for _ in range(size)
        ]
    elif isinstance(parameters, CNGParams):
        parameters: list[CNGParams] = [parameters for _ in range(size)]

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

        if approx_options.valid_problem(problem.result[0]) and approx_options.valid_problem(problem.result[1]):
            problems.append(problem)
        else:
            print(f"Problem rejected, {approx_options.valid_problem(problem.result[0])}, {approx_options.valid_problem(problem.result[1])}")
            print(problem.result[0])
            print(problem.result[1])

        if len(problems) != 0 and len(problems) % 10 == 0:
            print(f"{len(problems)} problems generated.")

    return problems


def generate_payoff_problems(
    size: int,
    n: int,
    r=25,
    parameters: CNGParams | list[CNGParams] | None = None,
    approx_options: ApproxOptions | None = None,
    rng: Generator | None = None,
    verbose=False,
) -> list[CriticalNodeGame]:
    if rng is None:
        rng = np.random.default_rng()

    if approx_options is None:
        approx_options = ApproxOptions()

    if parameters is None:
        parameters: list[CNGParams] = [
            CNGParams(None, None, None, None, None, rng) for _ in range(size)
        ]
    elif isinstance(parameters, CNGParams):
        parameters: list[CNGParams] = [parameters for _ in range(size)]

    problems = []

    payoffs = r + rng.integers(1, r + 1, (2, n))

    while len(problems) < size:
        weights = payoffs - rng.integers(1, r + 1, (2, n))

        problem = CriticalNodeGame(weights, payoffs, parameters[len(problems)], rng)

        result = problem.solve(approx_options.timelimit, verbose)
        if approx_options.valid_problem(result[0]) and approx_options.valid_problem(result[1]):
            problems.append(problem)

        if len(problems) != 0 and len(problems) % 10 == 0:
            print(f"{len(problems)} problems generated.")

    return problems


def inverse_payoffs(
    problems: list[CriticalNodeGame],
    learn_defence = True,
    learn_attack = True,
    sub_timelimit: int | None = None,
    verbose=False,
) -> np.ndarray:
    n_problems = len(problems)
    n_items = problems[0].n  # number of items

    model = gp.Model("Inverse CNG (Payoffs)")

    delta = model.addMVar((n_problems, 2), name="delta")
    p = model.addMVar((2, n_items), vtype=GRB.INTEGER, lb=1, name="p")

    if not learn_defence:
        for i in range(n_items):
            p[0, i].lb = problems[0].payoffs[0, i]
            p[0, i].ub = problems[0].payoffs[0, i]

    if not learn_attack:
        for i in range(n_items):
            p[1, i].lb = problems[1].payoffs[1, i]
            p[1, i].ub = problems[1].payoffs[1, i]


    model.setObjective(delta.sum())

    # model.addConstrs(p[j].sum() == problems[0].payoffs[j].sum() for j in [0, 1])

    true_objs = {}

    for i, problem in enumerate(problems):
        for index in [0, 1]:
            defence = problem.solution[index][0]
            attack = problem.solution[index][1]
            true_objs[i, 0] = p[0] @ (
                (1 - defence) * (1 - attack)
                + problem.mitigated * defence * attack
                + problem.overcommit * defence * (1 - attack)
                + problem.success * (1 - defence) * attack
            )

            true_objs[i, 1] = p[1] @ (
                -problem.normal * (1 - defence) * (1 - attack)
                + (1 - defence) * attack
                + (1 - problem.mitigated) * defence * attack
            )
            if np.array_equal(problem.solution[0], problem.solution[1]):
                break

    model.optimize()

    if model.Status == GRB.INFEASIBLE:
        raise ValueError("Problem is Infeasible!")

    solutions = {(i, j): set() for i in range(n_problems) for j in [0, 1]}

    while True:
        new_constraint = False
        current_p = p.X

        for i, problem in enumerate(problems):
            for index in [0, 1]:
                defence = problem.solution[index][0]
                attack = problem.solution[index][1]

                # Defender
                new_def_x = problem.solve_player(
                    True, problem.solution[index], payoffs=p.X, timelimit=sub_timelimit
                )
                if tuple(new_def_x) not in solutions[i, 0]:
                    new_def_obj = p[0] @ (
                        (1 - new_def_x) * (1 - attack)
                        + problem.mitigated * new_def_x * attack
                        + problem.overcommit * new_def_x * (1 - attack)
                        + problem.success * (1 - new_def_x) * attack
                    )

                    model.addConstr(delta[i, 0] >= new_def_obj - true_objs[i, 0] + problem.result.phi)
                    new_constraint = True

                    solutions[i, 0].add(tuple(new_def_x))

                # Attacker
                new_att_x = problem.solve_player(
                    False, problem.solution, payoffs=p.X, timelimit=sub_timelimit
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