import numpy as np
from numpy.random import Generator

from problems.attacker_defender_game import AttackerDefenderGame, ADGParams
from problems.base import ApproxOptions


def generate_weight_problems(
    size: int,
    n: int,
    r=25,
    parameters: ADGParams | list[ADGParams] | None = None,
    approx_options: ApproxOptions | None = None,
    rng: Generator | None = None,
    verbose=False,
) -> list[AttackerDefenderGame]:
    if rng is None:
        rng = np.random.default_rng()
    if approx_options is None:
        approx_options = ApproxOptions()

    problems = []

    weights = rng.integers(1, r + 1, (2, n))
    if parameters is None:
        parameters: list[ADGParams] = [
            ADGParams(None, None, None, None, None, rng) for _ in range(size)
        ]
    elif isinstance(parameters, ADGParams):
        parameters: list[ADGParams] = [parameters for _ in range(size)]

    while len(problems) < size:
        payoffs = weights + rng.integers(1, r + 1, (2, n))

        problem = AttackerDefenderGame(weights, payoffs, parameters[len(problems)], rng)

        result = problem.solve(verbose, approx_options.timelimit)
        if approx_options.valid_problem(result):
            problem.append(problem)

        if len(problems) != 0 and len(problems) % 10 == 0:
            print(f"{len(problems)} problems generated.")

    return problems


def generate_payoff_problems(
    size: int,
    n: int,
    r=25,
    parameters: ADGParams | list[ADGParams] | None = None,
    approx_options: ApproxOptions | None = None,
    rng: Generator | None = None,
    verbose=False,
) -> list[AttackerDefenderGame]:
    if rng is None:
        rng = np.random.default_rng()

    problems = []

    payoffs = r + rng.integers(1, r + 1, (2, n))

    if parameters is None:
        parameters: list[ADGParams] = [
            ADGParams(None, None, None, None, None, rng) for _ in range(size)
        ]
    elif isinstance(parameters, ADGParams):
        parameters: list[ADGParams] = [parameters for _ in range(size)]

    while len(problems) < size:
        weights = payoffs - rng.integers(1, r + 1, (2, n))

        problem = AttackerDefenderGame(weights, payoffs, parameters[len(problems)], rng)

        result = problem.solve(verbose, approx_options.timelimit)
        if approx_options.valid_problem(result):
            problem.append(problem)

        if len(problems) != 0 and len(problems) % 10 == 0:
            print(f"{len(problems)} problems generated.")

    return problems
