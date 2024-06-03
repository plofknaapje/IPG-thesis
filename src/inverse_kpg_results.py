from time import time

import numpy as np
import pandas as pd

from methods.inverse_kpg import (
    generate_payoff_problems,
    generate_weight_problems,
    inverse_weights,
    inverse_payoffs,
)
from problems.base import ApproxOptions
from problems.utils import rel_error

columns = ["players", "items", "range", "obs", "runtime", "diff", "inf"]

results = []

approach = "weights"
repeats = 5

if approach == "weights":
    players = [2, 3]
    items = [10, 20]
    mults = [0.5, 1, 2, 4, 6]
    ranges = [100]
elif approach == "payoffs":
    players = [2, 3]
    items = [10, 20]
    mults = [1, 2, 4, 6, 8]
    ranges = [100]

if approach == "weights":
    options = ApproxOptions(allow_phi_ne=False, timelimit=10, allow_timelimit_reached=False)
    for n in players:
        rng = np.random.default_rng(n)

        for m in items:
            observations = [m * mult for mult in mults]
            for r in ranges:
                runtimes = [[] for _ in observations]
                diffs = [[] for _ in observations]
                for i in range(repeats):
                    problems = generate_weight_problems(
                        max(observations), n, m, r, 0.5, approx_options=options, rng=rng)
                    weights = problems[0].weights
                    for j, o in enumerate(observations):
                        start = time()
                        inverse = inverse_weights(problems[:o], timelimit=n*o/2, sub_timelimit=1)
                        diffs[j].append(rel_error(weights, inverse))
                        runtimes[j].append(time() - start)

                for j, o in enumerate(observations):
                    results.append([n, m, r, o, np.mean(runtimes[j]), np.mean(diffs[j])])
                    print(results[-1])

        df = pd.DataFrame(results, columns=columns)
        df.to_csv(f"results/inverse_kpg-weights-{repeats}-{n}-players.csv", float_format="%6.3f")
        results = []

elif approach == "payoffs":
    options = ApproxOptions(allow_phi_ne=True, timelimit=10, allow_timelimit_reached=False)

    for n in players:
        rng = np.random.default_rng(n)

        for m in items:
            observations = [m * mult for mult in mults]
            for r in ranges:
                runtimes = [[] for _ in observations]
                diffs = [[] for _ in observations]
                for i in range(repeats):
                    problems = generate_payoff_problems(
                        max(observations), n, m, r, 0.5, approx_options=options, rng=rng)
                    payoffs = problems[0].payoffs
                    for j, o in enumerate(observations):
                        start = time()
                        inverse = inverse_payoffs(problems[:o], timelimit=n*o/2, sub_timelimit=1)
                        diffs[j].append(rel_error(payoffs, inverse))
                        runtimes[j].append(time() - start)

                for j, o in enumerate(observations):
                    results.append([n, m, r, o, np.mean(runtimes[j]), np.mean(diffs[j])])
                    print(results[-1])

        df = pd.DataFrame(results, columns=columns)
        df.to_csv(f"results/kpg/inverse_kpg-payoffs-{repeats}-{n}-players.csv", float_format="%6.3f")
        results = []