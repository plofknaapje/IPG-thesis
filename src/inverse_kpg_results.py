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

rng = np.random.default_rng(42)
options = ApproxOptions(allow_phi_ne=False, timelimit=10, allow_timelimit_reached=False)
columns = ["players", "items", "range", "obs", "runtime", "diff"]

results = []

approach = "weights"
repeats = 3

if approach == "weights":
    players = [2, 3]
    items = [10, 25]
    observations = [50, 75, 100]
    ranges = [100, 500]
elif approach == "payoffs":
    players = [2, 3]
    items = [10, 25]
    observations = [50, 100, 200]
    ranges = [100, 500]

if approach == "weights":
    options = ApproxOptions(allow_phi_ne=False, timelimit=10, allow_timelimit_reached=False)

    for n in players:
        for m in items:
            for r in ranges:
                runtimes = [[] for _ in observations]
                diffs = [[] for _ in observations]
                for i in range(repeats):
                    problems = generate_weight_problems(
                        max(observations), n, m, r, 0.5, approx_options=options, rng=rng)
                    weights = problems[0].weights
                    for j, o in enumerate(observations):
                        start = time()
                        inverse = inverse_weights(problems[:o], 1)
                        diffs[j].append(rel_error(weights, inverse))
                        runtimes[j].append(time() - start)

                for j, o in enumerate(observations):
                    results.append([n, m, r, o, np.mean(runtimes[j]), np.mean(diffs[j])])
                    print(results[-1])

    df = pd.DataFrame(results, columns=columns)
    df.to_csv(f"results/inverse_kpg-weights-{repeats}.csv", float_format="%6.3f")

elif approach == "payoffs":
    options = ApproxOptions(allow_phi_ne=True, timelimit=10, allow_timelimit_reached=False)

    for n in players:
        for m in items:
            for r in ranges:
                runtimes = [[] for _ in observations]
                diffs = [[] for _ in observations]
                for i in range(repeats):
                    problems = generate_payoff_problems(
                        max(observations), n, m, r, 0.5, approx_options=options, rng=rng)
                    payoffs = problems[0].payoffs
                    for j, o in enumerate(observations):
                        start = time()
                        inverse = inverse_payoffs(problems[:o], 1)
                        diffs[j].append(rel_error(payoffs, inverse))
                        runtimes[j].append(time() - start)

                for j, o in enumerate(observations):
                    results.append([n, m, r, o, np.mean(runtimes[j]), np.mean(diffs[j])])
                    print(results[-1])

    df = pd.DataFrame(results, columns=columns)
    df.to_csv(f"results/kpg/inverse_kpg-payoffs-{repeats}.csv", float_format="%6.3f")
