from time import time

import numpy as np
import pandas as pd

from methods.inverse_kp import (
    generate_payoff_problems,
    generate_weight_problems,
    inverse_payoffs_delta,
    inverse_weights,
)
from problems.utils import rel_error
from problems.base import Target


repeats = 5

columns = ["n", "r", "o", "runtime", "error"]

approach = Target.PAYOFFS
print(approach)

if approach is Target.WEIGHTS:
    ranges = [100]
    n_items = [20, 40, 60]
    mults = [1, 2, 3, 4]
elif approach is Target.PAYOFFS:
    ranges = [100]
    n_items = [20, 40, 60]
    mults = [0.5, 1, 2, 4, 6, 8]

results = []

if approach is Target.WEIGHTS:
    for n in n_items:
        rng = np.random.default_rng(n)
        observations = [int(n * mult) for mult in mults]
        for r in ranges:
            runtimes = [[] for _ in observations]
            error = [[] for _ in observations]
            for i in range(repeats):
                print(f"Problem {i}")
                problems = generate_weight_problems(
                    size=observations[-1], n=n, r=r, capacity=0.5, corr=True, rng=rng
                )
                weights = problems[0].weights

                for j, o in enumerate(observations):
                    start = time()
                    inverse = inverse_weights(
                        problems[:o], timelimit=o / 2, verbose=False
                    )
                    end = time() - start
                    runtimes[j].append(end)
                    error[j].append(rel_error(weights, inverse))
                    print(o, "done")

            for j, o in enumerate(observations):
                results.append([n, r, o, np.mean(runtimes[j]), np.mean(error[j])])
                print(results[-1])

        df = pd.DataFrame(results, columns=columns)
        df.to_csv(
            f"./results/kp/inverse_kp-weights-{repeats}-{n}-items.csv",
            float_format="%6.3f",
            index=False,
        )
        results = []

elif approach is Target.PAYOFFS:
    for n in n_items:
        rng = np.random.default_rng(n)
        observations = [int(n * mult) for mult in mults]
        for r in ranges:
            runtimes = [[] for _ in observations]
            error = [[] for _ in observations]
            for i in range(repeats):
                print("Problem", i)
                problems = generate_payoff_problems(
                    size=observations[-1], n=n, r=r, capacity=0.5, corr=True, rng=rng
                )
                payoffs = problems[0].payoffs

                for j, o in enumerate(observations):
                    start = time()
                    inverse = inverse_payoffs_delta(
                        problems[:o], timelimit=o / 2, verbose=False
                    )
                    end = time() - start
                    runtimes[j].append(end)
                    error[j].append(rel_error(payoffs, inverse))

            for j, o in enumerate(observations):
                results.append([n, r, o, np.mean(runtimes[j]), np.mean(error[j])])
                print(results[-1])

        df = pd.DataFrame(results, columns=columns)
        df.to_csv(
            f"./results/kp/inverse_kp-payoffs-{repeats}-{n}-items.csv",
            float_format="%6.3f",
            index=False,
        )
        results = []
