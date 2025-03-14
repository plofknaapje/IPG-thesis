from time import time

import numpy as np
import pandas as pd

from methods.inverse_kpg import (
    generate_payoff_problems,
    generate_weight_problems,
    inverse_weights,
    inverse_payoffs,
)
from problems.base import ApproxOptions, Target
from problems.utils import rel_error

columns = ["players", "cap", "obs", "runtime", "diff"]

results = []

approach = Target.PAYOFFS
repeats = 5

r = 100
m = 20
caps = [0.2, 0.4, 0.6, 0.8]

players = [2, 3]
mults = [1, 2, 4, 6, 8]


if approach is Target.WEIGHTS:
    for cap in caps:
        options = ApproxOptions(
            allow_phi_ne=False, timelimit=m, allow_timelimit_reached=False
        )
        rng = np.random.default_rng(m)
        for n in players:
            observations = [int(m * mult) for mult in mults]
            runtimes = [[] for _ in observations]
            diffs = [[] for _ in observations]
            for i in range(repeats):
                problems = generate_weight_problems(
                    max(observations), n, m, r=r, capacity=0.5, approx_options=options, rng=rng
                )
                weights = problems[0].weights
                for j, o in enumerate(observations):
                    start = time()
                    inverse = inverse_weights(
                        problems[:o], timelimit=n * o / 2, sub_timelimit=1
                    )
                    diffs[j].append(rel_error(weights, inverse))
                    runtimes[j].append(time() - start)

            for j, o in enumerate(observations):
                results.append(
                    [n, cap, o, np.mean(runtimes[j]), np.mean(diffs[j])]
                )
                print(results[-1])

    df = pd.DataFrame(results, columns=columns)
    df.to_csv(
        f"results/kpg/inverse_kpg-weights-{repeats}-{m}-items-{r}.csv",
        float_format="%6.3f",
        index=False,
    )

elif approach is Target.PAYOFFS:
    for phi in [True, False]:
        for cap in caps:
            options = ApproxOptions(
                allow_phi_ne=phi, timelimit=m, allow_timelimit_reached=False
            )
            if phi:
                rng = np.random.default_rng(m)
            else:
                rng = np.random.default_rng(m - 1)

            for n in players:
                observations = [int(m * mult) for mult in mults]
                runtimes = [[] for _ in observations]
                diffs = [[] for _ in observations]
                for i in range(repeats):
                    problems = generate_payoff_problems(
                        max(observations),
                        n,
                        m,
                        r=r,
                        capacity=0.5,
                        approx_options=options,
                        rng=rng,
                    )
                    payoffs = problems[0].payoffs
                    for j, o in enumerate(observations):
                        start = time()
                        inverse = inverse_payoffs(
                            problems[:o], timelimit=n * o / 2, sub_timelimit=1
                        )
                        diffs[j].append(rel_error(payoffs, inverse))
                        runtimes[j].append(time() - start)

                for j, o in enumerate(observations):
                    results.append(
                        [n, cap, o, np.mean(runtimes[j]), np.mean(diffs[j])]
                    )
                    print(results[-1])

            df = pd.DataFrame(results, columns=columns)
            df.to_csv(
                f"results/kpg/inverse_kpg-payoffs-{repeats}-{m}-items-{phi}-{r}.csv",
                float_format="%6.3f",
                index=False,
            )
            results = []
