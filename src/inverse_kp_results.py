from time import time

import numpy as np
import pandas as pd

from methods.inverse_kp import (
    generate_payoff_problems, generate_weight_problems, inverse_payoffs_delta, inverse_weights)
from problems.utils import rel_error


rng = np.random.default_rng(0)
repeats = 5

columns = ["n", "r", "o", "runtime", "error"]

ranges = [100, 500]
n_items = [100, 500]

results = []

approach = "payoffs"

if approach == "weights":
    weight_problems = generate_weight_problems(size=100, n=30, rng=rng, corr=True)
    print("Finished generating problems")

    values = weight_problems[0].weights
    inverse = inverse_weights(weight_problems, verbose=True)

elif approach == "payoffs":
    for n in n_items:
        observations = [int(n), int(2*n), int(3*n)]
        for r in ranges:
            runtimes = [[] for _ in observations]
            error = [[] for _ in observations]
            for i in range(repeats):

                payoff_problems = generate_payoff_problems(size=observations[-1], n=n, r=r, capacity=0.5, rng=rng)
                payoffs = payoff_problems[0].payoffs

                for j, o in enumerate(observations):
                    start = time()
                    inverse = inverse_payoffs_delta(payoff_problems[:o])
                    end = time() - start
                    runtimes[j].append(end)
                    error[j].append(rel_error(payoffs, inverse))

            for j, o in enumerate(observations):
                results.append([n, r, o, np.mean(runtimes[j]), np.mean(error[j])])
                print(results[-1])

df = pd.DataFrame(results, columns=columns)
df.to_csv(f"./results/kp/inverse_kp-{approach}-{repeats}.csv")