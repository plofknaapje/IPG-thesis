from time import time
import os.path

import numpy as np
import pandas as pd

from methods.local_inverse_kp import (
    generate_problem,
    local_inverse_payoffs,
    local_inverse_weights,
)

rng = np.random.default_rng(42)
n_items = [100, 500, 1000, 5000, 10000]
ranges = [500, 1000, 5000]
repeats = 30

if os.path.isfile(f"./results/local_inverse_kp-payoffs-{repeats}.csv") and \
    os.path.isfile(f"./results/local_inverse_kp-weights-{repeats}.csv"):
    print("Already generated")

else:
    payoff_data = []
    weight_data = []

    header = ["items", "range", "avg", "sdev", "min", "max", "change"]

    total_runs = len(n_items) * len(ranges) * repeats
    runs = 0

    for n in n_items:
        for r in ranges:
            payoff_results = np.zeros((repeats))
            p_change = np.zeros((repeats))

            weight_results = np.zeros((repeats))
            w_change = np.zeros((repeats))

            for i in range(repeats):
                problem = generate_problem(n, r=r, capacity=0.5, rng=rng)
                greedy_solution = problem.solve_greedy()
                start = time()
                local_inverse_payoffs(problem)
                runtime = time() - start
                payoff_results[i] = runtime

                start = time()
                local_inverse_weights(problem)
                runtime = time() - start
                weight_results[i] = runtime

                runs += 1

                if runs % 10 == 0:
                    print(f"{runs} out of {total_runs} done!")

            payoff_data.append([n, r, np.mean(payoff_results), np.std(payoff_results),
                                np.min(payoff_results), np.max(payoff_results), np.mean(p_change)])
            weight_data.append([n, r, np.mean(weight_results), np.std(weight_results),
                                np.min(weight_results), np.max(weight_results), np.mean(w_change)])

        print(f"{n} items done")

    payoff_df = pd.DataFrame(payoff_data, columns=header)
    payoff_df.to_csv(f"./results/local_inverse_kp-payoffs-{repeats}.csv", float_format="%6.3f")

    weight_df = pd.DataFrame(weight_data, columns=header)
    weight_df.to_csv(f"./results/local_inverse_kp-weights-{repeats}.csv", float_format="%6.3f")
