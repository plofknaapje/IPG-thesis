from time import time
import os.path

import numpy as np
import pandas as pd

from methods.inverse_kpg import generate_weight_problems

from methods.local_inverse_kpg import (
    local_inverse_payoffs,
    local_inverse_weights,
)

rng = np.random.default_rng(42)

players = [2, 3]

ranges = [250, 500]

repeats = 30
neg_inter = True

mode = "payoffs"

if mode == "weights":
    ranges = [100, 5000]
    n_items = [100, 500, 1000, 5000]
elif mode == "payoffs":
    ranges = [250, 500]
    n_items = [50, 100, 250]

header = ["players", "items", "range", "avg", "sdev", "min", "max", "inf", "change"]
total_runs = len(players) * len(n_items) * len(ranges) * repeats

if mode == "weights":

    if os.path.isfile(f"./results/local_inverse_kpg-weights-{repeats}-{max(players)}-{neg_inter}.csv"):
        print("Already generated")

    else:
        weight_data = []
        runs = 0

        for n in players:
            for m in n_items:
                for r in ranges:

                    weight_results = np.zeros((repeats))
                    change = np.zeros((repeats))
                    w_inf = 0

                    for i in range(repeats):
                        print(n, m, r)
                        problem = generate_weight_problems(
                            size=1, n=n, m=m, r=r, capacity=0.5, neg_inter=neg_inter, rng=rng, solve=False)[0]
                        greedy_solution = problem.solve_greedy()

                        start = time()
                        try:
                            weights = local_inverse_weights(problem)
                            change[i] = np.abs(weights - problem.weights).sum() / problem.weights.sum()
                        except ValueError:
                            w_inf += 1

                        runtime = time() - start
                        print(runtime)
                        weight_results[i] = runtime

                        runs += 1

                        if runs % 10 == 0:
                            print(f"{runs} out of {total_runs} done!")

                    weight_data.append([n, m, r, np.mean(weight_results), np.std(weight_results),
                                        np.min(weight_results), np.max(weight_results), w_inf, np.mean(change)])

            print(f"{n} players done")

        weight_df = pd.DataFrame(weight_data, columns=header)
        weight_df.to_csv(
            f"./results/local_inverse_kpg-weights-{repeats}-{max(players)}-{neg_inter}.csv", float_format="%3.3f")

elif mode == "payoffs":
    if os.path.isfile(f"./results/local_inverse_kpg-payoffs-{repeats}-{max(players)}-{neg_inter}.csv"):
        print("Already generated")

    else:
        payoff_data = []

        runs = 0

        for n in players:
            for m in n_items:
                for r in ranges:

                    payoff_results = np.zeros((repeats))
                    p_inf = 0
                    change = np.zeros((repeats))

                    for i in range(repeats):
                        print(n, m, r)
                        problem = generate_weight_problems(
                            size=1, n=n, m=m, r=r, capacity=0.5, neg_inter=neg_inter, rng=rng, solve=False)[0]
                        greedy_solution = problem.solve_greedy()
                        start = time()
                        try:
                            payoffs, inter = local_inverse_payoffs(problem, timelimit=30)
                            change[i] = (np.abs(payoffs - problem.payoffs).sum() + np.abs(inter - problem.inter_coefs).sum()) / \
                                    (problem.payoffs.sum() + problem.inter_coefs.sum())
                        except ValueError:
                            p_inf += 1

                        runtime = time() - start
                        print(runtime)
                        payoff_results[i] = runtime

                        runs += 1

                        if runs % 10 == 0:
                            print(f"{runs} out of {total_runs} done!")

                    payoff_data.append([n, m, r, np.mean(payoff_results), np.std(payoff_results),
                                        np.min(payoff_results), np.max(payoff_results), p_inf, np.mean(change)])

            print(f"{n} players done")

        payoff_df = pd.DataFrame(payoff_data, columns=header)
        payoff_df.to_csv(
            f"./results/local_inverse_kpg-payoffs-{repeats}-{max(players)}-{neg_inter}.csv", float_format="%3.3f")
