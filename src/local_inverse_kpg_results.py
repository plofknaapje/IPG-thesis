from time import time

import numpy as np
import pandas as pd

from problems.utils import rel_error, abs_error
from methods.inverse_kpg import generate_weight_problems
from methods.local_inverse_kpg import (
    local_inverse_payoffs,
    local_inverse_weights,
)

rng = np.random.default_rng(42)

players = [2, 3, 4]

repeats = 30
neg_inter = True

mode = "weights"

if mode == "weights":
    ranges = [500, 1000]
    if neg_inter:
        n_items = [100, 500]
    else:
        n_items = [100, 500, 1000]
elif mode == "payoffs":
    ranges = [500, 1000]
    if neg_inter:
        n_items = [100, 500]
    else:
        n_items = [100, 500, 1000]

header = ["players", "items", "range", "avg", "sdev", "min", "max", "inf", "change"]
total_runs = len(players) * len(n_items) * len(ranges) * repeats

timelimit=60
print(mode)

if mode == "weights":
    weight_data = []
    runs = 0

    for n in players:
        for m in n_items:
            for r in ranges:

                weight_results = np.zeros((repeats))
                change = []
                w_inf = 0

                for i in range(repeats):
                    problem = generate_weight_problems(
                        size=1, n=n, m=m, r=r, capacity=0.5, neg_inter=neg_inter, rng=rng, solve=False)[0]
                    greedy_solution = problem.solve_greedy()

                    start = time()
                    try:
                        inverse_w = local_inverse_weights(problem, timelimit=timelimit)
                    except ValueError:
                        w_inf += 1
                        weight_results[i] = timelimit
                    else:
                        change.append(rel_error(problem.weights, inverse_w))

                        runtime = time() - start
                        weight_results[i] = runtime

                    runs += 1

                    if runs % 10 == 0:
                        print(f"{runs} out of {total_runs} done!")

                weight_data.append([n, m, r, np.mean(weight_results), np.std(weight_results),
                                    np.min(weight_results), np.max(weight_results), w_inf, np.mean(change)])

        print(f"{n} players done")

    weight_df = pd.DataFrame(weight_data, columns=header)
    weight_df.to_csv(
        f"./results/kpg/local_inverse_kpg-weights-{repeats}-{max(players)}-{neg_inter}.csv", float_format="%6.3f", index=False)

elif mode == "payoffs":
    payoff_data = []

    runs = 0

    for n in players:
        for m in n_items:
            for r in ranges:

                runtimes = np.zeros((repeats))
                p_inf = 0
                change = []

                for i in range(repeats):
                    problem = generate_weight_problems(
                        size=1, n=n, m=m, r=r, capacity=0.5, neg_inter=neg_inter, rng=rng, solve=False)[0]
                    greedy_solution = problem.solve_greedy()
                    start = time()
                    try:
                        inverse_p, inverse_i = local_inverse_payoffs(problem, timelimit=timelimit)
                    except ValueError:
                        p_inf += 1
                        runtimes[i] = timelimit
                    else:
                        change.append((abs_error(problem.payoffs, inverse_p) + abs_error(problem.inter_coefs, inverse_i)) / \
                                (problem.payoffs.sum() + problem.inter_coefs.sum()))
                        runtime = time() - start
                        runtimes[i] = runtime

                    runs += 1

                    if runs % 10 == 0:
                        print(f"{runs} out of {total_runs} done!")

                payoff_data.append([n, m, r, np.mean(runtimes), np.std(runtimes),
                                    np.min(runtimes), np.max(runtimes), p_inf, np.mean(change)])

        print(f"{n} players done")

    payoff_df = pd.DataFrame(payoff_data, columns=header)
    payoff_df.to_csv(
        f"./results/kpg/local_inverse_kpg-payoffs-{repeats}-{max(players)}-{neg_inter}.csv", float_format="%6.3f", index=False)
