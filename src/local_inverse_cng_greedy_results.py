from time import time

import numpy as np
import pandas as pd

from problems.utils import rel_error
from problems.base import ApproxOptions
from problems.critical_node_game import CNGParams, generate_random_CNG
from methods.local_inverse_cng import local_inverse_weights, local_inverse_payoffs

repeats = 15

norms = [0, 0.1]
r = 25

n_nodes = [50, 70, 100, 150, 200]
mitigated = [0.6, 0.75]
cap = [0.3, 0.03]
payoff_cutoff = 100

weight_data = []
payoff_data = []

header = ["nodes", "range", "norm", "mit", "runtime", "pne", "phi", "diff"]

for n in n_nodes:
    timelimit = n
    rng = np.random.default_rng(n)  # Allows for generating partial results
    approx = ApproxOptions(allow_phi_ne=True, timelimit=200, allow_timelimit_reached=True)
    for norm in norms:
        for mit in mitigated:
            params = CNGParams(0.8 * mit, mit, 1.25 * mit, norm, capacity_perc=cap)
            weight_results = []
            weight_phi = []
            weight_times = []
            weight_pne = 0

            payoff_results = []
            payoff_phi = []
            payoff_times = []
            payoff_pne = 0

            print(n, norm, mit)

            for i in range(repeats):
                problem = generate_random_CNG(n, 25, params)
                weights = problem.weights
                payoffs = problem.payoffs
                sol = problem.solve_greedy(5)
                problem.solution = [sol, sol]

                # Weights
                print("Weights")
                weight_timelimit = timelimit
                start = time()
                phi = 0
                done = False
                while True:
                    weight_timelimit -= time() - start
                    if weight_timelimit <= 0:
                        weight_times.append(timelimit)
                        print("Timelimit reached")
                        break
                    try:
                        inverse_w = local_inverse_weights(problem, defender=True, phi=phi, timelimit=weight_timelimit)
                    except ValueError:
                        phi += 1
                    except UserWarning:
                        weight_times.append(timelimit)
                        print("Timelimit reached")
                        break
                    else:
                        weight_results.append(rel_error(weights, inverse_w))
                        weight_phi.append(phi)
                        weight_times.append(time() - start)
                        if phi == 0:
                            weight_pne += 1
                        break
                
                if n <= payoff_cutoff:
                    # Payoffs
                    print("Payoffs")
                    start = time()
                    try:
                        inverse_p, phi = local_inverse_payoffs(problem, defender=True, max_phi=None, timelimit=timelimit)
                    except ValueError:
                        print("Strange problem!")
                    except UserWarning:
                        payoff_times.append(timelimit)
                        print("Timelimit reached")
                    else:
                        if phi == 0:
                            payoff_pne += 1
                        payoff_results.append(rel_error(payoffs, inverse_p))
                        payoff_phi.append(phi)
                        payoff_times.append(time() - start)

                print(i)

            weight_data.append([n, r, norm, mit, np.mean(weight_times), weight_pne, np.mean(weight_phi), np.mean(weight_results)])
            print(weight_data[-1])
            if n <= payoff_cutoff:
                payoff_data.append([n, r, norm, mit, np.mean(payoff_times), payoff_pne, np.mean(payoff_phi), np.mean(payoff_results)])
                print(payoff_data[-1])

    print(f"{n} nodes finished!")

    weight_df = pd.DataFrame(weight_data, columns=header)
    weight_df.to_csv(f"results/local_inverse_cng-weights-{repeats}-{n}-nodes-greedy.csv", float_format="%6.3f", index=False)

    if n <= payoff_cutoff:
        payoff_df = pd.DataFrame(payoff_data, columns=header)
        payoff_df.to_csv(f"results/local_inverse_cng-payoffs-{repeats}-{n}-nodes-greedy.csv", float_format="%6.3f", index=False)

    weight_data = []
    payoff_data = []