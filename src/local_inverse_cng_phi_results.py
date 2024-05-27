from time import time

import numpy as np
import pandas as pd

from problems.utils import rel_error
from problems.base import ApproxOptions
from problems.critical_node_game import CNGParams
from methods.inverse_cng import generate_weight_problems
from methods.local_inverse_cng import local_inverse_weights, local_inverse_payoffs

repeats = 5

norms = [0, 0.1]
r = 25

n_nodes = [20, 30, 40, 50]
mitigated = [0.6, 0.75]
cap = [0.3, 0.03]

weight_data = []
payoff_data = []

header = ["nodes", "range", "norm", "mit", "runtime", "pne", "phi", "diff"]

n_nodes = [50]

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
                duplicate_solution = True
                while True:
                    problem = generate_weight_problems(size=1, n=n, r=r, parameters=params, approx_options=approx, rng=rng)[0]
                    weights = problem.weights
                    payoffs = problem.payoffs

                    if problem.PNE:
                        print("Miracle, rejected!")
                        continue
                    elif problem.result[0].phi < 1 or problem.result[1].phi < 1:
                        print("Phi still 0")
                        continue
                    else:
                        break

                if np.array_equal(problem.solution[0], problem.solution[1]) >= 1:
                    print("Difference!")
                    duplicate_solution = False

                for defender in [True, False]:
                    if defender:
                        problem_phi = problem.result[0].phi
                    else:
                        problem_phi = problem.result[1].phi

                    print(f"Phi to beat is {problem_phi}")

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
                            if duplicate_solution:
                                weight_times.append(timelimit)
                            break

                        try:
                            inverse_w = local_inverse_weights(problem, defender=defender, phi=phi, timelimit=weight_timelimit)
                        except ValueError:
                            phi += 1
                        except UserWarning:
                            weight_times.append(timelimit)
                            print("Timelimit reached")
                            if duplicate_solution:
                                weight_times.append(timelimit)
                            break
                        else:
                            weight_results.append(rel_error(weights, inverse_w))
                            weight_phi.append((problem_phi - phi)/problem_phi)
                            weight_times.append(time() - start)
                            if phi == 0:
                                weight_pne += 1
                            if duplicate_solution:
                                weight_results.append(rel_error(weights, inverse_w))
                                weight_phi.append((problem_phi - phi)/problem_phi)
                                weight_times.append(time() - start)
                                if phi == 0:
                                    weight_pne += 1
                            break

                    print(problem_phi, phi)

                    # Payoffs
                    print("Payoffs")
                    start = time()
                    try:
                        inverse_p, phi = local_inverse_payoffs(problem, defender=defender, max_phi=problem_phi, timelimit=timelimit)
                    except ValueError:
                        print("Strange problem!")
                        break
                    except UserWarning:
                        payoff_times.append(timelimit)
                        print("Timelimit reached")
                        if duplicate_solution:
                            payoff_times.append(timelimit)
                    else:
                        payoff_results.append(rel_error(payoffs, inverse_p))
                        payoff_phi.append((problem_phi - phi)/problem_phi)
                        payoff_times.append(time() - start)
                        if phi == 0:
                            payoff_pne += 1
                        if duplicate_solution:
                            payoff_results.append(rel_error(payoffs, inverse_p))
                            payoff_phi.append((problem_phi - phi)/problem_phi)
                            payoff_times.append(time() - start)
                            if phi == 0:
                                payoff_pne += 1


                    if duplicate_solution:
                        break

                print(i)

            weight_data.append([n, r, norm, mit, np.mean(weight_times), weight_pne, np.mean(weight_phi), np.mean(weight_results)])
            print(weight_data[-1])
            payoff_data.append([n, r, norm, mit, np.mean(payoff_times), payoff_pne, np.mean(payoff_phi), np.mean(payoff_results)])
            print(payoff_data[-1])

    print(f"{n} nodes finished!")

    weight_df = pd.DataFrame(weight_data, columns=header)
    weight_df.to_csv(f"results/cng/local_inverse_cng-weights-{repeats}-{n}-nodes.csv", float_format="%6.3f", index=False)

    payoff_df = pd.DataFrame(payoff_data, columns=header)
    payoff_df.to_csv(f"results/cng/local_inverse_cng-payoffs-{repeats}-{n}-nodes.csv", float_format="%6.3f", index=False)

    weight_data = []
    payoff_data = []