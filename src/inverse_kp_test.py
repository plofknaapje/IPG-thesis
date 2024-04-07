from time import time

import numpy as np

from methods.inverse_kp import *

start = time()
rng = np.random.default_rng(0)

approach = "weight"

match approach:
    case "weight":
        weight_problems = generate_weight_problems(size=60, m=30, rng=rng, corr=True)
        print("Finished generating problems")

        values = weight_problems[0].weights
        inverse = inverse_weights(weight_problems, verbose=True)

    case "payoff":
        payoff_problems = generate_payoff_problems(size=100, m=25, rng=rng)
        print("Finished generating problems")

        values = payoff_problems[0].payoffs
        # inverse = inverse_payoffs_direct(payoff_problems)
        inverse = inverse_payoffs_delta(payoff_problems)
        # inverse = inverse_payoffs_hybrid(payoff_problems)

print(values.sum())

error = np.abs(values - inverse).sum()
print(error, error / values.sum())
print(time() - start)
