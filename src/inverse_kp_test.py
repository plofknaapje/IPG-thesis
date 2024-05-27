from time import time

import numpy as np

from methods.inverse_kp import generate_payoff_problems, generate_weight_problems, inverse_payoffs_delta, inverse_weights

start = time()
rng = np.random.default_rng(0)

approach = "payoffs"

match approach:
    case "weights":
        weight_problems = generate_weight_problems(size=60, n=30, rng=rng, corr=True)
        print("Finished generating problems")

        values = weight_problems[0].weights
        inverse = inverse_weights(weight_problems, verbose=True)

    case "payoffs":
        payoff_problems = generate_payoff_problems(size=200, n=25, rng=rng, corr=False)
        print("Finished generating problems")

        values = payoff_problems[0].payoffs
        inverse = inverse_payoffs_delta(payoff_problems)

print(values.sum())

error = np.abs(values - inverse).sum()
print(error, error / values.sum())
print(time() - start)
