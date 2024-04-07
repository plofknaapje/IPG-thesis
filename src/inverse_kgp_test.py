from time import time

import numpy as np

from methods.inverse_kpg import *

start = time()
rng = np.random.default_rng(1)

approach = "weight"

match approach:
    case "weight":
        weight_problems = generate_weight_problems(
            50, 2, 20, 100, [0.5, 0.5], corr=True, rng=rng
        )
        print("Problem generation finished")

        values = weight_problems[0].weights

        inverse = inverse_weights(weight_problems, verbose=False)

    case "payoff":
        payoff_problems = generate_payoff_problems(
            100, 2, 10, 100, [0.5, 0.5], corr=False, rng=rng
        )

        print("Problem generation finished")

        values = payoff_problems[0].payoffs

        inverse = inverse_payoffs(payoff_problems, verbose=True)
    case _:
        raise ValueError("Unknown approach!")

print(values)
print(inverse)

error = np.abs(inverse - values).sum()
print(error, error / values.sum())
print(time() - start)
