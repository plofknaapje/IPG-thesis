from time import time

import numpy as np

from methods.inverse_kpg import (
    generate_payoff_problems,
    generate_weight_problems,
    inverse_weights,
    inverse_payoffs,
)
from problems.base import ApproxOptions

start = time()
rng = np.random.default_rng(1)

approach = "payoffs"
options = ApproxOptions(allow_phi_ne=True, timelimit=10, allow_timelimit_reached=False)

if approach == "weights":
    weight_problems = generate_weight_problems(
        50,
        2,
        20,
        100,
        0.5,
        corr=True,
        neg_inter=False,
        approx_options=options,
        rng=rng,
        verbose=False,
    )
    print("Problem generation finished")

    values = weight_problems[0].weights

    inverse = inverse_weights(weight_problems, verbose=False)

elif approach == "payoffs":
    payoff_problems = generate_payoff_problems(
        100,
        2,
        10,
        100,
        [0.5, 0.5],
        corr=True,
        neg_inter=False,
        approx_options=options,
        rng=rng,
        verbose=False,
    )

    print("Problem generation finished")

    values = payoff_problems[0].payoffs

    inverse = inverse_payoffs(payoff_problems, verbose=False)

else:
    raise ValueError("Unknown approach!")

print(values)
print(inverse)

error = np.abs(inverse - values).sum()
print(error, error / values.sum())
print(time() - start)
