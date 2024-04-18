import numpy as np

from methods.inverse_kpg import generate_weight_problems
from methods.local_inverse_kpg import local_inverse_weights, local_inverse_payoffs

rng = np.random.default_rng(1)

for _ in range(10):
    instance = generate_weight_problems(1, 2, 25, neg_inter=True, rng=rng, solve=False)[
        0
    ]

    new_weights = local_inverse_weights(instance)
    print(np.abs(new_weights - instance.weights))
    print(np.abs(new_weights - instance.weights).sum())

    new_payoffs = local_inverse_payoffs(instance)
    print(np.abs(new_payoffs - instance.payoffs))
    print(np.abs(new_payoffs - instance.payoffs).sum())
