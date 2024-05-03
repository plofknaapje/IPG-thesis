import numpy as np
import pandas as pd
from time import time

from methods.local_inverse_kp import (
    generate_problem,
    local_inverse_payoffs,
    local_inverse_weights,
)

rng = np.random.default_rng(42)
n_items = [100, 500, 1000, 5000, 10000]
ranges = [1000, 5000, 10000]
repeats = 30

payoff_data = []
weight_data = []

for size in n_items:
    for r in ranges:
        payoff_results = []
        for _ in repeats:
            problem = generate_problem(5000, r=10000, capacity=0.5, rng=rng)
            greedy_solution = problem.solve_greedy()
            start = time()
            payoffs = local_inverse_payoffs(problem)
            end = time() - start
            payoff_results.append

payoffs = local_inverse_payoffs(problem)
print("Payoffs change:", np.abs(problem.payoffs - payoffs).sum())



weights = local_inverse_weights(problem)
print("Weights change:", np.abs(problem.weights - weights).sum())
