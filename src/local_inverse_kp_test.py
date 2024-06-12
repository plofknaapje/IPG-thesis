import numpy as np
from time import time

from methods.local_inverse_kp import (
    generate_problem,
    local_inverse_payoffs,
    local_inverse_weights,
)

problem = generate_problem(25, r=100, capacity=0.5)

greedy_solution = problem.solve_greedy()

start = time()
payoffs = local_inverse_payoffs(problem)
print("Payoffs change:", np.abs(problem.payoffs - payoffs).sum())
print(time() - start)

start = time()
weights = local_inverse_weights(problem)
print("Weights change:", np.abs(problem.weights - weights).sum())
print(time() - start)
