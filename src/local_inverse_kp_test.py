import numpy as np

from methods.local_inverse_kp import (
    generate_problem,
    local_inverse_payoffs,
    local_inverse_weights,
    local_inverse_payoffs_dynamic
)

problem = generate_problem(50, capacity=0.5)

greedy_solution = problem.solve_greedy()

payoffs = local_inverse_payoffs(problem)
print("Payoffs change:", np.abs(problem.payoffs - payoffs).sum())

payoffs = local_inverse_payoffs_dynamic(problem)
print("Payoffs change:", np.abs(problem.payoffs - payoffs).sum())

weights = local_inverse_weights(problem)
print("Weights change:", np.abs(problem.weights - weights).sum())
