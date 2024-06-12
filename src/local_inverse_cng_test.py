from time import time

import numpy as np

from problems.base import ApproxOptions
from problems.critical_node_game import CNGParams, generate_random_CNG
from methods.inverse_cng import generate_weight_problems
from methods.local_inverse_cng import local_inverse_weights, local_inverse_payoffs


mitigated = 0.6
params = CNGParams(
    0.8 * mitigated, mitigated, 1.25 * mitigated, 0, capacity_perc=[0.3, 0.03]
)
approx = ApproxOptions(allow_phi_ne=True, timelimit=None, allow_timelimit_reached=False)

problem = generate_random_CNG(100, params=params)
sol = problem.solve_greedy()
problem.solution = [sol, sol]

print(sol[0])
print(sol[1])
print("Greedy solution done")

inverse_p, phi = local_inverse_payoffs(problem)
print(np.abs(inverse_p - problem.payoffs).sum() / problem.payoffs.sum())
