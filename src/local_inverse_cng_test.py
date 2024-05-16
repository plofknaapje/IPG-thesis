import numpy as np

from problems.base import ApproxOptions
from problems.critical_node_game import CNGParams
from methods.inverse_cng import generate_weight_problems, generate_payoff_problems
from methods.local_inverse_cng import local_inverse_weights, local_inverse_payoffs


mitigated = 0.6
params = CNGParams(0.8*mitigated, mitigated, 1.25 * mitigated, 0.1, capacity_perc=[0.75, 0.1])
approx = ApproxOptions(True, 10, False)

problem = generate_weight_problems(1, 25, 25, params, approx)[0]
print(problem.result.PNE)
inverse_w = local_inverse_weights(problem)
print(np.abs(inverse_w - problem.weights).sum() / problem.weights.sum())

inverse_p = local_inverse_payoffs(problem)
print(np.abs(inverse_p - problem.payoffs).sum() / problem.payoffs.sum())