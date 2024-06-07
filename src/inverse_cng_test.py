import numpy as np

from problems.critical_node_game import CNGParams
from methods.inverse_cng import generate_payoff_problems, generate_weight_problems, generate_param_problems, inverse_payoffs, inverse_params
from problems.base import ApproxOptions, Target

rng = np.random.default_rng(42)

mit = 0.75
norm = 0.1
cap = [0.3, 0.1]
params = CNGParams(0.8 * mit, mit, 1.25 * mit, norm, capacity_perc=cap)
approx_options = ApproxOptions(allow_phi_ne=True, timelimit=10, allow_timelimit_reached=False)
approach = Target.PARAMS

# 10 seconds is too short for instances with 20 nodes.

if approach is Target.WEIGHTS:
    problems = generate_weight_problems(50, 10, parameters=params, approx_options=approx_options, rng=rng)
    original = problems[0].weights
elif approach is Target.PAYOFFS:
    problems = generate_payoff_problems(100, 10, approx_options=approx_options, rng=rng)
    original = problems[0].payoffs
    inverse = inverse_payoffs(problems, defense=True, sub_timelimit=1, verbose=True)
elif approach is Target.PARAMS:
    problems = generate_param_problems(30, 10, params, approx_options=approx_options, rng=rng)
    original = params.to_array()
    inverse = inverse_params(problems)

print(original)
print(inverse)
print(np.abs(inverse - original).sum())