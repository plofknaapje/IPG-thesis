import numpy as np

from problems.critical_node_game import CNGParams
from methods.inverse_cng import (
    generate_payoff_problems,
    generate_weight_problems,
    generate_param_problems,
    inverse_payoffs,
    inverse_params,
    inverse_weights,
)
from problems.base import ApproxOptions, Target

rng = np.random.default_rng(42)

mit = 0.75
norm = 0.1
cap = [0.5, 0.1]
params = CNGParams(
    success=0.8 * mit,
    mitigated=mit,
    unchallenged=1.25 * mit,
    normal=norm,
    capacity_perc=cap,
)
approx_options = ApproxOptions(
    allow_phi_ne=True, timelimit=10, allow_timelimit_reached=False
)
approach = Target.WEIGHTS


if approach is Target.WEIGHTS:
    problems = generate_weight_problems(
        50, 10, parameters=params, approx_options=approx_options, rng=rng
    )
    original = problems[0].weights
    inverse = inverse_weights(problems)
elif approach is Target.PAYOFFS:
    problems = generate_payoff_problems(200, 10, approx_options=approx_options, rng=rng)
    original = problems[0].payoffs
    inverse = inverse_payoffs(problems)
elif approach is Target.PARAMS:
    problems = generate_param_problems(
        10, 10, params, approx_options=approx_options, rng=rng
    )
    original = params.to_array()
    inverse = inverse_params(problems)

print(original)
print(inverse)
print(np.abs(inverse[0] - original[0]).sum() / original[0].sum())
print(np.abs(inverse[1] - original[1]).sum() / original[1].sum())
