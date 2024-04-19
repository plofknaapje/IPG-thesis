import numpy as np

from methods.inverse_adg import generate_payoff_problems, generate_weight_problems, inverse_payoffs
from problems.base import ApproxOptions

rng = np.random.default_rng(42)

approx_options = ApproxOptions(allow_phi_ne=False, timelimit=10, allow_timelimit_reached=False)
approach = "payoffs"

# 10 seconds is too short for instances with 20 nodes.

if approach == "weights":
    problems = generate_weight_problems(50, 10, rng=rng, approx_options=approx_options)

elif approach == "payoffs":
    problems = generate_payoff_problems(50, 10, rng=rng, approx_options=approx_options)
    inverse = inverse_payoffs(problems, learn_defence=False, sub_timelimit=1, verbose=True)

else:
    raise ValueError("Invalid approach")

# print(problems[0])
print(inverse)