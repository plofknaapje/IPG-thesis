import numpy as np

from methods.inverse_adg import generate_payoff_problems, generate_weight_problems, inverse_payoffs
from problems.base import ApproxOptions

rng = np.random.default_rng(42)

approx_options = ApproxOptions(allow_phi_ne=True, timelimit=10, allow_timelimit_reached=False)
approach = "payoffs"

# 10 seconds is too short for instances with 20 nodes.

if approach == "weights":
    problems = generate_weight_problems(50, 10, rng=rng, approx_options=approx_options)
    original = problems[0].weights
elif approach == "payoffs":
    problems = generate_payoff_problems(100, 10, rng=rng, approx_options=approx_options)
    original = problems[0].payoffs
    inverse = inverse_payoffs(problems, learn_defence=False, sub_timelimit=1, verbose=True)

else:
    raise ValueError("Invalid approach")

print(original)
print(inverse)
print(np.abs(inverse - original).sum())