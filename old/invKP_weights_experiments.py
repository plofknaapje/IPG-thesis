from time import time

import numpy as np
import pandas as pd

from methods.inverse_kp import *

repetitions = 10
items = 25
problem_size = items * 3
r = 100
capacity = 0.5
corr = True
seed = 42

rng = np.random.default_rng(seed)
size_range = list(range(problem_size, items, -10))
print(size_range)


problems = [
    generate_weight_problems(problem_size, items, r, capacity, corr, rng=rng)
    for _ in range(repetitions)
]

abs_errors = np.zeros((repetitions, len(size_range)))
run_times = np.zeros((repetitions, len(size_range)))

for i, problem in enumerate(problems):
    for j, size in enumerate(size_range):
        print(f"Problem {i}, size {size}")
        problem_subset = problem[:size]
        start = time()
        values = problem_subset[0].weights
        inverse = inverse_weights(problem_subset)
        error = np.abs(values - inverse).sum()
        run_time = time() - start
        print(error, run_time)
        abs_errors[i, j] = error
        run_times[i, j] = int(run_time)

print(errors)
print(run_times)

abs_errors_df = pd.DataFrame(abs_errors)
abs_errors_df.columns = size_range
abs_errors_df["value_sum"] = [problem[0].weights.sum() for problem in problems]
run_times_df = pd.DataFrame(run_times)

abs_errors_df.to_csv(
    f"data/errors-{repetitions}-{items}-{problem_size}-{r}-{capacity}-{corr}-{seed}.csv",
    index=False,
)
run_times_df.to_csv(
    f"data/run_times-{repetitions}-{items}-{problem_size}-{r}-{capacity}-{corr}-{seed}.csv",
    index=False,
)
