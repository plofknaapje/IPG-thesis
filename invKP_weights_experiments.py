from int_invKP import *
import numpy as np
from time import time

rng = np.random.default_rng(42)

repetitions = 10
items = 50
problem_size = items * 3
r = 100
capacity = 0.5
corr = True

size_range = list(range(problem_size, items - 1, -10))
print(size_range)

problems = [generate_weight_problems(problem_size, items, r, capacity, corr, rng=rng) 
            for _ in range(repetitions)]

errors = np.zeros((repetitions, len(size_range)))
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
        errors[i, j] = error
        run_times[i, j] = run_time

print(errors)

print(run_times)
