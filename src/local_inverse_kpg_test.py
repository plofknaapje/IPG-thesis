import numpy as np

from problems.knapsack_packing_game import generate_random_KPG
from methods.local_inverse_kpg import local_inverse_weights

instance = generate_random_KPG()

solution = instance.solve_greedy()
print(solution)

new_weights = local_inverse_weights(instance)
print(np.abs(new_weights - instance.weights).sum())
