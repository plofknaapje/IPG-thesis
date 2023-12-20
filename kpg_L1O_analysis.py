import kpg
import numpy as np
import networkx as nx

prefix = "instances_kp/generated/"
file = prefix + "2-25-5-cij.txt"

master_kpg = kpg.KPG.read_file(file)
master_result = kpg.zero_regrets(master_kpg)
results = {}

# Run the problem while ignoring one item each time.
for i in range(master_kpg.m):
    if sum(master_result.X[:, i]) > 0:
        payoffs = master_kpg.payoffs.copy()
        payoffs[:, i] = 0
        interaction_coefs = master_kpg.interaction_coefs.copy()
        interaction_coefs[:, :, i] = 0
        weights = master_kpg.weights
        instance = kpg.KPG(weights, payoffs, interaction_coefs)
        instance.capacity = master_kpg.capacity
        results[i] = (kpg.zero_regrets(instance))
        print(results[i].ObjVal)

chosen_items = np.sum(result.X for _, result in results.items())
print(results.keys())