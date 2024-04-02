import kpg
import pickle
import os

# Generate and store result dictionaries of instances using the leave-one-out
# method.

prefix = "instances_kp/generated/"
files = ["2-25-2-cij", "2-25-5-cij", "2-25-8-cij", "3-25-2-cij", "3-25-5-cij", "3-25-8-cij"]

for file in files:
    if os.path.isfile(f"pickles/{file}.pickle"):
        continue
    path = prefix + file + ".txt"
    master_kpg = kpg.read_file(path)
    master_result = kpg.zero_regrets(master_kpg)
    results = {"master": master_result}

    # Run the problem while ignoring one item each time.
    for i in range(master_kpg.m):
        if sum(master_result.X[:, i]) > 0:
            payoffs = master_kpg.payoffs.copy()
            payoffs[:, i] = 0
            interaction_coefs = master_kpg.inter_coefs.copy()
            interaction_coefs[:, :, i] = 0
            weights = master_kpg.weights
            instance = kpg.KPG(weights, payoffs, interaction_coefs, capacity=master_kpg.capacity)
            results[i] = (kpg.zero_regrets(instance))
            print(results[i].ObjVal)

    with open(f"pickles/{file}.pickle", "wb") as f:
        pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)


