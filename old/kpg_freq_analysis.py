import kpg
import numpy as np

n = 2
m = 25
games = 50
np.random.seed(42)
payoffs = np.random.randint(1, 101, (n, m))
weights = [np.random.randint(1, 101, (n, m)) for _ in range(games)]
interaction_coefs = np.zeros((n, n, m))
coefs = np.random.randint(1, 101, (n, n))
for j in range(m):
    interaction_coefs[:, :, j] = coefs
for p in range(n):
    interaction_coefs[p, p, :] = 0

player_results = np.zeros((n, m))
scores = []

for i in range(games):
    instance = kpg.KPG(weights[i], payoffs, interaction_coefs, capacity=0.5)
    result = kpg.zero_regrets(instance)
    player_results += np.divide(result.X, weights[i])
    scores.append(result.ObjVal)

print(player_results)
print(scores)