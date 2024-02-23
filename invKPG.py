import gurobipy as gp
from gurobipy import GRB
import kpg
import numpy as np
from copy import copy
from utils import duplicate_array

def generate_problems(size: int, n: int, m: int, capacity: float, weight_type: str="sym", 
                      payoff_type: str="sym", interaction_type: str="sym") -> list[kpg.KPG]:
    # Generate KPG instances with the same payoff and interactions.
    # Payoffs and interactions are normalised
    problems = []
    players = list(range(n))

    match payoff_type:
        case "sym":
            payoff = np.random.randint(1, 101, m)
            payoffs = np.zeros((n, m))
            for p in players:
                payoffs[p, :] = payoff
        case "asym":
            payoffs = np.random.randint(1, 101, (n, m))
        case _:
            raise ValueError("Payoff type not recognised!")
    # payoffs = payoffs / payoffs.sum()

    match interaction_type:
        case "sym":
            coefs = np.random.randint(1, 101, (n, n))
            interaction_coefs = np.zeros((n, n, m))
            for j in range(m):
                interaction_coefs[:, :, j] = coefs
        case "asym":
            interaction_coefs = np.random.randint(1, 101, (n, n, m))
        case "negasym":
            interaction_coefs = np.random.randint(-100, 101, (n, n, m))
        case _:
            raise ValueError("Interaction type not recognised!")
    for p in players:
        interaction_coefs[p, p, :] = 0
    # How to deal with negative interactions?
    # interaction_coefs = interaction_coefs / np.abs(interaction_coefs).sum()
        
    for _ in range(size):
        match weight_type:
            case "sym":
                weight = np.random.randint(1, 101, m)
                weights = np.zeros((n, m))
                for p in players:
                    weights[p, :] = weight
            case "asym":
                weights = np.random.randint(1, 101, (n, m))
            case _:
                raise ValueError("Weight type not recognised!")
        problem = kpg.KPG(weights, payoffs, interaction_coefs, capacity)

        problem.interaction_type = interaction_type
        problem.weights_type = weight_type
        problem.payoffs_type = payoff_type

        problems.append(problem)

    return problems

def solve_problems(problems: list[kpg.KPG]) -> list:
    return [kpg.zero_regrets(problem) for problem in problems]

def solve_player_problem(obs: kpg.KPG, sol: np.ndarray, p: int, payoffs: np.ndarray, 
                         interaction_coefs: np.ndarray) -> np.ndarray:
    players = obs.players
    rivals = [[opp for opp in players if opp != player] for player in players]

    env = gp.Env()
    env.setParam("OutputFlag", 0)
    model = gp.Model("PlayerKPG")
    x = model.addMVar((obs.m), vtype=GRB.BINARY, name="x")

    base_payoff = payoffs[p, :] @ x 
    interactions = gp.quicksum(interaction_coefs[p, p2, i] * sol[p2, i] * x[i]
                                   for p2 in rivals[p] for i in range(obs.m))
    
    model.setObjective(base_payoff + interactions, GRB.MAXIMIZE)
    
    model.addConstr(obs.weights[p, :] @ x <= obs.capacity[p])

    model.optimize()

    return x.X



def inverse_KPG(obss: list[kpg.KPG], solutions: list[kpg.KPGResult]) -> tuple:
    # Learn the payoffs and interactions using solutions to problems with varying weights
    example = obss[0]
    players = example.players
    rivals = [[opp for opp in players if opp != player] for player in players]
    partial_sols = {(o, p): [] for o in range(len(obss)) for p in players}
    
    env = gp.Env()
    env.setParam("OutputFlag", 0)
    pm = gp.Model("InverseKPG")
    payoff = pm.addMVar(example.payoffs.shape, lb=1, name="payoff")
    inter = pm.addMVar(example.interaction_coefs.shape, name="inter")
    for p in players:
        for i in range(example.m):
            inter[p, p, i].lb = 0
            inter[p, p, i].ub = 0
    delta = pm.addMVar((len(obss), len(players)), name="delta")

    pm.setObjective(delta.sum())

    og_profit = {(o, p): (payoff[p, :] * solutions[o].X[p, :]).sum() +
                    gp.quicksum(inter[p, p2, :] * solutions[o].X[p2, :] * solutions[o].X[p, :] 
                                for p2 in rivals[p])
                    for o, _ in enumerate(obss) for p in players}
    
    pm.addConstrs(inter[p1, p2, 0] == inter[p1, p2, i] for i in range(1, example.m)
                  for p1 in example.players for p2 in example.players)
    
    pm.optimize()

    if pm.Status == GRB.INFEASIBLE:
        raise ValueError("Problem is Infeasible!")
    
    while True:    
        new_partial = False
        for o, obs in enumerate(obss):
            for p in players:
                sol = solutions[o].X
                new_x = solve_player_problem(obs, sol, p, payoff.X, inter.X)
                if not duplicate_array(partial_sols[o, p], new_x):
                    # new information
                    new_partial = True
                    # add to solutions
                    partial_sols[o, p].append(new_x)
                    # add to pm
                    new_profit = (payoff[p, :] * new_x).sum() + \
                                 gp.quicksum(inter[p, p1, :] * new_x * sol[p1, :]
                                             for p1 in rivals[p])
                    pm.addConstr(delta[o, p] >= new_profit - og_profit[o, p])
        
        if new_partial:
            pm.optimize()
        else:
            break

    return payoff.X, inter.X

if __name__ == "__main__":
    n = 2
    m = 10

    problems = generate_problems(n*m, n, m, 0.5)
    solutions = solve_problems(problems)
    example = problems[0]

    payoff, interaction = inverse_KPG(problems, solutions)

    print("Original")
    print(example.payoffs) 
    print(example.interaction_coefs)
    print(example.payoffs/example.payoffs.sum())
    print(example.interaction_coefs/example.interaction_coefs.sum())

    print("Inverse")
    print(payoff)
    print(interaction)