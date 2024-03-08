import gurobipy as gp
from gurobipy import GRB
import kpg
import numpy as np
from copy import copy
from utils import duplicate_array

def generate_payoff_problems(size: int, n: int, m: int, capacity: float, weight_type: str="sym", 
                      payoff_type: str="sym", interaction_type: str="sym") -> list[kpg.KPG]:
    # Generate KPG instances with the same payoff and interactions.
    # Payoffs are in range 1, m
    # Interactions are picked from same range
    problems = []
    players = list(range(n))

    payoffs = np.zeros((n, m))
    match payoff_type:
        case "sym":
            payoff = np.arange(m) + 1
            np.random.shuffle(payoff)
            for p in players:
                payoffs[p, :] = payoff
        case "asym":
            for p in players:
                payoff = np.arange(m) + 1
                np.random.shuffle(payoff)
                payoffs[p, :] = payoff
        case _:
            raise ValueError("Payoff type not recognised!")
    # payoffs = payoffs / payoffs.sum()

    match interaction_type:
        case "none":
            interaction_coefs = np.zeros((n, n, m))
        case "sym":
            coefs = np.random.randint(1, m + 1, (n, n))
            interaction_coefs = np.zeros((n, n, m))
            for j in range(m):
                interaction_coefs[:, :, j] = coefs
        case "asym":
            interaction_coefs = np.random.randint(1, m+1, (n, n, m))
        case "negasym":
            interaction_coefs = np.random.randint(-m, m+1, (n, n, m))
        case _:
            raise ValueError("Interaction type not recognised!")
    for p in players:
        interaction_coefs[p, p, :] = 0  
    
    for _ in range(size):
        weights = np.zeros((n, m))
        match weight_type:
            case "sym":
                weight = np.arange(m) + 1
                np.random.shuffle(weight)
                for p in players:
                    weights[p, :] = weight
            case "asym":
                for p in players:
                    weight = np.arange(m) + 1
                    np.random.shuffle(weight)
                    weights[p, :] = weight
            case _:
                raise ValueError("Weight type not recognised!")
        problem = kpg.KPG(weights, payoffs, interaction_coefs, capacity)

        problem.interaction_type = interaction_type
        problem.weights_type = weight_type
        problem.payoffs_type = payoff_type

        problems.append(problem)

    return problems

def generate_weight_problems(size: int, n: int, m: int, capacity: float, weight_type: str="sym", 
                      payoff_type: str="sym", interaction_type: str="sym") -> list[kpg.KPG]:
    # Generate KPG instances with the same payoff and interactions.
    # Payoffs are in range 1, m
    # Interactions are picked from same range
    problems = []
    players = list(range(n))


    weights = np.zeros((n, m))
    match weight_type:
        case "sym":
            weight = np.arange(m) + 1
            np.random.shuffle(weight)
            for p in players:
                weights[p, :] = weight
        case "asym":
            for p in players:
                weight = np.arange(m) + 1
                np.random.shuffle(weight)
                weights[p, :] = weight
        case _:
            raise ValueError("Weight type not recognised!")
        
    
    for _ in range(size):

        payoffs = np.zeros((n, m))
        match payoff_type:
            case "sym":
                payoff = np.arange(m) + 1
                np.random.shuffle(payoff)
                for p in players:
                    payoffs[p, :] = payoff
            case "asym":
                for p in players:
                    payoff = np.arange(m) + 1
                    np.random.shuffle(payoff)
                    payoffs[p, :] = payoff
            case _:
                raise ValueError("Payoff type not recognised!")
        # payoffs = payoffs / payoffs.sum()

        match interaction_type:
            case "none":
                interaction_coefs = np.zeros((n, n, m))
            case "sym":
                coefs = np.random.randint(1, m + 1, (n, n))
                interaction_coefs = np.zeros((n, n, m))
                for j in range(m):
                    interaction_coefs[:, :, j] = coefs
            case "asym":
                interaction_coefs = np.random.randint(1, m+1, (n, n, m))
            case "negasym":
                interaction_coefs = np.random.randint(-m, m+1, (n, n, m))
            case _:
                raise ValueError("Interaction type not recognised!")
        for p in players:
            interaction_coefs[p, p, :] = 0
        
        problem = kpg.KPG(weights, payoffs, interaction_coefs, capacity)

        problem.interaction_type = interaction_type
        problem.weights_type = weight_type
        problem.payoffs_type = payoff_type

        problems.append(problem)

    return problems

def solve_problems(problems: list[kpg.KPG]) -> list:
    return [kpg.zero_regrets(problem) for problem in problems]

def solve_player_payoff_problem(obs: kpg.KPG, sol: np.ndarray, p: int, payoffs: np.ndarray, 
                         interaction_coefs: np.ndarray) -> np.ndarray:
    players = obs.players
    rivals = [[opp for opp in players if opp != player] for player in players]

    env = gp.Env()
    env.setParam("OutputFlag", 0)
    model = gp.Model("PlayerKPG")
    x = model.addMVar((obs.m), vtype=GRB.BINARY, name="x")

    base_payoff = payoffs[p, :] @ x 
    interactions = gp.quicksum(interaction_coefs[p, p2, :] * sol[p2, :] @ x
                                   for p2 in rivals[p])
    
    model.setObjective(base_payoff + interactions, GRB.MAXIMIZE)
    
    model.addConstr(obs.weights[p, :] @ x <= obs.capacity[p])

    model.optimize()

    return x.X

def solve_player_weights_problem(obs: kpg.KPG, sol: np.ndarray, p: int, weights: np.ndarray) -> np.ndarray:
    players = obs.players
    rivals = [[opp for opp in players if opp != player] for player in players]

    env = gp.Env()
    env.setParam("OutputFlag", 0)
    model = gp.Model("PlayerKPG")
    x = model.addMVar((obs.m), vtype=GRB.BINARY, name="x")

    base_payoff = obs.payoffs[p, :] @ x 
    interactions = gp.quicksum(obs.interaction_coefs[p, p2, :] * sol[p2, :] @ x
                                   for p2 in rivals[p])
    
    model.setObjective(base_payoff + interactions, GRB.MAXIMIZE)
    
    model.addConstr(weights[p, :] @ x <= obs.capacity[p])

    model.optimize()

    return x.X

def inverse_payoff_KPG(obss: list[kpg.KPG], solutions: list[kpg.KPGResult]) -> tuple:
    # Learn the payoffs and interactions using solutions to problems with varying weights
    example = obss[0]
    players = example.players
    rivals = [[opp for opp in players if opp != player] for player in players]
    partial_sols = {(o, p): [] for o in range(len(obss)) for p in players}
    
    env = gp.Env()
    env.setParam("OutputFlag", 0)
    pm = gp.Model("InverseKPG")
    payoff = pm.addMVar(example.payoffs.shape, vtype=GRB.INTEGER, name="payoff")
    inter = pm.addMVar(example.interaction_coefs.shape, vtype=GRB.INTEGER, name="inter")
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
    
    pm.addConstrs(payoff[p, :].sum() == (example.m * example.m + example.m) / 2 for p in players)

    pm.optimize()

    if pm.Status == GRB.INFEASIBLE:
        raise ValueError("Problem is Infeasible!")
    
    while True:    
        new_partial = False
        for o, obs in enumerate(obss):
            for p in players:
                sol = solutions[o].X
                new_x = solve_player_payoff_problem(obs, sol, p, payoff.X, inter.X)
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

def inverse_range_payoff_KPG(obss: list[kpg.KPG], solutions: list[kpg.KPGResult]) -> tuple:
    # Learn the payoffs and interactions using solutions to problems with varying weights
    example = obss[0]
    players = example.players
    items = list(range(example.m))
    rivals = [[opp for opp in players if opp != player] for player in players]
    partial_sols = {(o, p): [] for o in range(len(obss)) for p in players}
    
    env = gp.Env()
    env.setParam("OutputFlag", 0)
    pm = gp.Model("InverseRangeKPG")
    bin_payoff = pm.addMVar((example.n, example.m, example.m), vtype=GRB.BINARY, name="bin_payoff")
    payoff = pm.addMVar(example.payoffs.shape, vtype=GRB.INTEGER, name="payoff")
    inter = pm.addMVar(example.interaction_coefs.shape, vtype=GRB.INTEGER, name="inter")
    for p in players:
        for i in items:
            inter[p, p, i].lb = 0
            inter[p, p, i].ub = 0
    delta = pm.addMVar((len(obss), len(players)), name="delta")

    pm.setObjective(delta.sum())

    og_profit = {(o, p): (payoff[p, :] * solutions[o].X[p, :]).sum() +
                    gp.quicksum(inter[p, p2, :] * solutions[o].X[p2, :] * solutions[o].X[p, :] 
                                for p2 in rivals[p])
                    for o, _ in enumerate(obss) for p in players}
    
    pm.addConstrs(bin_payoff[p, i, :].sum() == 1
                  for p in players for i in items)
    
    pm.addConstrs(bin_payoff[p, :, i].sum() == 1
                  for p in players for i in items)
    
    pm.addConstrs(payoff[p, i] == gp.quicksum(bin_payoff[p, i, val] * (val + 1) for val in items)
              for p in players for i in items)
    
    pm.optimize()

    if pm.Status == GRB.INFEASIBLE:
        raise ValueError("Problem is Infeasible!")
    
    while True:    
        new_partial = False
        for o, obs in enumerate(obss):
            for p in players:
                sol = solutions[o].X
                new_x = solve_player_payoff_problem(obs, sol, p, payoff.X, inter.X)
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


def inverse_weight_KPG(obss: list[kpg.KPG], solutions: list[kpg.KPGResult]) -> tuple:

    example = obss[0]
    players = example.players
    items = list(range(example.m))
    rivals = [[opp for opp in players if opp != player] for player in players]
    partial_sols = {(o, p): [] for o in range(len(obss)) for p in players}
    
    env = gp.Env()
    env.setParam("OutputFlag", 0)
    pm = gp.Model("InverseWeightsKPG")
    bin_weights = pm.addMVar((example.n, example.m, example.m), vtype=GRB.BINARY, name="bin_payoff")
    weights = pm.addMVar(example.weights.shape, vtype=GRB.INTEGER, name="payoff")
    delta = pm.addMVar((len(obss), len(players)), name="delta")

    pm.setObjective(delta.sum())

    og_profit = {(o, p): (obs.payoffs[p, :] * solutions[o].X[p, :]).sum() +
                    sum(obs.interaction_coefs[p, p2, :] * solutions[o].X[p2, :] * solutions[o].X[p, :] 
                                for p2 in rivals[p])
                    for o, obs in enumerate(obss) for p in players}
    
    pm.addConstrs(bin_weights[p, i, :].sum() == 1
                  for p in players for i in items)
    
    pm.addConstrs(bin_weights[p, :, i].sum() == 1
                  for p in players for i in items)
    
    pm.addConstrs(weights[p, i] == gp.quicksum(bin_weights[p, i, val] * (val + 1) for val in items)
              for p in players for i in items)
    
    pm.optimize()

    if pm.Status == GRB.INFEASIBLE:
        raise ValueError("Problem is Infeasible!")
    
    while True:    
        new_partial = False
        for o, obs in enumerate(obss):
            for p in players:
                sol = solutions[o].X
                new_x = solve_player_weights_problem(obs, sol, p, weights.X)
                if not duplicate_array(partial_sols[o, p], new_x):
                    # new information
                    new_partial = True
                    # add to solutions
                    partial_sols[o, p].append(new_x)
                    # add to pm
                    new_profit = (obs.payoffs[p, :] * new_x).sum() + \
                                 sum(obs.interaction_coefs[p, p1, :] * new_x * sol[p1, :]
                                             for p1 in rivals[p])
                    pm.addConstr(delta[o, p] >= new_profit - og_profit[o, p])
        
        if new_partial:
            pm.optimize()
        else:
            break

    return weights.X


    return weights.X

if __name__ == "__main__":
    n = 2
    m = 10

    approach = "weights"

    if approach == "payoffs":
        problems = generate_payoff_problems(n*n*m, n, m, 0.5, payoff_type="asym", weight_type="asym", interaction_type="none")
        print(problems)
        input("Press Enter to run")
        solutions = solve_problems(problems)
        example = problems[0]
        payoff, interaction = inverse_range_payoff_KPG(problems, solutions)
        print("Original")
        print(example.payoffs) 
        print(example.interaction_coefs)
        # print(example.payoffs/example.payoffs.sum())
        # print(example.interaction_coefs/example.interaction_coefs.sum())

        print("Inverse")
        print(payoff)
        print(interaction)
    
    elif approach == "weights":
        problems = generate_weight_problems(n*m*m, n, m, 0.5, payoff_type="asym", weight_type="asym", interaction_type="none")
        print(problems)
        input("Press Enter to run")
        solutions = solve_problems(problems)
        example = problems[0]

        weights = inverse_weight_KPG(problems, solutions)

        print("Original")
        print(example.weights) 
        # print(example.payoffs/example.payoffs.sum())
        # print(example.interaction_coefs/example.interaction_coefs.sum())

        print("Inverse")
        print(weights)

        print(np.linalg.norm(example.weights - weights))
