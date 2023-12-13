import numpy as np
import gurobipy as gp
from gurobipy import GRB
from dataclasses import dataclass
import time
import math

@dataclass
class KPG:
    # Class for storing all data of a binary Knapsack Packing Game.
    n: int # number of players
    m: int # number of items
    players: list # list of player indices, length n
    capacity: list # carrying capacity of each player, length n
    weights: np.ndarray # weights of the items, size (n, m)
    payoffs: np.ndarray # payoffs of the items, size (n, m)
    interaction_coefs: np.ndarray # interaction payoff of the items (n, m)
    weights_type: str # weights setup: sym or asym
    payoffs_type: str # payoffs setup: sym or asym
    interaction_type: str # interaction setup: sym, asym or negasym

    def __init__(self, weights: np.ndarray, payoffs: np.ndarray, interaction_coefs: np.ndarray, capacity=0.2):
        self.n, self.m = weights.shape
        self.weights = weights
        self.payoffs = payoffs
        self.interaction_coefs = interaction_coefs
        self.players = list(range(self.n))
        self.capacity = [self.weights[p, :].sum() * capacity for p in self.players]

        # Assign type with heuristic
        if (self.weights[0, :] == self.weights[1, :]).all():
            self.weights_type = "sym"
        else:
            self.weights_type = "asym"

        if (self.payoffs[0, :] == self.payoffs[1, :]).all():
            self.payoffs_type = "sym"
        else:
            self.payoffs_type = "asym"
        
        if (np.abs(self.interaction_coefs) != self.interaction_coefs).any():
            self.interaction_type = "negasym"
        elif (self.interaction_coefs[0, :] == self.interaction_coefs[1, :]).all():
            self.interaction_type = "asym"
        else:
            self.interaction_type = "sym"

    def generate(n=2, m=25, capacity=0.2, weight_type="sym", payoff_type="sym", interaction_type="sym"):
        players = list(range(n))
        match weight_type:
            case "sym":
                weight = np.random.randint(1, 101, m)
                weights = np.zeros((n, m))
                for player in players:
                    weights[player, :] = weight
            case "asym":
                weights = np.random.randint(1, 101, (n, m))
            case _:
                raise ValueError("Weight type not recognised!")

        match payoff_type:
            case "sym":
                payoff = np.random.randint(1, 101, m)
                payoffs = np.zeros((n, m))
                for player in players:
                    payoffs[player, :] = payoff
            case "asym":
                payoffs = np.random.randint(1, 101, (n, m))
            case _:
                raise ValueError("Payoff type not recognised!")

        match interaction_type:
            case "sym":
                coefs = np.random.randint(1, 101, m)
                interaction_coefs = np.zeros((n, m))
                for player in players:
                    interaction_coefs[player, :] = coefs
            case "asym":
                interaction_coefs = np.random.randint(1, 101, (n, m))
            case "negasym":
                interaction_coefs = np.random.randint(-100, 101, (n, m))
            case _:
                raise ValueError("Interaction type not recognised!")
        
        kpg = KPG(weights, payoffs, interaction_coefs, capacity)

        kpg.interaction_type = interaction_type
        kpg.weights_type = weight_type
        kpg.payoffs_type = payoff_type

        return kpg
    
    def read_file(file: str):
        with open(file) as f:
            lines = [line.strip() for line in f]

        n, m = [int(i) for i in lines[0].split(" ")]
        capacity = [int(i) for i in lines[1].split(" ")]
        weights = np.zeros((n, m))
        payoffs = np.zeros((n, m))
        interaction_coefs = np.zeros((math.factorial(n),m))

        for i, line in enumerate(lines[2:]):
            line = [int(i) for i in line.split()]
            payoffs[: ,i] = line[1:n+1]
            weights[:, i] = line[n+1:2*n + 1]
            interaction_coefs[:, i] = line[2*n + 1:]

        kpg = KPG(weights, payoffs, interaction_coefs)
        kpg.capacity = capacity
        
        return kpg


def oracle_optimization(kpg: KPG, point_x: np.ndarray, p: int, verbose=False) -> tuple[np.ndarray, int]:
    """ Detect local suboptimal solutions
    Function checks if a solution is suboptimal for player p and returns a local improvement if possible.

    Args:
        kpg (KPG): KPG problem.
        point_x (np.ndarray): point to check.
        p (int): index of player to check.
        verbose (bool, optional): enable verbose output. Defaults to False.

    Raises:
        ValueError: If problem is infeasible, but then there is a problem higher up!

    Returns:
        tuple[np.ndarray, int]: new matrix x as well as new player objective.
    """    
    # Only for n=2!
    # TODO: extend for higher n.
    m = gp.Model("LocalKPG", env=env)
    x = m.addMVar((kpg.n, kpg.m), vtype=GRB.BINARY, name="x")
    for i in kpg.players:
        # Fix actions of other players
        if i != p:
            for j in range(kpg.m):
                x[i, j].lb = point_x[i, j]
                x[i, j].ub = point_x[i, j]
    z = m.addMVar(kpg.m, vtype=GRB.BINARY, name="z")
    
    m.setObjective(kpg.payoffs[p, :] @ x[p, :] + kpg.interaction_coefs[p, :] @ z, GRB.MAXIMIZE)

    for i in kpg.players:
        m.addConstr(kpg.weights[i, :] @ x[i, :] <= kpg.capacity[i])
        for j in range(kpg.m):
            m.addConstr(z[j] <= x[i, j])

    for j in range(kpg.m):
        m.addConstr(z[j] >= (x[:, j]).sum() - 1)
    
    m.optimize()

    if verbose:
        for v in m.getVars():
            print(f"{v.VarName} {v.X:g}")

        print(f"Obj: {m.ObjVal:g}")

    if m.Status == GRB.INFEASIBLE:
        raise ValueError("Problem is Infeasible! This is not possible!")

    return x.X, m.ObjVal


def zero_regrets(kpg: KPG, verbose=False) -> tuple[np.ndarray, int]:
    """Optimises kpg using the ZeroRegrets methods of cutting.

    Args:
        kpg (KPG): KPG problem.
        verbose (bool, optional): _description_. Defaults to False.

    Raises:
        ValueError: _description_

    Returns:
        tuple[np.ndarray, int]: _description_
    """    
    # Only for n=2!
    # TODO: extend for higher n.
    start = time.time()
    pm = gp.Model("ZeroRegrets", env=env)
    x = pm.addMVar((kpg.n, kpg.m), vtype=GRB.BINARY, name="x")
    z = pm.addMVar(kpg.m, vtype=GRB.BINARY, name="z")

    pm.setObjective(gp.quicksum(kpg.payoffs[i, j] * x[i, j] + kpg.interaction_coefs[i, j] * z[j] 
                                for j in range(kpg.m) for i in kpg.players), GRB.MAXIMIZE)

    for i in kpg.players:
        # Capacity limitations of players
        pm.addConstr(gp.quicksum(kpg.weights[i, j] * x[i, j] for j in range(kpg.m)) <= kpg.capacity[i])
        for j in range(kpg.m):
            # Setup for indicator variable z
            pm.addConstr(z[j] <= x[i, j])
    
    for j in range(kpg.m):
        # Setup for indicator variable z
        pm.addConstr(z[j] >= x[0, j] + x[1, j] - 1)  
    

    added_constraints = 0
    while True:
        pm.optimize()
        
        if pm.Status == GRB.INFEASIBLE:
            raise ValueError("IPG is not feasible!")
        
        current_x = x.X
        current_z = z.X
        finished = True
        print("====")
        print(f"Current total is {pm.ObjVal}")
        for player in kpg.players:
            obj = kpg.payoffs[player, :] @ current_x[player, :] + kpg.interaction_coefs[player, :] @ current_z
            
            new_x, new_obj = oracle_optimization(kpg, current_x, player)
            print(obj, new_obj)
            if new_obj > obj:
                # Add constraint!
                if player == 0:
                    opponent = 1
                else:
                    opponent = 0
                pm.addConstr(kpg.payoffs[player, :] @ new_x[player, :] + 
                             gp.quicksum(kpg.interaction_coefs[player, j] * new_x[player, j] * x[opponent, j] for j in range(kpg.m)) <= 
                             gp.quicksum(kpg.payoffs[player, j] * x[player, j] for j in range(kpg.m)) + 
                             gp.quicksum(kpg.interaction_coefs[player, j] * z[j] for j in range(kpg.m)))  
                print("Added constraint")
                added_constraints += 1
                finished = False
        print("====")
        if finished:
            print(f"Added {added_constraints} constraints!")
            print(x.X)
            break

    if verbose:
        for v in pm.getVars():
            print(f"{v.VarName} {v.X:g}")

        print(f"Obj: {pm.ObjVal:g}")

    return x.X, pm.ObjVal, round(time.time() - start, 1)


if __name__ == "__main__":
    np.random.seed(42)
    env = gp.Env("gurobi.log")
    env.setParam("OutputFlag", 0)
    prefix = "instances_kp/generated/"
    
    for capacity in [2, 5, 8]:
        file = prefix + f"2-25-{capacity}-pot.txt"
        print(file)
        instance = KPG.read_file(file)
        print(instance)
        x, obj, runtime = zero_regrets(instance)
        print(f"{obj} in {runtime} seconds")
    
    # kpg = KPG.generate(m=25, capacity=0.2, weight_type="asym", payoff_type="asym", interaction_type="asym")

    # x, obj, runtime = zero_regrets(kpg)
    # print(runtime)