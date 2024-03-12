import gurobipy as gp
from gurobipy import GRB
from dataclasses import dataclass
import numpy as np
from utils import duplicate_array

@dataclass
class KP:
    n_items: int
    payoffs: np.ndarray
    weights: np.ndarray
    capacity: int
    solution: np.ndarray

    def __init__(self, payoffs: np.ndarray, weights: np.ndarray, capacity: int | float):
        self.payoffs = payoffs
        self.weights = weights
        self.capacity = capacity
        self.solution = None
        self.n_items = len(payoffs)
    
    def solve(self) -> None:
        if self.solution is not None:
            print("Already solved")
            return None
        
        model = gp.Model("Knapsack Problem")

        x = model.addMVar((self.n_items), vtype=GRB.BINARY, name="x")

        model.setObjective(x @ self.payoffs, GRB.MAXIMIZE)

        model.addConstr(x @ self.weights <= self.capacity)

        model.optimize()

        if model.Status == GRB.INFEASIBLE:
            raise ValueError("Problem is Infeasible!")
        
        self.solution = x.X
    
    def weight_solve(self, weights: np.ndarray) -> np.ndarray:
        model = gp.Model("Knapsack Problem")

        x = model.addMVar((self.n_items), vtype=GRB.BINARY, name="x")

        model.setObjective(x @ self.payoffs, GRB.MAXIMIZE)

        model.addConstr(x @ weights <= self.capacity)

        model.optimize()

        if model.Status == GRB.INFEASIBLE:
            raise ValueError("Problem is Infeasible!")
        
        return x.X
    
    def payoff_solve(self, payoffs: np.ndarray) -> np.ndarray:
        model = gp.Model("Knapsack Problem")

        x = model.addMVar((self.n_items), vtype=GRB.BINARY, name="x")

        model.setObjective(x @ payoffs, GRB.MAXIMIZE)

        model.addConstr(x @ self.weights <= self.capacity)

        model.optimize()

        if model.Status == GRB.INFEASIBLE:
            raise ValueError("Problem is Infeasible!")
        
        return x.X



def generate_weight_problems(n_problems=10, n_items=10, capacity_perc=0.5) -> list[KP]:
    problems = []
    weights = np.arange(1, n_items+1)
    payoffs = np.arange(1, n_items+1)
    np.random.shuffle(weights)
    capacity = weights.sum() * capacity_perc
    for _ in range(n_problems):
        problem_payoffs = payoffs.copy()
        np.random.shuffle(problem_payoffs)
        problem = KP(problem_payoffs, weights, capacity)
        problem.solve()
        problems.append(problem)
    
    return problems

def generate_payoff_problems(n_problems=10, n_items=10, capacity_perc=0.5) -> list[KP]:
    problems = []
    weights = np.arange(1, n_items+1)
    payoffs = np.arange(1, n_items+1)
    np.random.shuffle(payoffs)
    capacity = weights.sum() * capacity_perc
    for _ in range(n_problems):
        problem_weights = weights.copy()
        np.random.shuffle(problem_weights)
        problem = KP(payoffs, problem_weights, capacity)
        problem.solve()
        problems.append(problem)
    
    return problems


def inverse_weights(problems: list[KP], verbose=False) -> np.ndarray:
    n_items = problems[0].n_items
    capacity = problems[0].capacity

    model = gp.Model("Inverse Knapsack (Weights)")

    w = model.addMVar((n_items), lb=1, vtype=GRB.INTEGER, name="weights")

    model.setObjective(w.sum())

    for problem in problems:
        model.addConstr(problem.solution @ w <= capacity)

    model.optimize()

    if model.Status == GRB.INFEASIBLE:
            raise ValueError("Problem is Infeasible!")
    
    solutions = [[problem.solution] for problem in problems]
    new_constraint = True

    while new_constraint:
        new_constraint = False
        new_weights = w.X

        for i, problem in enumerate(problems):
            new_solution = problem.weight_solve(new_weights)

            if duplicate_array(solutions[i], new_solution):
                continue
                
            if new_solution @ problem.payoffs > problem.solution @ problem.payoffs:
                if verbose:
                    print(i, new_solution)
                new_constraint = True
                solutions[i].append(new_solution)
                model.addConstr(new_solution @ w >= capacity)

        model.optimize()

        if model.Status == GRB.INFEASIBLE:
            raise ValueError("Problem is Infeasible!")
    

    return w.X


def inverse_payoffs(problems: list[KP], verbose=False) -> np.ndarray:
    n_items = problems[0].n_items
    values = np.arange(1, n_items + 1)

    model = gp.Model("Inverse Knapsack (Payoffs)")

    p = model.addMVar((n_items), lb=1, vtype=GRB.INTEGER, name="payoffs")
    # binary_p = model.addMVar((n_items, n_items), vtype=GRB.BINARY)

    model.setObjective(p.sum())
    
    # model.addConstrs(binary_p[i, :].sum() == 1 for i in range(n_items))
    # model.addConstrs(binary_p[:, i].sum() == 1 for i in range(n_items))
    # model.addConstrs(p[i] == binary_p[i, :] @ values for i in range(n_items))
    model.addConstr(p.sum() >= (n_items + 1) * n_items / 2)

    model.optimize()

    if model.Status == GRB.INFEASIBLE:
            raise ValueError("Problem is Infeasible!")

    solutions = [[problem.solution] for problem in problems]
    new_constraint = True

    while new_constraint:
        new_constraint = False
        new_payoffs = p.X

        for i, problem in enumerate(problems):
            new_solution = problem.payoff_solve(new_payoffs)

            if duplicate_array(solutions[i], new_solution):
                continue

            if new_solution @ new_payoffs > problem.solution @ new_payoffs:
                if verbose:
                    print(i, new_solution)
                new_constraint = True
                solutions[i].append(new_solution)

                model.addConstr(p @ new_solution <= p @ problem.solution)

        model.optimize()

        if model.Status == GRB.INFEASIBLE:
            raise ValueError("Problem is Infeasible!")
        
        
    return p.X


if __name__ == "__main__":
    n_items = 10
    extras = 10
    repeat = 10
    errors = [[] for _ in range(extras)]
    extras_errors = []

    approach = "payoffs"
    if approach == "weights":
        for _ in range(repeat):
            problems = generate_weight_problems(n_items + extras, n_items, capacity_perc=0.5)
            weights = problems[0].weights
            print(weights)

            for extra in range(extras):
                inv_weights = inverse_weights(problems[:n_items + extra])
                # print(inv_weights)
                # print(np.abs(weights - inv_weights).sum())
                errors[extra].append(np.abs(weights - inv_weights).sum())
    
    if approach == "payoffs":
        for _ in range(repeat):
            problems = generate_payoff_problems(n_items*9 + extras, n_items, capacity_perc=0.5)
            payoffs = problems[0].payoffs
            print(payoffs)

            for extra in range(extras):
                inv_payoffs = inverse_payoffs(problems[:n_items + extra])
                # print(inv_payoffs)
                # print(np.abs(payoffs - inv_payoffs).sum())
                errors[extra].append(np.abs(payoffs - inv_payoffs).sum())
    
    for extra in range(extras):
        print(errors[extra])
        if approach == "payoffs":
            extras_errors.append(np.mean(errors[extra])/(n_items*9 + extra))
        else:
            extras_errors.append(np.mean(errors[extra])/(n_items + extra))
    print(extras_errors)

    
        