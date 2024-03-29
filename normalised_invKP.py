import enum
import gurobipy as gp
from gurobipy import GRB
from dataclasses import dataclass
import numpy as np
from time import time

@dataclass
class KP:
    n_items: int
    payoffs: np.ndarray
    weights: np.ndarray
    capacity: float
    solution: np.ndarray

    def __init__(self, payoffs: np.ndarray, weights: np.ndarray, capacity: int):
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
    

def generate_weight_problems(items=10, n_problems=100, r=100, capacity:float|list=0.5, corr=True, rng=None) -> list[KP]:
    if rng is None:
        rng = np.random.default_rng()
    
    problems = []
    weights = rng.integers(1, r+1, items)
    norm_weights = weights / weights.sum()

    if type(capacity) is float:
        capacity = [capacity for _ in range(n_problems)]

    if corr:
        lower = np.maximum(weights - r / 10, 1)
        upper = np.minimum(weights + r / 10, r)

    for i in range(n_problems):
        if corr:
            payoffs = rng.integers(lower, upper)
        else:
            payoffs = rng.integers(1, r+1, items)
        norm_payoffs = payoffs / payoffs.sum()
        problems.append(KP(norm_payoffs, norm_weights, capacity[i]))

    return problems


def generate_payoff_problems(items=10, n_problems=100, r=100, capacity:float|list=0.5, corr=True, rng=None) -> list[KP]:
    if rng is None:
        rng = np.random.default_rng()
    
    problems = []
    payoffs = rng.integers(1, r+1, items)
    norm_payoffs = payoffs / payoffs.sum()

    if type(capacity) is float:
        capacity = [capacity for _ in range(n_problems)]

    if corr:
        lower = np.maximum(payoffs - r / 10, 1)
        upper = np.minimum(payoffs + r / 10, r)

    for i in range(n_problems):
        if corr:
            weights = rng.integers(lower, upper)
        else:
            weights = rng.integers(1, r+1, items)
        norm_weights = weights / weights.sum()
        problems.append(KP(norm_payoffs, norm_weights, capacity[i]))

    return problems


def inverse_weights(problems: list[KP], eps=0.0001) -> tuple[np.ndarray, int]:
    items = problems[0].n_items
    true_value = [problem.solution @ problem.payoffs for problem in problems]

    model = gp.Model("Inverse Knapsack (Weights)")
    model.Params.PoolSearchMode = 2
    model.Params.PoolSolutions = 10

    w = model.addMVar((items), name="w")

    model.setObjective(w.sum())

    model.addConstr(w.sum() >= 1)

    for problem in problems:
        model.addConstr(problem.solution @ w <= problem.capacity)

    model.optimize()
    
    if model.Status == GRB.INFEASIBLE:
        raise ValueError("Problem is Infeasible!")
    
    while True:
        new_constraint = False

        for i, problem in enumerate(problems):
            new_solution = problem.weight_solve(w.X)
            new_value = new_solution @ problem.payoffs

            if new_value >= true_value[i] + eps:
                model.addConstr(new_solution @ w >= problem.capacity + eps)
                new_constraint = True

        if not new_constraint:
            break
            
        model.optimize()

        if model.Status == GRB.INFEASIBLE:
            raise ValueError("Problem is Infeasible!")
        
    return w.X, model.SolCount


def inverse_payoffs_direct(problems: list[KP], eps = 0.001) -> tuple[np.ndarray, int]:
    n_items = problems[0].n_items

    model = gp.Model("Inverse Knapsack (Payoffs)")
    model.Params.PoolSearchMode = 2
    model.Params.PoolSolutions = 10

    p = model.addMVar((n_items), name="p")

    model.setObjective(gp.quicksum(p @ problem.solution for problem in problems), GRB.MAXIMIZE)

    model.addConstr(p.sum() <= 1)

    model.optimize()
    
    if model.Status == GRB.INFEASIBLE:
        raise ValueError("Problem is Infeasible!")
    
    while True:
        new_constraint = False

        for _, problem in enumerate(problems):
            new_solution = problem.payoff_solve(p.X)
            new_value = new_solution @ p.X
            true_value = problem.solution @ p.X

            if new_value >= true_value + eps:
                model.addConstr(problem.solution @ p >= new_solution @ p)
                new_constraint = True

        if not new_constraint:
            break
            
        model.optimize()

        if model.Status == GRB.INFEASIBLE:
            raise ValueError("Problem is Infeasible!")
        
    return p.X, model.SolCount


def inverse_payoffs_delta(problems: list[KP]) -> tuple[np.ndarray, int]:
    n_items = problems[0].n_items
    n_problems = len(problems)

    model = gp.Model("Inverse Knapsack (Payoffs)")
    model.Params.PoolSearchMode = 2
    model.Params.PoolSolutions = 10

    delta = model.addMVar((n_problems), name="delta")
    p = model.addMVar((n_items), name="p")

    model.setObjective(delta.sum())

    model.addConstr(p.sum() == 1)

    model.optimize()

    if model.Status == GRB.INFEASIBLE:
        raise ValueError("Problem is Infeasible!")
    
    solutions = [set() for _ in problems]

    while True:
        new_constraint = False

        for i, problem in enumerate(problems):
            new_solution = problem.payoff_solve(p.X)
            if tuple(new_solution) in solutions[i]:
                continue
            model.addConstr(delta[i] >= new_solution @ p - problem.solution @ p)
            new_constraint = True
            solutions[i].add(tuple(new_solution))
        
        if not new_constraint:
            break

        model.optimize()

        if model.Status == GRB.INFEASIBLE:
            raise ValueError("Problem is Infeasible!")

    return p.X, model.SolCount


def inverse_eps_opt(inverse_function, problems: list[KP], truth: np.ndarray, start: int, end: int):
    for i in range(start, end + 1):
        try:
            values, solutions = inverse_function(problems, 10**-(i/2))
            print(np.abs(values - truth).sum(), solutions)
        except ValueError:
            print(f"Problem not possible with eps={10**-(i/2)}")

if __name__ == "__main__":

    start = time()
    rng = np.random.default_rng(0)

    weight_problems = generate_weight_problems(50, 150, rng=rng, corr=False)
    payoff_problems = generate_payoff_problems(20, 100, rng=rng)

    for problem in weight_problems:
        problem.solve()
    for problem in payoff_problems:
        problem.solve()

    original = payoff_problems[0].payoffs
    inverse, _ = inverse_payoffs_delta(payoff_problems)
    
    print(np.abs(original - inverse).sum())
    # inverse_eps_opt(inverse_weights, weight_problems, weight_problems[0].weights, 5, 10)
    # inverse_eps_opt(inverse_payoffs_direct, payoff_problems, payoff_problems[0].payoffs, 4, 10)

    # original = weight_problems[0].weights
    # inverse = inverse_weights(weight_problems, 0.0001)[0]

    # print(original)
    # print(inverse)

    # error = np.abs(original - inverse).sum()
    # print(error)
    print(time() - start)