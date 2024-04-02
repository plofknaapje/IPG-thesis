from dataclasses import dataclass
import numpy as np
import gurobipy as gp
from gurobipy import GRB

@dataclass
class KP:
    n_items: int
    payoffs: np.ndarray
    weights: np.ndarray
    capacity: float
    solution: np.ndarray

    def __init__(self, payoffs: np.ndarray, weights: np.ndarray, capacity: int|float):
        self.payoffs = payoffs
        self.weights = weights
        self.capacity = capacity
        self.solution = None
        self.n_items = len(payoffs)

    def solve(self) -> np.ndarray:
        if self.solution is not None:
            print("Already solved")
            return self.solution

        model = gp.Model("Knapsack Problem")

        x = model.addMVar((self.n_items), vtype=GRB.BINARY, name="x")

        model.setObjective(x @ self.payoffs, GRB.MAXIMIZE)

        model.addConstr(x @ self.weights <= self.capacity)

        model.optimize()

        if model.Status == GRB.INFEASIBLE:
            raise ValueError("Problem is Infeasible!")

        self.solution = x.X
        return self.solution

    def solve_weights(self, weights: np.ndarray) -> np.ndarray:
        model = gp.Model("Knapsack Problem")

        x = model.addMVar((self.n_items), vtype=GRB.BINARY, name="x")

        model.setObjective(x @ self.payoffs, GRB.MAXIMIZE)

        model.addConstr(x @ weights <= self.capacity)

        model.optimize()

        if model.Status == GRB.INFEASIBLE:
            raise ValueError("Problem is Infeasible!")

        return x.X

    def solve_payoffs(self, payoffs: np.ndarray) -> np.ndarray:
        model = gp.Model("Knapsack Problem")

        x = model.addMVar((self.n_items), vtype=GRB.BINARY, name="x")

        model.setObjective(x @ payoffs, GRB.MAXIMIZE)

        model.addConstr(x @ self.weights <= self.capacity)

        model.optimize()

        if model.Status == GRB.INFEASIBLE:
            raise ValueError("Problem is Infeasible!")

        return x.X