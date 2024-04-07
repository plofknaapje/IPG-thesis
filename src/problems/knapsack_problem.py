from dataclasses import dataclass

import numpy as np
import gurobipy as gp
from gurobipy import GRB


@dataclass
class KnapsackProblem:
    n_items: int
    payoffs: np.ndarray
    weights: np.ndarray
    capacity: float
    solution: np.ndarray | None

    def __init__(self, payoffs: np.ndarray, weights: np.ndarray, capacity: float | int):
        self.payoffs = payoffs
        self.weights = weights
        self.capacity = round(float(capacity), 2)
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

    def solve_greedy(self) -> np.ndarray:
        p_w_ratio = self.payoffs / self.weights
        x = np.zeros(self.n_items)
        total_weight = 0

        while p_w_ratio.sum() > 0:
            highest = np.argmax(p_w_ratio)
            if total_weight + self.weights[highest] <= self.capacity:
                x[highest] = 1
                total_weight += self.weights[highest]

            p_w_ratio[highest] = 0

        print(x @ self.weights, self.capacity)

        return x