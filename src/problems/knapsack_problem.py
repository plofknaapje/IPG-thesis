from dataclasses import dataclass

import numpy as np
import gurobipy as gp
from gurobipy import GRB


@dataclass
class KnapsackProblem:
    # Class for storing KP instances.
    n: int  # number of items
    payoffs: np.ndarray  # payoffs of items, (n)
    weights: np.ndarray  # weights of items, (n)
    capacity: int  # problem capacity
    solution: np.ndarray | None  # optimal solution to the KP

    def __init__(self, payoffs: np.ndarray, weights: np.ndarray, capacity: float | int):
        self.payoffs = payoffs
        self.weights = weights
        self.capacity = int(
            capacity
        )  # Round capacity to one digit to prevent boundary problems
        self.solution = None  # Lazy class. Call self.solve() to solve.
        self.n = len(payoffs)

    def solve(self) -> np.ndarray:
        """
        Solves the Knapsack Problem maximising x @ self.payoffs constrained by
        x @ self.weights <= self.capacity where the vector x is binary.
        The function also updates self.solution if that was still None.

        Raises:
            ValueError: The KP is infeasible.

        Returns:
            np.ndarray: An optimal solution to the KP.
        """
        if self.solution is not None:
            print("Already solved")
            return self.solution

        model = gp.Model("Knapsack Problem")

        x = model.addMVar((self.n), vtype=GRB.BINARY, name="x")

        model.setObjective(x @ self.payoffs, GRB.MAXIMIZE)

        model.addConstr(x @ self.weights <= self.capacity)

        model.optimize()

        if model.Status == GRB.INFEASIBLE:
            raise ValueError("Problem is Infeasible!")

        self.solution = x.X
        return self.solution

    def solve_weights(self, weights: np.ndarray) -> np.ndarray:
        """
        Solves the KP with provided weights vector.

        Args:
            weights (np.ndarray): A replacement weights vector.

        Raises:
            ValueError: The KP is infeasible.

        Returns:
            np.ndarray: An optimal solution to the KP.
        """
        model = gp.Model("Knapsack Problem")

        x = model.addMVar((self.n), vtype=GRB.BINARY, name="x")

        model.setObjective(x @ self.payoffs, GRB.MAXIMIZE)

        model.addConstr(x @ weights <= self.capacity)

        model.optimize()

        if model.Status == GRB.INFEASIBLE:
            raise ValueError("Problem is Infeasible!")

        return x.X

    def solve_payoffs(self, payoffs: np.ndarray) -> np.ndarray:
        """
        Solves the KP with a provided payoffs vector.

        Args:
            payoffs (np.ndarray): A replacement payoffs vector.

        Raises:
            ValueError: The KP is infeasible.

        Returns:
            np.ndarray: An optimal solution to the KP.
        """
        model = gp.Model("Knapsack Problem")

        x = model.addMVar((self.n), vtype=GRB.BINARY, name="x")

        model.setObjective(x @ payoffs, GRB.MAXIMIZE)

        model.addConstr(x @ self.weights <= self.capacity)

        model.optimize()

        if model.Status == GRB.INFEASIBLE:
            raise ValueError("Problem is Infeasible!")

        return x.X

    def solve_greedy(self) -> np.ndarray:
        """
        Solves the KP using the greedy heuristic of ranking items by their
        payoff / weight ratio.

        Returns:
            np.ndarray: Greedy solution to the KP.
        """
        p_w_ratio = self.payoffs / self.weights
        x = np.zeros(self.n)
        total_weight = 0

        while p_w_ratio.sum() > 0:
            highest = np.argmax(p_w_ratio)
            if total_weight + self.weights[highest] <= self.capacity:
                x[highest] = 1
                total_weight += self.weights[highest]

            p_w_ratio[highest] = 0

        print(x @ self.weights, self.capacity)

        return x
