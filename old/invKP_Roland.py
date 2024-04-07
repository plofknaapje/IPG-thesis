import numpy as np
from dataclasses import dataclass
import gurobipy as gp
from gurobipy import GRB


@dataclass
class KP:
    n_items: int
    payoffs: np.ndarray
    weights: np.ndarray
    capacity: int
    solution: np.ndarray | None = None
    solution_type: str | None = None

    def solve(self):
        if self.solution is not None and self.solution_type == "exact":
            print("Solution is already exact")
            return None

        model = gp.Model("Knapsack Problem")

        x = model.addMVar((self.n_items), vtype=GRB.BINARY, name="x")

        model.setObjective(x @ self.payoffs, GRB.MAXIMIZE)

        model.addConstr(x @ self.weights <= self.capacity)

        model.optimize()

        if model.Status == GRB.INFEASIBLE:
            raise ValueError("Problem is Infeasible!")

        self.solution = x.X
        self.solution_type = "exact"

    def greedy_solve(self):
        if self.solution is not None and self.solution_type == "greedy":
            print("Solution is already greedy")
            return None

        p_w_ratio = self.payoffs / self.weights
        x = np.zeros(self.n_items)
        total_weight = 0

        while p_w_ratio.sum() > 0:
            highest = np.argmax(p_w_ratio)
            if total_weight + self.weights[highest] <= self.capacity:
                x[highest] = 1
                total_weight += self.weights[highest]

            p_w_ratio[highest] = 0

        self.solution = x
        self.solution_type = "greedy"


def generate_roland_instances(
    n=100, r=1000, instance_class=1, type=1, p=0.5, solution="greedy"
) -> list[KP]:
    payoffs = []
    weights = []
    capacities = []

    match type:
        case 1:
            s = 100
        case 2:
            s = 30
        case 3:
            s = 30
        case _:
            raise ValueError("Unknown type")

    match instance_class:
        case 1:
            for i in range(s):
                rng = np.random.default_rng(i)
                weights.append(rng.integers(1, r + 1, n))
                payoffs.append(rng.integers(1, r + 1, n))
        case 2:
            for i in range(s):
                rng = np.random.default_rng(i)
                weights.append(rng.integers(1, r + 1, n))
                lower = np.maximum(weights[-1] - r / 10, 1)
                upper = np.minimum(weights[-1] + r / 10, r)
                payoffs.append(rng.integers(lower, upper))
        case 3:
            for i in range(s):
                rng = np.random.default_rng(i * 10)
                weights.append(rng.integers(1, r + 1, n))
                payoffs.append(weights[-1] + 10)
        case _:
            raise ValueError("Unknown instance class")

    match type:
        case 1:
            for i in range(s):
                capacities.append(max(r, np.floor(i / (s + 1) * weights[i].sum())))
        case 2:
            for i in range(s):
                capacities.append(max(r, np.floor(p * weights[i].sum())))
        case 3:
            for i in range(s):
                capacities.append(max(r, np.floor(p * weights[i].sum())))
        case _:
            raise ValueError("Unknown type")

    instances = [KP(n, payoffs[i], weights[i], capacities[i]) for i in range(s)]
    match solution:
        case "greedy":
            for instance in instances:
                instance.greedy_solve()
        case "exact":
            for instance in instances:
                instance.solve()
        case _:
            raise ValueError("Unknown solution type")

    return instances


instances = generate_roland_instances(instance_class=2, type=2, solution="greedy")


def inverse_IKP_inf(instance: KP, p=None):
    if p is None:
        p = instance.payoffs

    C = np.max((1 - instance.solution) * instance.payoffs)
    a = 0
    b = C
    while a != b:
        k = a + np.floor((b - a) / 2)
        d = np.zeros(instance.n_items)

        for i in range(instance.n_items):
            if instance.solution[i] == 0:
                d[i] = np.maximum(0, instance.payoffs[i] - k)
            else:
                d[i] = instance.payoffs[i] + k

        new_KP = KP(instance.n_items, d, instance.weights, instance.capacity)
        new_KP.solve()
        opt = d @ new_KP.solution
        if opt == d @ instance.solution:
            b = k
        else:
            a = k + 1
    return d


def inverse_IKP_L1(instance: KP) -> np.ndarray:
    n = instance.n_items

    model = gp.Model("Knapsack Problem")

    delta = model.addMVar((n), vtype=GRB.INTEGER, name="delta")
    d = model.addMVar((n), vtype=GRB.INTEGER, name="d")

    model.setObjective(delta.sum())

    model.addConstrs(delta[i] >= d[i] - instance.payoffs[i] for i in range(n))
    model.addConstrs(delta[i] >= instance.payoffs[i] - d[i] for i in range(n))

    return d.X


print(instances[0].payoffs)

print(inverse_IKP_inf(instances[0]))
