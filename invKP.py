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



def generate_weight_problems(n_problems=10, n_items=10, capacity_perc=0.5, rng=None) -> list[KP]:
    if rng is None:
        rng = np.random.default_rng()
    problems = []
    weights = np.arange(1, n_items+1)
    payoffs = np.arange(1, n_items+1)
    rng.shuffle(weights)
    capacity = weights.sum() * capacity_perc
    for _ in range(n_problems):
        problem_payoffs = payoffs.copy()
        rng.shuffle(problem_payoffs)
        problem = KP(problem_payoffs, weights, capacity)
        problem.solve()
        problems.append(problem)

    return problems

def generate_payoff_problems(n_problems=10, n_items=10, capacity_perc=0.5, rng=None) -> list[KP]:
    if rng is None:
        rng = np.random.default_rng()
    problems = []
    weights = np.arange(1, n_items+1)
    payoffs = np.arange(1, n_items+1)
    rng.shuffle(payoffs)
    capacity = int(weights.sum() * capacity_perc)
    for _ in range(n_problems):
        problem_weights = weights.copy()
        rng.shuffle(problem_weights)
        problem = KP(payoffs, problem_weights, capacity)
        problem.solve()
        problems.append(problem)

    return problems


def inverse_weights(problems: list[KP], verbose=False) -> tuple[np.ndarray, int]:
    n_items = problems[0].n_items
    capacity = problems[0].capacity
    proxy_weights = np.linspace(1, 1.01, n_items)
    true_value = [problem.solution @ problem.payoffs for problem in problems]

    model = gp.Model("Inverse Knapsack (Weights)")
    model.Params.PoolSearchMode = 2
    model.Params.PoolSolutions = 10

    w = model.addMVar((n_items), lb=1, vtype=GRB.INTEGER, name="weights")

    model.setObjective(w @ proxy_weights)

    model.addConstr(w.sum() >= (n_items + 1) * n_items / 2)
    model.addConstrs(w[i] <= n_items for i in range(n_items))

    for problem in problems:
        model.addConstr(problem.solution @ w <= capacity)

    model.optimize()
    if model.Status == GRB.INFEASIBLE:
            raise ValueError("Problem is Infeasible!")

    while True:
        new_constraint = False

        for i, problem in enumerate(problems):
            new_solution = problem.weight_solve(w.X)
            new_value = new_solution @ problem.payoffs

            if new_value <= true_value[i]:
                continue

            if verbose:
                print(i, new_solution)

            new_constraint = True
            model.addConstr(new_solution @ w >= capacity)

        if not new_constraint:
            break

        model.optimize()
        if model.Status == GRB.INFEASIBLE:
            raise ValueError("Problem is Infeasible!")

    # print()
    # print(problems[0].weights)
    # for s in range(model.SolCount):
    #     model.Params.SolutionNumber  = s
    #     print(w.Xn.astype(np.int64))

    return w.X.astype(np.int64), model.SolCount


def inverse_payoffs(problems: list[KP], verbose=False) -> tuple[np.ndarray, int]:
    n_items = problems[0].n_items
    values = np.arange(1, n_items + 1)
    proxy_weights = np.linspace(1, 1.01, n_items)

    model = gp.Model("Inverse Knapsack (Payoffs)")

    model.Params.PoolSearchMode = 2
    model.Params.PoolSolutions = 10

    p = model.addMVar((n_items), lb=1, vtype=GRB.INTEGER, name="payoffs")

    model.setObjective(p @ proxy_weights)

    model.addConstr(p.sum() >= (n_items + 1) * n_items / 2)

    model.optimize()

    if model.Status == GRB.INFEASIBLE:
            raise ValueError("Problem is Infeasible!")

    #solutions = [[problem.solution] for problem in problems]

    while True:
        new_constraint = False

        for i, problem in enumerate(problems):
            new_solution = problem.payoff_solve(p.X)

            if new_solution @ new_payoffs <= problem.solution @ new_payoffs:
                continue

            if verbose:
                print(i, new_solution)
            new_constraint = True

            model.addConstr(p @ new_solution <= p @ problem.solution)

        if not new_constraint:
            break

        model.optimize()
        if model.Status == GRB.INFEASIBLE:
            raise ValueError("Problem is Infeasible!")

    return p.X.astype(np.int64), model.SolCount


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n_items = 10
    extras = 20
    repeat = 20
    extras_errors = []
    extra_solutions = []

    approach = "payoffs"

    print(approach)
    if approach == "weights":
        problem_sets = [generate_weight_problems(n_items + extras, n_items, capacity_perc=0.5, rng=rng)
                        for _ in range(repeat)]

        for extra in range(extras):
            index = n_items + extra
            errors = []
            num_sols = []
            for problems in problem_sets:
                weights = problems[0].weights
                inv_weights, sols = inverse_weights(problems[:index])
                errors.append(abs(weights - inv_weights).sum())
                num_sols.append(sols)

            print(errors)
            extras_errors.append(np.mean(errors))
            extra_solutions.append(np.mean(num_sols))

    if approach == "payoffs":
        problem_sets = [generate_payoff_problems(n_items*2 + extras, n_items, capacity_perc=0.5, rng=rng)
                        for _ in range(repeat)]

        for extra in range(extras):
            index = n_items*2 + extra
            errors = []
            num_sols = []

            for problems in problem_sets:
                payoffs = problems[0].payoffs
                inv_payoffs, sols = inverse_payoffs(problems[:index])
                errors.append(abs(payoffs - inv_payoffs).sum())
                num_sols.append(sols)

            extras_errors.append(np.mean(errors))
            extra_solutions.append(np.mean(num_sols))

    print(extras_errors)
    print(extra_solutions)

