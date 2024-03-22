import gurobipy as gp
from gurobipy import GRB
from dataclasses import dataclass
import numpy as np
from time import time
from utils import duplicate_array

@dataclass
class KP:
    n_items: int
    payoffs: np.ndarray
    weights: np.ndarray
    capacity: int
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


def generate_weight_problems(n_problems=10, n_items=10, capacity_perc=0.5, rng=None) -> list[KP]:
    if rng is None:
        rng = np.random.default_rng()
    problems = []
    weights = np.arange(1, n_items+1)
    payoffs = np.arange(1, n_items+1)
    rng.shuffle(weights)
    capacity = int(weights.sum() * capacity_perc)
    for _ in range(n_problems):
        problem_payoffs = payoffs.copy()
        rng.shuffle(problem_payoffs)
        problem = KP(problem_payoffs, weights, capacity)
        problem.solve()
        problems.append(problem)

    return problems


def generate_payoff_problems(n_problems=10, n_items=10, capacity_perc: float|list=0.5, rng=None) -> list[KP]:
    if rng is None:
        rng = np.random.default_rng()
    problems = []
    weights = np.arange(1, n_items+1)
    payoffs = np.arange(1, n_items+1)
    rng.shuffle(payoffs)
    if type(capacity_perc) is float:
        capacity = [int(weights.sum() * capacity_perc)]
    elif type(capacity_perc) is list:
        capacity = [int(weights.sum() * perc) for perc in capacity_perc]

    for i in range(n_problems):
        problem_weights = weights.copy()
        rng.shuffle(problem_weights)
        problem = KP(payoffs, problem_weights, capacity[i % len(capacity)])
        problem.solve()
        problems.append(problem)

    return problems


def inverse_weights(problems: list[KP], verbose=False, trim_lower=True) -> tuple[np.ndarray, int]:
    n_items = problems[0].n_items
    proxy_weights = np.linspace(1, 1.01, n_items)
    true_value = [problem.solution @ problem.payoffs for problem in problems]

    model = gp.Model("Inverse Knapsack (Weights)")
    model.Params.PoolSearchMode = 2
    model.Params.PoolSolutions = 10

    w = model.addMVar((n_items), lb=1, vtype=GRB.INTEGER, name="weights")

    model.setObjective(w.sum())

    model.addConstr(w.sum() >= (n_items + 1) * n_items / 2)
    model.addConstrs(w[i] <= n_items for i in range(n_items))

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

            if new_value < true_value[i]:
                # Experiment
                if trim_lower:
                    selected_sum = new_solution @ w.X
                    model.addConstr(new_solution @ w >= selected_sum + 1)
                else:
                    continue

            elif new_value == true_value[i]:
                continue
            else:
                model.addConstr(new_solution @ w >= problem.capacity + 1)

            new_constraint = True

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

    p = model.addMVar((n_items), lb=1, ub=n_items, vtype=GRB.INTEGER, name="payoffs")

    model.setObjective(p.sum())

    model.addConstr(p.sum() == (n_items + 1) * n_items / 2)

    model.optimize()

    if model.Status == GRB.INFEASIBLE:
            raise ValueError("Problem is Infeasible!")

    #solutions = [[problem.solution] for problem in problems]

    while True:
        new_constraint = False

        for i, problem in enumerate(problems):
            new_solution = problem.payoff_solve(p.X)

            if new_solution @ p.X <= problem.solution @ p.X:
                continue
            
            if verbose:
                print(i)
                print(problem.solution, problem.solution @ p.X, problem.solution @ problem.payoffs)
                print(new_solution, new_solution @ p.X)
                print(problem.payoffs)
                print(p.X)

            new_constraint = True
            model.addConstr(problem.solution @ p >= new_solution @ p)

        if not new_constraint:
            break

        model.optimize()

        if model.Status == GRB.INFEASIBLE:
            raise ValueError("Problem is Infeasible!")

    print() 
    print(problem.payoffs)
    print(p.X)

    return p.X.astype(np.int64), model.SolCount


def inverse_delta_payoffs(problems: list[KP], verbose=False) -> tuple[np.ndarray, int]:
    n_items = problems[0].n_items

    model = gp.Model("Inverse Knapsack (Payoffs)")

    model.Params.PoolSearchMode = 2
    model.Params.PoolSolutions = 10

    p = model.addMVar((n_items), lb=1, ub=n_items, vtype=GRB.INTEGER, name="payoffs")
    delta = model.addMVar((len(problems)), name="delta")

    model.setObjective(delta.sum(), GRB.MAXIMIZE)

    model.addConstr(p.sum() == (n_items + 1) * n_items / 2)
    for i, problem in enumerate(problems):
        model.addConstr(delta[i] <= problem.solution @ p)

    model.optimize()

    if model.Status == GRB.INFEASIBLE:
            raise ValueError("Problem is Infeasible!")

    solutions = [[problem.solution] for problem in problems]

    while True:
        new_constraint = False

        for i, problem in enumerate(problems):
            new_solution = problem.payoff_solve(p.X)

            if new_solution @ p.X <= problem.solution @ p.X:
                continue
            
            if duplicate_array(solutions[i], new_solution):
                continue

            if verbose:
                print(i, new_solution)
                print(problem.solution)
                print()
            new_constraint = True

            model.addConstr(delta[i] >= new_solution @ p)
            solutions[i].append(new_solution)

        if not new_constraint:
            break

        model.optimize()
        if model.Status == GRB.INFEASIBLE:
            raise ValueError("Problem is Infeasible!")

    if verbose:
        print(problem.payoffs)
        print(p.X)

    return p.X.astype(np.int64), model.SolCount


def inverse_wang_payoffs(problems: list[KP], verbose=False) -> tuple[np.ndarray, int]:

    n_items = problems[0].n_items

    model = gp.Model("Inverse Knapsack (Payoffs)")

    model.Params.PoolSearchMode = 2
    model.Params.PoolSolutions = 10

    p = model.addMVar((n_items), lb=1, ub=n_items, vtype=GRB.INTEGER)

    model.setObjective(gp.quicksum(p @ problem.solution for problem in problems), GRB.MAXIMIZE)

    model.addConstr(p.sum() <=  (n_items + 1) * n_items / 2)

    model.optimize()

    if model.Status == GRB.INFEASIBLE:
        raise ValueError("Problem is Infeasible!")
    
    while True:
        new_constraint = False

        for i, problem in enumerate(problems):
            new_solution = problem.payoff_solve(p.X)

            if new_solution @ p.X <= problem.solution @ p.X:
                continue

            new_constraint = True
            model.addConstr(problem.solution @ p >= new_solution @ p)

        if not new_constraint:
            break

        model.optimize()

        if model.Status == GRB.INFEASIBLE:
            raise ValueError("Problem is Infeasible!")

    return p.X.astype(np.int64), model.SolCount


if __name__ == "__main__":

    start = time()
    rng = np.random.default_rng(0)
    n_items = 10
    extras = n_items
    repeat = 10
    extras_mae_errors = []
    extras_mse_errors = []
    extra_solutions = []
    verbose = False

    approach = "payoffs"

    print(approach)
    if approach == "weights":
        problem_sets = [generate_weight_problems(n_items + extras, n_items, capacity_perc=0.5, rng=rng)
                        for _ in range(repeat)]

        for extra in range(extras):
            index = n_items + extra
            abs_errors = []
            square_errors = []
            num_sols = []
            for problems in problem_sets:
                weights = problems[0].weights
                inv_weights, sols = inverse_weights(problems[:index], verbose)
                abs_errors.append(abs(weights - inv_weights).sum())
                square_errors.append(((weights - inv_weights)**2).sum())
                num_sols.append(sols)

            extras_mae_errors.append(np.mean(abs_errors))
            extras_mse_errors.append(np.mean(square_errors))
            extra_solutions.append(np.mean(num_sols))

    if approach == "payoffs":
        payoff_bonus = n_items
        problem_sets = [generate_payoff_problems(n_items + payoff_bonus + extras, n_items, capacity_perc=[0.5], rng=rng)
                        for _ in range(repeat)]

        for extra in range(extras):
            index = n_items + payoff_bonus + extra
            abs_errors = []
            square_errors = []
            num_sols = []

            for problems in problem_sets:
                payoffs = problems[0].payoffs
                inv_payoffs, sols = inverse_wang_payoffs(problems[:index], verbose)
                abs_errors.append(abs(payoffs - inv_payoffs).sum())
                square_errors.append(((payoffs - inv_payoffs)**2).sum())
                num_sols.append(sols)

            extras_mae_errors.append(np.mean(abs_errors))
            extras_mse_errors.append(np.mean(square_errors))
            extra_solutions.append(np.mean(num_sols))

            print(abs_errors)

    print(extras_mae_errors)
    print(extra_solutions)

    print(time() - start )