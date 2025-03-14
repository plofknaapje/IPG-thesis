import gurobipy as gp
from gurobipy import GRB
import numpy as np
from time import time
from utils import duplicate_array
from problems.knapsack_problem import KnapsackProblem


def generate_weight_problems(
    n_problems=10,
    r=100,
    n_items=10,
    capacity: float | np.ndarray = 0.5,
    corr=True,
    rng=None,
) -> list[KnapsackProblem]:
    if rng is None:
        rng = np.random.default_rng()
    problems = []
    weights = rng.integers(1, r + 1, n_items)

    if corr:
        lower = np.maximum(weights - r / 10, 1)
        upper = np.minimum(weights + r / 10, r + 1)

    if type(capacity) is float:
        capacity = np.ones(n_problems) * capacity

    for i in range(n_problems):
        if corr:
            problem_payoffs = rng.integers(lower, upper)
        else:
            problem_payoffs = rng.integers(1, r + 1, n_items)

        problem = KnapsackProblem(
            problem_payoffs, weights, int(capacity[i] * weights.sum())
        )
        problem.solve()
        problems.append(problem)

    return problems


def generate_weight_range_problems(
    n_problems=10, n_items=10, capacity: float | np.ndarray = 0.5, rng=None
) -> list[KnapsackProblem]:
    if rng is None:
        rng = np.random.default_rng()
    problems = []
    weights = np.arange(1, n_items + 1)
    payoffs = np.arange(1, n_items + 1)
    rng.shuffle(weights)

    if type(capacity) is float:
        capacity = np.ones(n_problems) * capacity

    for i in range(n_problems):
        problem_payoffs = payoffs.copy()
        rng.shuffle(problem_payoffs)
        problem = KnapsackProblem(
            problem_payoffs, weights, int(capacity[i] * weights.sum())
        )
        problem.solve()
        problems.append(problem)

    return problems


def generate_payoff_problems(
    n_problems=10,
    r=100,
    n_items=10,
    capacity: float | np.ndarray = 0.5,
    corr=True,
    rng=None,
) -> list[KnapsackProblem]:
    if rng is None:
        rng = np.random.default_rng()
    problems = []
    payoffs = rng.integers(1, r + 1, n_items)

    if corr:
        lower = np.maximum(payoffs - r / 10, 1)
        upper = np.minimum(payoffs + r / 10, r + 1)

    if type(capacity) is float:
        capacity = np.ones(n_problems) * capacity

    for i in range(n_problems):
        if corr:
            problem_weights = rng.integers(lower, upper)
        else:
            problem_weights = rng.integers(1, r + 1, n_items)

        problem = KnapsackProblem(
            payoffs, problem_weights, int(capacity[i] * problem_weights.sum())
        )
        problem.solve()
        problems.append(problem)

    return problems


def generate_payoff_range_problems(
    n_problems=10, n_items=10, capacity: float | np.ndarray = 0.5, rng=None
) -> list[KnapsackProblem]:
    if rng is None:
        rng = np.random.default_rng()
    problems = []
    weights = np.arange(1, n_items + 1)
    payoffs = np.arange(1, n_items + 1)
    rng.shuffle(payoffs)

    if isinstance(capacity, float):
        capacity = np.ones(n_problems) * capacity

    for i in range(n_problems):
        problem_weights = weights.copy()
        rng.shuffle(problem_weights)
        problem = KnapsackProblem(
            payoffs, problem_weights, int(capacity[i] * problem_weights.sum())
        )
        problem.solve()
        problems.append(problem)

    return problems


def inverse_weights(
    problems: list[KnapsackProblem], verbose=False, trim_lower=True
) -> tuple[np.ndarray, int]:
    n_items = problems[0].n
    true_value = [problem.solution @ problem.payoffs for problem in problems]

    model = gp.Model("Inverse Knapsack (Weights)")
    model.Params.PoolSearchMode = 2
    model.Params.PoolSolutions = 10

    w = model.addMVar((n_items), lb=1, vtype=GRB.INTEGER, name="weights")

    model.setObjective(w.sum())

    model.addConstr(w.sum() >= problems[0].weights.sum())
    model.addConstrs(w[i] <= n_items for i in range(n_items))

    for problem in problems:
        model.addConstr(problem.solution @ w <= problem.capacity)

    model.optimize()
    if model.Status == GRB.INFEASIBLE:
        raise ValueError("Problem is Infeasible!")

    while True:
        new_constraint = False

        for i, problem in enumerate(problems):
            new_x = problem.solve(weights=w.X)
            new_obj = new_x @ problem.payoffs

            if new_obj < true_value[i] and trim_lower:
                selected_sum = new_x @ w.X
                model.addConstr(new_x @ w >= selected_sum + 1)

            elif new_obj <= true_value[i]:
                continue
            else:
                model.addConstr(new_x @ w >= problem.capacity + 1)

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

    return w.X.astype(np.short), model.SolCount


def inverse_payoffs(
    problems: list[KnapsackProblem], verbose=False
) -> tuple[np.ndarray, int]:
    n_items = problems[0].n
    # values = np.arange(1, n_items + 1)
    # proxy_weights = np.linspace(1, 1.01, n_items)

    model = gp.Model("Inverse Knapsack (Payoffs)")

    model.Params.PoolSearchMode = 2
    model.Params.PoolSolutions = 10

    p = model.addMVar((n_items), lb=1, ub=n_items, vtype=GRB.INTEGER, name="payoffs")

    model.setObjective(p.sum())

    model.addConstr(p.sum() == problems[0].payoffs.sum())

    model.optimize()

    if model.Status == GRB.INFEASIBLE:
        raise ValueError("Problem is Infeasible!")

    # solutions = [[problem.solution] for problem in problems]

    while True:
        new_constraint = False

        for i, problem in enumerate(problems):
            new_x = problem.solve(payoffs=p.X)

            if new_x @ p.X <= problem.solution @ p.X:
                continue

            if verbose:
                print(i)
                print(
                    problem.solution,
                    problem.solution @ p.X,
                    problem.solution @ problem.payoffs,
                )
                print(new_x, new_x @ p.X)
                print(problem.payoffs)
                print(p.X)

            new_constraint = True
            model.addConstr(problem.solution @ p >= new_x @ p)

        if not new_constraint:
            break

        model.optimize()

        if model.Status == GRB.INFEASIBLE:
            raise ValueError("Problem is Infeasible!")

    print()
    print(problem.payoffs)
    print(p.X)

    return p.X.astype(np.short), model.SolCount


def inverse_delta_payoffs(
    problems: list[KnapsackProblem], verbose=False
) -> tuple[np.ndarray, int]:
    n_items = problems[0].n

    model = gp.Model("Inverse Knapsack (Payoffs)")

    model.Params.PoolSearchMode = 2
    model.Params.PoolSolutions = 10

    p = model.addMVar((n_items), lb=1, vtype=GRB.INTEGER, name="payoffs")
    delta = model.addMVar((len(problems)), name="delta")

    model.setObjective(delta.sum(), GRB.MAXIMIZE)

    model.addConstr(p.sum() == problems[0].payoffs.sum())
    for i, problem in enumerate(problems):
        model.addConstr(delta[i] <= problem.solution @ p)

    model.optimize()

    if model.Status == GRB.INFEASIBLE:
        raise ValueError("Problem is Infeasible!")

    solutions = [[problem.solution] for problem in problems]

    while True:
        new_constraint = False

        for i, problem in enumerate(problems):
            new_x = problem.solve(payoffs=p.X)

            if new_x @ p.X <= problem.solution @ p.X:
                continue

            if duplicate_array(solutions[i], new_x):
                continue

            if verbose:
                print(i, new_x)
                print(problem.solution)
                print()
            new_constraint = True

            model.addConstr(delta[i] >= new_x @ p)
            solutions[i].append(new_x)

        if not new_constraint:
            break

        model.optimize()
        if model.Status == GRB.INFEASIBLE:
            raise ValueError("Problem is Infeasible!")

    if verbose:
        print(problem.payoffs)
        print(p.X)

    return p.X.astype(np.short), model.SolCount


def inverse_direct_payoffs(
    problems: list[KnapsackProblem], verbose=False
) -> tuple[np.ndarray, int]:
    n_items = problems[0].n

    model = gp.Model("Inverse Knapsack (Payoffs)")

    model.Params.PoolSearchMode = 2
    model.Params.PoolSolutions = 10

    p = model.addMVar((n_items), lb=1, vtype=GRB.INTEGER)

    model.setObjective(
        gp.quicksum(p @ problem.solution for problem in problems), GRB.MAXIMIZE
    )

    model.addConstr(p.sum() <= problems[0].payoffs.sum())

    model.optimize()

    if model.Status == GRB.INFEASIBLE:
        raise ValueError("Problem is Infeasible!")

    while True:
        new_constraint = False

        for i, problem in enumerate(problems):
            new_solution = problem.solve(payoffs=p.X)

            if new_solution @ p.X <= problem.solution @ p.X:
                continue

            new_constraint = True
            model.addConstr(problem.solution @ p >= new_solution @ p)

        if not new_constraint:
            break

        model.optimize()

        if model.Status == GRB.INFEASIBLE:
            raise ValueError("Problem is Infeasible!")

    return p.X.astype(np.short), model.SolCount


def inverse_wang_payoffs(
    problems: list[KnapsackProblem], verbose=False
) -> tuple[np.ndarray, int]:
    n_items = problems[0].n
    base_vector = np.full((n_items), problems[0].payoffs.mean(), dtype=np.short)

    model = gp.Model("Inverse Knapsack (Payoffs)")

    model.Params.PoolSearchMode = 2
    model.Params.PoolSolutions = 10

    h = model.addMVar((n_items), lb=0, vtype=GRB.INTEGER)
    l = model.addMVar((n_items), lb=0, vtype=GRB.INTEGER)

    model.setObjective(h.sum() + l.sum(), GRB.MINIMIZE)

    model.addConstr((base_vector + h - l).sum() <= problems[0].payoffs.sum())

    model.optimize()

    if model.Status == GRB.INFEASIBLE:
        raise ValueError("Problem is Infeasible!")

    while True:
        new_constraint = False

        for i, problem in enumerate(problems):
            p = base_vector + h.X - l.X
            new_solution = problem.solve(payoffs=p)

            if new_solution @ p <= problem.solution @ p:
                continue

            new_constraint = True
            model.addConstr(
                problem.solution @ (base_vector + h - l)
                >= new_solution @ (base_vector + h - l)
            )

        if not new_constraint:
            break

        model.optimize()

        if model.Status == GRB.INFEASIBLE:
            raise ValueError("Problem is Infeasible!")

    return (base_vector + h.X - l.X).astype(np.short), model.SolCount


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
        problem_sets = [
            generate_weight_range_problems(
                n_items + extras, n_items, capacity=0.5, rng=rng
            )
            for _ in range(repeat)
        ]

        for extra in range(extras):
            index = n_items + extra
            abs_errors = []
            square_errors = []
            num_sols = []
            for problems in problem_sets:
                weights = problems[0].weights
                inv_weights, sols = inverse_weights(problems[:index], verbose)
                abs_errors.append(abs(weights - inv_weights).sum())
                square_errors.append(((weights - inv_weights) ** 2).sum())
                num_sols.append(sols)

            extras_mae_errors.append(np.mean(abs_errors))
            extras_mse_errors.append(np.mean(square_errors))
            extra_solutions.append(np.mean(num_sols))

    if approach == "payoffs":
        problem_sets = [
            generate_payoff_problems(
                n_items * n_items, 100, n_items, capacity=0.5, rng=rng
            )
            for _ in range(repeat)
        ]

        for extra in range(n_items):
            index = n_items * (n_items - 1) + extra
            abs_errors = []
            square_errors = []
            num_sols = []

            for problems in problem_sets:
                payoffs = problems[0].payoffs
                inv_payoffs, sols = inverse_delta_payoffs(problems[:index], verbose)
                abs_errors.append(abs(payoffs - inv_payoffs).sum())
                square_errors.append(((payoffs - inv_payoffs) ** 2).sum())
                num_sols.append(sols)

            extras_mae_errors.append(np.mean(abs_errors))
            extras_mse_errors.append(np.mean(square_errors))
            extra_solutions.append(np.mean(num_sols))

            print(abs_errors)

    print(extras_mae_errors)
    print(extra_solutions)

    print(time() - start)
