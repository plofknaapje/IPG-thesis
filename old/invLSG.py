import gurobipy as gp
from gurobipy import GRB
import numpy as np
import copy
from utils import duplicate_array
from problems.location_selection_game import LocationSelectionGame


def inverse_LSG(obss: list[LocationSelectionGame], sols: list[np.ndarray]) -> tuple:
    observations = list(range(len(obss)))
    incumbents = obss[0].incumbents
    customers = obss[0].customers
    locations = obss[0].locations
    observations = list(range(len(obss)))

    sub_sols = {(o, i): [] for i in incumbents for o in observations}

    while True:
        model = gp.Model("MasterProblem")

        # Parameters of interest
        alpha = model.addVar(lb=0, ub=1, name="alpha")
        betas = model.addMVar(len(incumbents), lb=0, ub=1, name="beta")
        # Meta variable
        delta = model.addMVar((len(observations), len(incumbents)), name="delta")

        # Support variables
        true_f = model.addVars(
            len(observations),
            len(incumbents),
            len(customers),
            lb=0,
            ub=1,
            name="true_f",
        )
        alt_f_indices = [
            (o, i, j, s)
            for o in observations
            for i in incumbents
            for j in customers
            for s in range(len(sub_sols[o, i]))
        ]
        alt_f = model.addVars(alt_f_indices, lb=0, ub=1, name="alt_f")

        true_profit = model.addVars(
            len(observations), len(incumbents), name="true_profit"
        )
        alt_profit_indices = [
            (o, i, s)
            for o in observations
            for i in incumbents
            for s in range(len(sub_sols[o, i]))
        ]
        alt_profit = model.addVars(alt_profit_indices, name="alt_profit")

        M = model.addVars(len(obss), len(incumbents), len(customers), name="M")
        util = model.addVars(
            len(obss), len(incumbents), len(customers), len(locations), name="util"
        )

        model.setObjective(delta.sum())

        model.addConstr(betas[0] + betas[1] <= 1)

        # Utility
        model.addConstrs(
            util[o, i, j, k]
            == betas[i]
            + alpha * obss[o].norm_distance[j, k]
            + (1 - alpha) * obss[o].conveniences[k]
            for o in observations
            for i in incumbents
            for j in customers
            for k in locations
        )

        # Big M for calculating f
        model.addConstrs(
            M[o, i, j] == gp.min_(util[o, i, j, k] for k in locations)
            for o in observations
            for i in incumbents
            for j in customers
        )

        for o, obs in enumerate(obss):
            viable = obs.viable
            for i in incumbents:
                true_sol = sols[o]
                # Define true_f
                model.addConstrs(
                    true_f[o, i, j]
                    * gp.quicksum(
                        gp.quicksum(
                            sols[o][ii, k] * util[o, ii, j, k]
                            for ii in incumbents
                            if ii != i
                        )
                        + sols[o][i, k] * util[o, i, j, k]
                        for k in viable[j]
                    )
                    <= gp.quicksum(sols[o][i, k] * util[o, i, j, k] for k in viable[j])
                    for j in customers
                )

                model.addConstrs(
                    true_f[o, i, j] * M[o, i, j]
                    <= gp.quicksum(sols[o][i, k] * util[o, i, j, k] for k in viable[j])
                    for j in customers
                )

                # Define alt f
                for s, alt_sol in enumerate(sub_sols[o, i]):
                    model.addConstrs(
                        alt_f[o, i, j, s]
                        * gp.quicksum(
                            gp.quicksum(
                                sols[o][ii, k] * util[o, ii, j, k]
                                for ii in incumbents
                                if ii != i
                            )
                            + alt_sol[k] * util[o, i, j, k]
                            for k in viable[j]
                        )
                        <= gp.quicksum(alt_sol[k] * util[o, i, j, k] for k in viable[j])
                        for j in customers
                    )

                    model.addConstrs(
                        alt_f[o, i, j, s] * M[o, i, j]
                        <= gp.quicksum(alt_sol[k] * util[o, i, j, k] for k in viable[j])
                        for j in customers
                    )

        for o in observations:
            for i in incumbents:
                model.addConstr(
                    true_profit[o, i]
                    <= gp.quicksum(
                        obs.margin[i, j] * obs.pop_count[j] * true_f[o, i, j]
                        for j in customers
                    )
                    - gp.quicksum(obs.costs[i, k] * true_sol[i, k] for k in locations)
                )

                for s, alt_sol in enumerate(sub_sols[o, i]):
                    model.addConstr(
                        alt_profit[o, i, s]
                        <= gp.quicksum(
                            obs.margin[i, j] * obs.pop_count[j] * alt_f[o, i, j, s]
                            for j in customers
                        )
                        - gp.quicksum(obs.costs[i, k] * alt_sol[k] for k in locations)
                    )

                    model.addConstr(
                        delta[o, i] >= alt_profit[o, i, s] - true_profit[o, i]
                    )

        print("Optimizing")
        model.optimize()

        if model.Status == GRB.INFEASIBLE:
            raise ValueError("Problem is Infeasible!")

        new_alpha = alpha.X
        new_betas = betas.X

        new_solution = False
        for o in observations:
            for i in incumbents:
                obs = obss[o]
                obs.update_alpha_beta(new_alpha, new_betas)
                new_sol = lsg.solve_partial_LSG(obs, sols[o], i)
                if lsg.duplicate_array(sub_sols[o, i], new_sol):
                    continue

                # print(new_sol)
                sub_sols[o, i].append(new_sol)
                new_solution = True

        model.close()

        if not new_solution:
            break

    return new_alpha, new_betas


if __name__ == "__main__":
    alpha = 0.4
    betas = [0.4, 0.6]

    errors = []
    for i in range(1):
        print()
        print(i)

        obss = [lsg.random_LSG(2, 10, 10, alpha=alpha, betas=betas) for _ in range(15)]

        solutions = [lsg.solve_LSG(obs) for obs in obss]
        print(solutions)

        learned_alpha, learned_betas = inverse_LSG(obss, solutions)

        error = (
            (alpha - learned_alpha) ** 2
            + (betas[0] - learned_betas[0]) ** 2
            + (betas[1] - learned_betas[1]) ** 2
        )
        errors.append(error)
    print(np.min(errors), np.mean(errors), np.max(errors))
