import gurobipy as gp
from gurobipy import GRB
import numpy as np
import copy

import lsg

def inverse_LSG(obss: list[lsg.LSG], sols: list[np.ndarray]) -> list:
    incumbents = obss[0].incumbents
    customers = obss[0].customers
    locations = obss[0].locations
    observations = list(range(len(obss)))

    solutions = {(o, i): [] for i in incumbents for o in range(len(obss))}

    while True:

        model = gp.Model("MasterProblem")

        alpha = model.addVar(lb=0, ub=1, name="alpha")
        betas = model.addVars(len(incumbents), lb=0, ub=1, name="beta")
        # x = model.addVars((len(observations), len(incumbents), len(locations)), vtype=GRB.BINARY, name="x")
        delta = model.addVars(len(obss), len(incumbents), name="delta")
        true_f = model.addVars(len(obss), len(incumbents), len(customers), 
                               lb=0, ub=1, name="f_true")
        new_f = model.addVars(len(obss), len(incumbents), len(customers), 
                              max(len(item) for _, item in solutions.items()),
                              lb=0, ub=1, name="f_new")
        M = model.addVars(len(obss), len(incumbents), len(customers), name="M")
        util = model.addVars(len(obss), len(incumbents), len(customers), len(locations), name="util")

        model.setObjective(gp.quicksum(delta[o, i] for o in observations for i in incumbents))

        model.addConstrs(
            M[o, i, j] == gp.min_(util[o, i, j, k] for k in locations)
            for o in observations for i in incumbents for j in customers
        )

        model.addConstrs(
            util[o, i, j, k] == betas[i] + alpha * obss[o].norm_distance[j, k] + 
                                         (1 - alpha) * obss[o].conveniences[k]
            for o in observations for i in incumbents for j in customers for k in locations
        )
        
        for o, obs in enumerate(obss):
            viable = {j: [k for k in obs.locations if obs.distance[j, k] < obs.max_dist]
                      for j in obs.customers}
            for i in incumbents:
                true_sol = sols[o]
                model.addConstrs(
                    true_f[o, i, j] * 
                    gp.quicksum(gp.quicksum(true_sol[ii, k] * util[o, ii, j, k] 
                                            for ii in incumbents if ii != i) +
                                true_sol[i, k] * util[o, i, j, k]
                                for k in viable[j]) <=
                    gp.quicksum(true_sol[i, k] * util[o, i, j, k] for k in viable[j])
                    for j in customers
                )
                model.addConstrs(
                    true_f[o, i, j] * M[o, i, j] <= 
                    gp.quicksum(true_sol[i, k] * util[o, i, j, k] for k in viable[j])
                    for o in observations for i in incumbents for j in customers
                )


                true_profit = gp.quicksum(obs.margin[i,j] * obs.pop_count[j] * true_f[o, i, j] 
                                          for j in customers) - \
                    gp.quicksum(obs.costs[i,k] * true_sol[i,k] for k in locations)

                for s, new_sol in enumerate(solutions[o, i]):
                    model.addConstrs(
                        new_f[o, i, j, s] * 
                        gp.quicksum(gp.quicksum(true_sol[ii, k] * util[o, ii, j, k] 
                                                for ii in incumbents if ii != i) +
                                    new_sol[k] * util[o, i, j, k]
                                    for k in viable[j]) <=
                        gp.quicksum(new_sol[k] * util[o, i, j, k] for k in viable[j]) 
                        for j in customers 
                    )

                    model.addConstrs(
                        new_f[o, i, j, s] * M[o, i, j] <= 
                        gp.quicksum(new_sol[k] * util[o, i, j, k] for k in viable[j]) 
                        for o in observations for i in incumbents for j in customers
                    )

                    new_profit = gp.quicksum(obs.margin[i,j] * obs.pop_count[j] * new_f[o, i, j, s] 
                                             for j in customers) - \
                        gp.quicksum(obs.costs[i,k] * new_sol[k] for k in locations)
                    model.addConstr(delta[o,i] >= new_profit - true_profit)

        model.optimize()

        new_alpha = float(alpha.X)
        new_betas = [betas[i].x for i in incumbents]
        print(new_alpha, new_betas)

        new_solution = False

        for o, obs in enumerate(obss):
            temp_obs = copy.copy(obs)
            temp_obs.update_alpha_beta(new_alpha, new_betas)
            new_x =  lsg.solve_LSG(temp_obs)
            for i in incumbents:
                if not lsg.duplicate_array(solutions[o, i], new_x[i, :]):
                    solutions[o, i].append(new_x[i, :])
                    print(new_x[i, :])
                    new_solution = True

        if not new_solution:
            break

    return [new_alpha, new_betas]

alpha = 0.4
betas = [0.4, 0.6]
obss = [lsg.random_LSG(2, 10, 10, alpha=alpha, betas=betas, seed=i) for i in range(4)]

solutions = [lsg.solve_LSG(obs) for obs in obss]

print(inverse_LSG(obss, solutions))