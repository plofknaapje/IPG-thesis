from methods.local_inverse_kp import generate_problem

problem = generate_problem(20, capacity=0.5)

print(problem.solve_greedy())