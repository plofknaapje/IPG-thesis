import networkx as nx
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB
from dataclasses import dataclass

env = gp.Env("gurobi.log")

@dataclass
class KEP:
    "Class for KEP problems"
    nodes: list
    edges: list
    G: nx.DiGraph
    cycles: list | None

    def __init__(self, n: int, p: float, k=1, seed=42):
        self.G = nx.fast_gnp_random_graph(n=n, p=p, seed=seed, directed=True)
        self.nodes = list(self.G.nodes)
        self.edges = list(self.G.edges)
        self.cycles = None
    
    def add_cycles(self, max_length):
        if self.cycles is None:
            self.cycles = [tuple(cycle) for cycle in nx.simple_cycles(self.G, max_length)]


def simple_kep_solver(kep: KEP) -> list:
    try:
        m = gp.Model("KEP", env=env)
        m_edges = m.addVars(kep.edges, vtype=GRB.BINARY, name="t")
        m_nodes = m.addVars(kep.nodes, vtype=GRB.BINARY, name="p")

        m.setObjective(m_edges.sum(), GRB.MAXIMIZE)

        m.addConstrs(
            gp.quicksum(m_edges[i, o] for i, o in edges if i == node) == m_nodes[node]
            for node in kep.nodes)

        m.addConstrs(
            gp.quicksum(m_edges[i, o] for i, o in edges if o == node) == m_nodes[node]
            for node in kep.nodes)

        m.optimize()

        for v in m.getVars():
            print(f"{v.VarName} {v.X:g}")

        print(f"Obj: {m.ObjVal:g}")

        transplants = [edge for edge, v in m_edges.items() if v.X == 1]

        return transplants

    except gp.GurobiError as e:
        print(f"Error code {e.errno}: {e}")

    except AttributeError:
        print("Encountered an attribute error")

def cycle_kep_solver(kep: KEP, max_length: int) -> list:
    kep.add_cycles(max_length)

    try:
        m = gp.Model("cycle KEP", env=env)
        m_cycles = m.addVars(kep.cycles, vtype=GRB.BINARY, name="c")
        m_nodes = m.addVars(kep.nodes, vtype=GRB.BINARY, name="p")

        m.setObjective(gp.quicksum(len(cycle) * m_cycles[cycle]
                    for cycle in kep.cycles), GRB.MAXIMIZE)

        m.addConstrs(
            gp.quicksum(m_cycles[cycle] for cycle in kep.cycles if node in cycle) == m_nodes[node]
            for node in kep.nodes
        )

        m.optimize()

        for v in m.getVars():
            print(f"{v.VarName} {v.X:g}")

        print(f"Obj: {m.ObjVal:g}")

        cycles = [cycle for cycle, v in m_cycles.items() if v.X == 1]

        return cycles

    except gp.GurobiError as e:
        print(f"Error code {e.errno}: {e}")

    except AttributeError:
        print("Encountered an attribute error")


problem = KEP(20, 0.1)

transplants = simple_kep_solver(problem)
unused_edges = [edge for edge in edges if edge not in transplants]
G = nx.DiGraph()
G.add_nodes_from(nodes)
G.add_edges_from(edges)
pos = nx.spring_layout(G, seed=1)
nx.draw_networkx_nodes(G, pos)
nx.draw_networkx_edges(G, pos, edgelist=unused_edges)
nx.draw_networkx_edges(G, pos, edgelist=transplants, width=2, edge_color="red")
plt.show()


cycles = cycle_kep_solver(problem, 5)
print(cycles)

nx.draw_networkx_nodes(G, pos)
for cycle in cycles:
    edges = []
    i = cycle[-1]
    width = 1
    for j in cycle:
        edges.append([i, j])
        i = j
    nx.draw_networkx_edges(G, pos, edgelist=edges, width=width)
    width += 0.2
plt.show()
