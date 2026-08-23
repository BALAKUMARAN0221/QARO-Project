"""
Sanity-check script: confirms Qiskit + QAOA work correctly on your machine.
This solves a tiny toy problem (Max-Cut on a 4-node graph) - NOT your
warehouse problem yet. Just a "does my environment work" check.
"""
import networkx as nx
from qiskit_optimization.applications import Maxcut
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import Sampler

# A tiny 4-node graph (like a simple square)
G = nx.Graph()
G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0)])

maxcut = Maxcut(G)
qp = maxcut.to_quadratic_program()

qaoa = QAOA(sampler=Sampler(), optimizer=COBYLA(), reps=1)
solver = MinimumEigenOptimizer(qaoa)
result = solver.solve(qp)

print("✅ QAOA ran successfully on your machine.")
print("Best solution found:", result.x)
print("Objective value:", result.fval)