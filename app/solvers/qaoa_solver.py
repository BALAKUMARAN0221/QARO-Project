"""
QAOA Solver (Module 6).

Solves the QUBO/QuadraticProgram (built by qubo_formulation.py) using the
Quantum Approximate Optimization Algorithm on a quantum simulator.
"""
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import Sampler
from qiskit_optimization.algorithms import MinimumEigenOptimizer


def solve_with_qaoa(qp, reps=1, optimizer=None):
    """
    Solves a QuadraticProgram using QAOA on the Qiskit Aer simulator.
    Returns the qiskit-optimization result object (result.x = solution bits).
    """
    if optimizer is None:
        optimizer = COBYLA()

    qaoa = QAOA(sampler=Sampler(), optimizer=optimizer, reps=reps)
    solver = MinimumEigenOptimizer(qaoa)
    result = solver.solve(qp)
    return result


if __name__ == "__main__":
    import time
    from app.warehouse.layout import generate_warehouse, generate_pick_list
    from app.solvers.qubo_formulation import formulate_qubo

    G = generate_warehouse(rows=5, cols=5)
    pick_list = generate_pick_list(G, num_items=3, seed=1)
    qp, tsp, node_mapping = formulate_qubo(G, pick_list)

    print(f"Solving TSP over {len(node_mapping)} locations with QAOA...")
    start = time.time()
    result = solve_with_qaoa(qp, reps=1)
    elapsed = time.time() - start

    route_indices = tsp.interpret(result.x)
    route_coords = [node_mapping[i] for i in route_indices]

    print(f"\nQAOA finished in {elapsed:.1f}s")
    print(f"Route (QUBO indices): {route_indices}")
    print(f"Route (warehouse coordinates): {route_coords}")
    print(f"Total route distance: {result.fval}")

    # --- Brute-force ground truth, since this instance is small enough ---
    import itertools

    def brute_force_optimal(sub_graph):
        n = sub_graph.number_of_nodes()
        best = None
        for perm in itertools.permutations(range(1, n)):
            route = [0] + list(perm) + [0]
            dist = sum(sub_graph[route[i]][route[i+1]]['weight'] for i in range(len(route)-1))
            if best is None or dist < best[0]:
                best = (dist, route)
        return best

    from app.solvers.qubo_formulation import build_subproblem_graph
    sub_graph, _ = build_subproblem_graph(G, pick_list)
    optimal_dist, optimal_route = brute_force_optimal(sub_graph)

    print(f"\nBrute-force optimal distance: {optimal_dist}")
    print(f"QAOA result matches optimal: {result.fval == optimal_dist}")