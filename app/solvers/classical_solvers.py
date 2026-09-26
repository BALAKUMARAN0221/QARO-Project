"""
Classical Baseline Solvers (Module 7).

Solves the SAME TSP sub-problem that qaoa_solver.py solves (the complete
graph built by qubo_formulation.build_subproblem_graph), but with classical
methods instead of QAOA. This gives a fair, apples-to-apples comparison:
every solver in this project operates on the identical sub_graph instance.

Two baselines:
1. Nearest Neighbor  - fast, simple greedy heuristic. Usually not optimal,
   but a common "does quantum even beat the dumb baseline" reference point.
2. Google OR-Tools    - a strong, industry-standard classical TSP solver.
   Near-optimal (often exactly optimal) on small instances like ours, so
   it's the more serious classical benchmark for QAOA to be compared against.
"""
import numpy as np
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from qiskit_optimization.converters import QuadraticProgramToQubo
import neal


def solve_nearest_neighbor(sub_graph, start=0):
    """
    Greedy Nearest Neighbor heuristic on the sub_graph produced by
    qubo_formulation.build_subproblem_graph.

    Starting at `start` (index 0 is always the depot), repeatedly moves to
    the closest unvisited node until all nodes are visited, then returns
    to the start. Returns (route, total_distance) where route is a list of
    sub_graph node indices, e.g. [0, 2, 1, 3, 0].
    """
    n = sub_graph.number_of_nodes()
    unvisited = set(range(n))
    unvisited.remove(start)

    route = [start]
    current = start
    total_distance = 0

    while unvisited:
        nearest = min(unvisited, key=lambda node: sub_graph[current][node]['weight'])
        total_distance += sub_graph[current][nearest]['weight']
        route.append(nearest)
        unvisited.remove(nearest)
        current = nearest

    # Return to depot
    total_distance += sub_graph[current][start]['weight']
    route.append(start)

    return route, total_distance


def _build_distance_matrix(sub_graph):
    """Converts the sub_graph into a dense distance matrix OR-Tools expects."""
    n = sub_graph.number_of_nodes()
    matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                matrix[i][j] = sub_graph[i][j]['weight']
    return matrix


def solve_with_ortools(sub_graph, start=0):
    """
    Solves the TSP over sub_graph using Google OR-Tools' routing solver.

    Uses PATH_CHEAPEST_ARC to construct an initial solution, then improves
    it with guided local search. Returns (route, total_distance) in the
    same format as solve_nearest_neighbor, so results are directly
    comparable to the other solvers in this project.
    """
    n = sub_graph.number_of_nodes()
    distance_matrix = _build_distance_matrix(sub_graph)

    manager = pywrapcp.RoutingIndexManager(n, 1, start)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return distance_matrix[from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_parameters.time_limit.FromSeconds(5)

    solution = routing.SolveWithParameters(search_parameters)

    if solution is None:
        raise RuntimeError("OR-Tools failed to find a solution for this instance.")

    route = []
    index = routing.Start(0)
    while not routing.IsEnd(index):
        route.append(manager.IndexToNode(index))
        index = solution.Value(routing.NextVar(index))
    route.append(manager.IndexToNode(index))  # closes the loop back to depot

    total_distance = solution.ObjectiveValue()

    return route, total_distance


def solve_with_simulated_annealing(qp, tsp, sub_graph, num_reads=200, seed=None):
    """
    Solves the SAME QUBO that QAOA solves (from qubo_formulation.formulate_qubo),
    but with classical Simulated Annealing (D-Wave's `neal` sampler) instead of
    a quantum circuit. Unlike Nearest Neighbor/OR-Tools, which work directly on
    the distance graph, this operates on the QUBO encoding - giving the most
    direct possible comparison to QAOA, since both solve the literal same
    optimization problem, just with a classical vs. quantum method.

    Note: because the QUBO enforces "visit every node once" via penalty terms
    rather than a hard constraint, a low-quality sample can occasionally
    produce an invalid route (skips or repeats a node). When that happens,
    this returns (route, None) so the caller can detect and discard it - a
    real trade-off worth noting when comparing against OR-Tools/NN, which
    always produce valid routes by construction.

    Returns (route, total_distance) in the same format as the other solvers,
    or (route, None) if the sample was infeasible.
    """
    converter = QuadraticProgramToQubo()
    qubo = converter.convert(qp)

    # Flatten the QUBO's linear + quadratic terms into the {(i,j): coeff}
    # dict format neal's sampler expects.
    Q = {}
    for i, coeff in qubo.objective.linear.to_dict().items():
        Q[(i, i)] = Q.get((i, i), 0.0) + coeff
    for (i, j), coeff in qubo.objective.quadratic.to_dict().items():
        Q[(i, j)] = Q.get((i, j), 0.0) + coeff

    sampler = neal.SimulatedAnnealingSampler()
    sampleset = sampler.sample_qubo(Q, num_reads=num_reads, seed=seed)
    best_sample = sampleset.first.sample

    num_vars = qubo.get_num_binary_vars()
    x = np.array([best_sample[i] for i in range(num_vars)])
    route = tsp.interpret(x)

    n = sub_graph.number_of_nodes()
    is_valid = len(route) == n and len(set(route)) == n
    if not is_valid:
        return route, None

    full_route = route + [route[0]]  # close the loop, matching NN/OR-Tools format
    total_distance = sum(
        sub_graph[full_route[i]][full_route[i + 1]]['weight'] for i in range(len(full_route) - 1)
    )
    return full_route, total_distance


if __name__ == "__main__":
    from app.warehouse.layout import generate_warehouse, generate_pick_list
    from app.solvers.qubo_formulation import build_subproblem_graph, formulate_qubo

    # Same instance used in qaoa_solver.py's __main__, so results line up
    G = generate_warehouse(rows=5, cols=5)
    pick_list = generate_pick_list(G, num_items=3, seed=1)
    sub_graph, node_mapping = build_subproblem_graph(G, pick_list)

    print(f"Pick list: {pick_list}")
    print(f"Solving TSP over {len(node_mapping)} locations with classical baselines...\n")

    nn_route, nn_distance = solve_nearest_neighbor(sub_graph)
    nn_coords = [node_mapping[i] for i in nn_route]
    print("--- Nearest Neighbor ---")
    print(f"Route (indices): {nn_route}")
    print(f"Route (coordinates): {nn_coords}")
    print(f"Total distance: {nn_distance}\n")

    or_route, or_distance = solve_with_ortools(sub_graph)
    or_coords = [node_mapping[i] for i in or_route]
    print("--- Google OR-Tools ---")
    print(f"Route (indices): {or_route}")
    print(f"Route (coordinates): {or_coords}")
    print(f"Total distance: {or_distance}\n")

    qp, tsp, _ = formulate_qubo(G, pick_list)
    sa_route, sa_distance = solve_with_simulated_annealing(qp, tsp, sub_graph, num_reads=200, seed=1)
    print("--- Simulated Annealing ---")
    if sa_distance is None:
        print(f"Route (indices): {sa_route} -- INFEASIBLE sample, discarded\n")
    else:
        sa_coords = [node_mapping[i] for i in sa_route]
        print(f"Route (indices): {sa_route}")
        print(f"Route (coordinates): {sa_coords}")
        print(f"Total distance: {sa_distance}\n")

    # --- Compare against the brute-force optimal (same helper qaoa_solver.py uses) ---
    import itertools

    def brute_force_optimal(sub_graph):
        n = sub_graph.number_of_nodes()
        best = None
        for perm in itertools.permutations(range(1, n)):
            route = [0] + list(perm) + [0]
            dist = sum(sub_graph[route[i]][route[i + 1]]['weight'] for i in range(len(route) - 1))
            if best is None or dist < best[0]:
                best = (dist, route)
        return best

    optimal_dist, optimal_route = brute_force_optimal(sub_graph)
    print(f"Brute-force optimal distance: {optimal_dist}")
    print(f"Nearest Neighbor matches optimal: {nn_distance == optimal_dist}")
    print(f"OR-Tools matches optimal: {or_distance == optimal_dist}")
    print(f"Simulated Annealing matches optimal: {sa_distance == optimal_dist}")