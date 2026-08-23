"""
QUBO Formulation Engine (Module 4).

Converts a warehouse graph + pick list into the routing problem QAOA solves:
"find the shortest route starting and ending at the depot that visits every
pick-list location exactly once" - a Travelling Salesman Problem (TSP)
restricted to the required locations.

Approach:
1. The picker can walk between ANY two pick locations via the shortest path
   through the warehouse aisles (not just directly-adjacent grid cells).
   So we first compute the shortest-path distance between every pair of
   locations we actually care about (depot + pick list).
2. This turns our warehouse grid into a small COMPLETE graph (every node
   connected to every other node, weight = shortest walking distance)
   - the standard way to reduce "visit only these N points" into a
   textbook TSP instance.
3. Qiskit Optimization's built-in Tsp application then converts that
   complete graph directly into a QUBO model.
"""
import networkx as nx
from qiskit_optimization.applications import Tsp


def build_subproblem_graph(warehouse_graph, pick_list, depot=(0, 0)):
    """
    Builds a small COMPLETE graph containing only the depot + pick-list
    locations, with edge weights = shortest walking distance in the
    warehouse. This is the graph QAOA/TSP will actually operate on.
    """
    nodes = [depot] + list(pick_list)

    # All-pairs shortest path distances within the full warehouse graph
    distances = dict(nx.all_pairs_shortest_path_length(warehouse_graph))

    sub_graph = nx.Graph()
    for i, u in enumerate(nodes):
        sub_graph.add_node(i, pos=u)  # store original coordinates for reference
    for i, u in enumerate(nodes):
        for j, v in enumerate(nodes):
            if i < j:
                dist = distances[u][v]
                sub_graph.add_edge(i, j, weight=dist)

    return sub_graph, nodes


def formulate_qubo(warehouse_graph, pick_list, depot=(0, 0)):
    """
    Full pipeline: warehouse graph + pick list -> QUBO (QuadraticProgram).
    Returns: (quadratic_program, tsp_application, node_mapping)
    """
    sub_graph, node_mapping = build_subproblem_graph(warehouse_graph, pick_list, depot)
    tsp = Tsp(sub_graph)
    qp = tsp.to_quadratic_program()
    return qp, tsp, node_mapping


if __name__ == "__main__":
    from app.warehouse.layout import generate_warehouse, generate_pick_list

    # Small test case on purpose - QUBO/QAOA are slow to grow, validate small first
    G = generate_warehouse(rows=5, cols=5)
    pick_list = generate_pick_list(G, num_items=3, seed=1)  # small: 3 items + depot = 4 nodes

    print(f"Pick list: {pick_list}")

    qp, tsp, node_mapping = formulate_qubo(G, pick_list)
    print(f"\nNode mapping (QUBO index -> warehouse coordinate): {list(enumerate(node_mapping))}")
    print(f"\nQUBO problem has {qp.get_num_binary_vars()} binary variables")
    print(f"Number of locations (including depot): {len(node_mapping)}")