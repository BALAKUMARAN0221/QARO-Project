"""
Warehouse layout + pick-list generator.
This IS your "dataset" - fully synthetic, reproducible, no download needed.
See project report Section 9 (Dataset) for justification.
"""
import networkx as nx
import random

def generate_warehouse(rows=5, cols=5, seed=42):
    """
    Creates a grid-based warehouse graph.
    Each node = a shelf/pick location. Node (0,0) is treated as the depot.
    Edge weight = walking distance (1 step between adjacent grid cells).
    """
    random.seed(seed)
    G = nx.grid_2d_graph(rows, cols)
    for u, v in G.edges():
        G.edges[u, v]['weight'] = 1
    return G

def generate_pick_list(G, num_items=6, seed=1, depot=(0, 0)):
    """Randomly selects `num_items` locations (excluding depot) for an order."""
    random.seed(seed)
    nodes = [n for n in G.nodes if n != depot]
    return random.sample(nodes, num_items)

if __name__ == "__main__":
    G = generate_warehouse(rows=5, cols=5)
    pick_list = generate_pick_list(G, num_items=6)
    print(f"Warehouse: {G.number_of_nodes()} locations, {G.number_of_edges()} walkable edges")
    print(f"Pick list for this order: {pick_list}")