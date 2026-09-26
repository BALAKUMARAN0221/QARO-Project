"""
Picker Dashboard - minimal version.

Takes a solved route (from ANY of the four solvers - NN, OR-Tools,
Simulated Annealing, or QAOA) and generates a single, self-contained HTML
page a warehouse picker could open and follow: a visual map of the
warehouse grid with the route drawn on it, plus an ordered stop list.

This is deliberately static (no server, no live updates) - it's the
"Picker Dashboard" block from the architecture diagram, built as a first,
minimal version. A live version (FastAPI, solver selection, one-click
optimize) is the natural next step once this is working.
"""
import os

from app.warehouse.layout import generate_warehouse, generate_pick_list
from app.solvers.qubo_formulation import build_subproblem_graph
from app.solvers.classical_solvers import solve_with_ortools


def _grid_dimensions(warehouse_graph):
    """Infers (rows, cols) from a grid graph's node coordinates."""
    rows = max(r for r, c in warehouse_graph.nodes) + 1
    cols = max(c for r, c in warehouse_graph.nodes) + 1
    return rows, cols


def draw_route_map(warehouse_graph, route, node_mapping, path=None):
    """
    Draws the warehouse grid with the solved route overlaid: every shelf
    location as a small dot, the depot marked distinctly, and the picking
    path drawn as numbered, connected arrows in visiting order.
    Saves a PNG and returns its path.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if path is None:
        path = os.path.join(os.path.dirname(__file__), "route_map.png")

    rows, cols = _grid_dimensions(warehouse_graph)
    fig, ax = plt.subplots(figsize=(6, 6))

    # Every warehouse location as a light background dot
    xs = [c for r, c in warehouse_graph.nodes]
    ys = [r for r, c in warehouse_graph.nodes]
    ax.scatter(xs, ys, s=15, color="lightgray", zorder=1)

    # The route, drawn in visiting order
    route_coords = [node_mapping[i] for i in route]
    route_xs = [c for r, c in route_coords]
    route_ys = [r for r, c in route_coords]
    ax.plot(route_xs, route_ys, "-", color="tab:blue", linewidth=2, zorder=2)

    for step, (r, c) in enumerate(route_coords):
        is_start = (step == 0)
        is_end = (step == len(route_coords) - 1)
        is_depot = is_start or is_end
        ax.scatter(c, r, s=180 if is_depot else 120,
                   color="tab:red" if is_depot else "tab:orange", zorder=3)
        # Start and end depot sit at the identical coordinate - offset their
        # labels in opposite directions so they don't overlap.
        offset = (0, 12) if is_start else (0, -16) if is_end else (0, 8)
        ax.annotate(str(step), (c, r), textcoords="offset points",
                    xytext=offset, fontsize=9, ha="center", weight="bold")

    ax.set_xlim(-1, cols)
    ax.set_ylim(-1, rows)
    ax.set_xticks(range(cols))
    ax.set_yticks(range(rows))
    ax.set_title("Picking Route")
    ax.invert_yaxis()
    ax.grid(True, linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def generate_dashboard_html(route, node_mapping, distance, solver_name, image_path, html_path=None):
    """
    Builds a single self-contained HTML page: the route map image plus an
    ordered stop list, styled simply enough to read clearly on a tablet or
    phone a picker might carry.
    """
    if html_path is None:
        html_path = os.path.join(os.path.dirname(__file__), "picker_dashboard.html")

    route_coords = [node_mapping[i] for i in route]
    stops_html = ""
    for step, (r, c) in enumerate(route_coords):
        label = "Depot (start)" if step == 0 else (
            "Depot (return)" if step == len(route_coords) - 1 else f"Pick item {step}"
        )
        stops_html += f"<li><strong>Stop {step}:</strong> {label} - location ({r}, {c})</li>\n"

    image_filename = os.path.basename(image_path)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>EQROS Picker Dashboard</title>
<style>
  body {{ font-family: Arial, sans-serif; max-width: 700px; margin: 40px auto; padding: 0 20px; }}
  h1 {{ color: #1a1a2e; }}
  .meta {{ background: #f4f4f8; padding: 12px 16px; border-radius: 8px; margin-bottom: 20px; }}
  img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 8px; }}
  ol {{ line-height: 1.8; }}
</style>
</head>
<body>
  <h1>Picking Route</h1>
  <div class="meta">
    <strong>Solver used:</strong> {solver_name}<br>
    <strong>Total distance:</strong> {distance}<br>
    <strong>Number of stops:</strong> {len(route_coords)}
  </div>
  <img src="{image_filename}" alt="Route map">
  <h2>Stop-by-stop sequence</h2>
  <ol>
  {stops_html}
  </ol>
</body>
</html>
"""
    with open(html_path, "w") as f:
        f.write(html)

    return html_path


if __name__ == "__main__":
    # Solve a small example instance with OR-Tools (fast + reliable) and
    # generate the dashboard for it. Swap in any other solver's (route,
    # distance) output - the dashboard doesn't care which solver produced it.
    G = generate_warehouse(rows=5, cols=5)
    pick_list = generate_pick_list(G, num_items=4, seed=1)
    sub_graph, node_mapping = build_subproblem_graph(G, pick_list)
    route, distance = solve_with_ortools(sub_graph)

    image_path = draw_route_map(G, route, node_mapping)
    html_path = generate_dashboard_html(route, node_mapping, distance, "OR-Tools", image_path)

    print(f"Route: {route}")
    print(f"Distance: {distance}")
    print(f"Saved route map to {image_path}")
    print(f"Saved dashboard to {html_path}")
    print(f"\nOpen {html_path} in a browser to view the picker dashboard.")
    