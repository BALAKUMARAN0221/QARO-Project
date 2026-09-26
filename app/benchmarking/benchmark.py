"""
Benchmarking (Module 8).

Runs QAOA, Nearest Neighbor, and OR-Tools on the SAME warehouse instances
across several pick-list sizes, timing each and comparing their route
distance against the brute-force optimal. Produces:
  - a printed comparison table
  - results.csv (raw numbers, for the report's appendix/tables)
  - comparison_chart.png (grouped bar chart of distance vs. pick-list size)

This is the module that actually answers the project's research question:
does QAOA compete with classical baselines on this routing problem?
"""
import csv
import itertools
import os
import time

from app.warehouse.layout import generate_warehouse, generate_pick_list
from app.solvers.qubo_formulation import build_subproblem_graph, formulate_qubo
from app.solvers.qaoa_solver import solve_with_qaoa
from app.solvers.classical_solvers import solve_nearest_neighbor, solve_with_ortools


def brute_force_optimal(sub_graph):
    """Ground truth for small instances, same helper used in the other solver modules."""
    n = sub_graph.number_of_nodes()
    best = None
    for perm in itertools.permutations(range(1, n)):
        route = [0] + list(perm) + [0]
        dist = sum(sub_graph[route[i]][route[i + 1]]['weight'] for i in range(len(route) - 1))
        if best is None or dist < best[0]:
            best = (dist, route)
    return best


def run_single_instance(warehouse_graph, pick_list, qaoa_reps=1):
    """
    Runs NN, OR-Tools, and QAOA on one warehouse + pick-list instance.
    Returns a dict of distances, runtimes, and optimality gaps (%),
    all measured against the same brute-force optimal for this instance.
    """
    sub_graph, _ = build_subproblem_graph(warehouse_graph, pick_list)
    optimal_dist, _ = brute_force_optimal(sub_graph)

    start = time.time()
    _, nn_dist = solve_nearest_neighbor(sub_graph)
    nn_time = time.time() - start

    start = time.time()
    _, or_dist = solve_with_ortools(sub_graph)
    or_time = time.time() - start

    qp, tsp, _ = formulate_qubo(warehouse_graph, pick_list)
    start = time.time()
    result = solve_with_qaoa(qp, reps=qaoa_reps)
    qaoa_time = time.time() - start
    qaoa_dist = result.fval

    def gap_pct(dist):
        if optimal_dist == 0:
            return 0.0
        return round(100 * (dist - optimal_dist) / optimal_dist, 1)

    return {
        "num_items": len(pick_list),
        "optimal_distance": optimal_dist,
        "nn_distance": nn_dist,
        "nn_time_s": round(nn_time, 4),
        "nn_gap_pct": gap_pct(nn_dist),
        "ortools_distance": or_dist,
        "ortools_time_s": round(or_time, 4),
        "ortools_gap_pct": gap_pct(or_dist),
        "qaoa_distance": qaoa_dist,
        "qaoa_time_s": round(qaoa_time, 4),
        "qaoa_gap_pct": gap_pct(qaoa_dist),
    }


def run_benchmark(pick_list_sizes=(2, 3, 4), rows=5, cols=5, seed=1, qaoa_reps=1):
    """
    Runs run_single_instance for each pick-list size and returns a list of
    result dicts, one per size. Sizes are kept small by default (QAOA/QUBO
    grow expensive fast - each extra pick-list item roughly doubles the
    qubit count).
    """
    warehouse_graph = generate_warehouse(rows=rows, cols=cols)
    results = []
    for size in pick_list_sizes:
        pick_list = generate_pick_list(warehouse_graph, num_items=size, seed=seed)
        print(f"Running benchmark for {size} pick-list items...")
        results.append(run_single_instance(warehouse_graph, pick_list, qaoa_reps=qaoa_reps))
    return results


def print_results_table(results):
    header = f"{'Items':<7}{'Optimal':<9}{'NN':<8}{'NN gap%':<9}{'ORTools':<9}{'OR gap%':<9}{'QAOA':<8}{'QAOA gap%':<10}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['num_items']:<7}{r['optimal_distance']:<9}"
            f"{r['nn_distance']:<8}{r['nn_gap_pct']:<9}"
            f"{r['ortools_distance']:<9}{r['ortools_gap_pct']:<9}"
            f"{r['qaoa_distance']:<8}{r['qaoa_gap_pct']:<10}"
        )


def save_results_csv(results, path=None):
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "results.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved raw results to {path}")


def plot_results(results, path=None):
    """Saves a grouped bar chart (distance per solver, by pick-list size)."""
    import matplotlib
    matplotlib.use("Agg")  # no display needed, just save the file
    import matplotlib.pyplot as plt

    if path is None:
        path = os.path.join(os.path.dirname(__file__), "comparison_chart.png")

    sizes = [r["num_items"] for r in results]
    nn = [r["nn_distance"] for r in results]
    ortools = [r["ortools_distance"] for r in results]
    qaoa = [r["qaoa_distance"] for r in results]
    optimal = [r["optimal_distance"] for r in results]

    x = range(len(sizes))
    width = 0.2

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([i - 1.5 * width for i in x], optimal, width, label="Optimal (brute-force)")
    ax.bar([i - 0.5 * width for i in x], nn, width, label="Nearest Neighbor")
    ax.bar([i + 0.5 * width for i in x], ortools, width, label="OR-Tools")
    ax.bar([i + 1.5 * width for i in x], qaoa, width, label="QAOA")

    ax.set_xlabel("Number of pick-list items")
    ax.set_ylabel("Route distance")
    ax.set_title("EQROS: Solver comparison across pick-list sizes")
    ax.set_xticks(list(x))
    ax.set_xticklabels(sizes)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"Saved comparison chart to {path}")


if __name__ == "__main__":
    # NOTE ON RUNTIME: QAOA's QUBO grows fast - going from 2 to 3 pick items
    # (9 to 16 binary variables) took QAOA from ~5s to ~220s in testing, since
    # the underlying statevector simulation scales with 2^(num_vars). This
    # slowdown IS the finding worth reporting (quantum simulation overhead on
    # small problems) - so (1, 2, 3) is kept as the default despite the ~4
    # minute total runtime. Add 4 only if you have time to spare (it can take
    # 10+ minutes) or want to push the scaling curve further for the report.
    results = run_benchmark(pick_list_sizes=(1, 2, 3), qaoa_reps=1)
    print()
    print_results_table(results)
    save_results_csv(results)
    plot_results(results)