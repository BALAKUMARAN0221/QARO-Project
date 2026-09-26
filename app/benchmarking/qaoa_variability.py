"""
QAOA Run-Variability Testing (extends Module 8).

QAOA is stochastic: because it samples a quantum circuit and relies on a
classical optimizer (COBYLA) to tune parameters, two runs on the EXACT
SAME problem instance can land on different solutions. We already saw
this directly - one run of the 3-item instance found the optimal (14),
another found 16.

This module quantifies that variability: it runs QAOA multiple times on
the same instance and reports best/worst/average/std-dev, plus how often
it actually matches the optimal (its "success rate"). This is compared
against OR-Tools and Nearest Neighbor, which are deterministic and will
give the identical answer every single trial - that determinism itself
is worth reporting as a contrast.
"""
import csv
import os
import statistics
import time

from app.warehouse.layout import generate_warehouse, generate_pick_list
from app.solvers.qubo_formulation import formulate_qubo
from app.solvers.qaoa_solver import solve_with_qaoa
from app.solvers.classical_solvers import solve_nearest_neighbor, solve_with_ortools
from app.benchmarking.benchmark import brute_force_optimal, build_subproblem_graph


def run_qaoa_trials(warehouse_graph, pick_list, num_trials=5, qaoa_reps=1):
    """
    Solves the SAME instance with QAOA `num_trials` times.
    Returns a list of (distance, runtime_seconds) tuples, one per trial.
    """
    qp, tsp, _ = formulate_qubo(warehouse_graph, pick_list)
    trials = []
    for trial_num in range(1, num_trials + 1):
        print(f"    QAOA trial {trial_num}/{num_trials}...")
        start = time.time()
        result = solve_with_qaoa(qp, reps=qaoa_reps)
        elapsed = time.time() - start
        trials.append((result.fval, elapsed))
    return trials


def summarize_trials(distances, optimal_dist):
    """Returns a dict of best/worst/mean/stdev/success_rate for a list of distances."""
    matches_optimal = sum(1 for d in distances if d == optimal_dist)
    return {
        "num_trials": len(distances),
        "optimal_distance": optimal_dist,
        "best_distance": min(distances),
        "worst_distance": max(distances),
        "mean_distance": round(statistics.mean(distances), 2),
        "stdev_distance": round(statistics.stdev(distances), 2) if len(distances) > 1 else 0.0,
        "success_rate_pct": round(100 * matches_optimal / len(distances), 1),
    }


def run_variability_study(pick_list_sizes_and_trials, rows=5, cols=5, seed=1, qaoa_reps=1):
    """
    pick_list_sizes_and_trials: list of (num_items, num_trials) pairs, e.g.
    [(2, 5), (3, 3)] - lets you run more trials on cheap small instances and
    fewer on expensive larger ones.

    Returns a list of result dicts, one per pick-list size, each containing
    the QAOA variability summary plus the deterministic NN/OR-Tools distances
    for direct contrast.
    """
    warehouse_graph = generate_warehouse(rows=rows, cols=cols)
    results = []

    for num_items, num_trials in pick_list_sizes_and_trials:
        pick_list = generate_pick_list(warehouse_graph, num_items=num_items, seed=seed)
        sub_graph, _ = build_subproblem_graph(warehouse_graph, pick_list)
        optimal_dist, _ = brute_force_optimal(sub_graph)

        _, nn_dist = solve_nearest_neighbor(sub_graph)
        _, or_dist = solve_with_ortools(sub_graph)

        print(f"Running {num_trials} QAOA trials for {num_items} pick-list items...")
        trials = run_qaoa_trials(warehouse_graph, pick_list, num_trials=num_trials, qaoa_reps=qaoa_reps)
        distances = [d for d, _ in trials]
        runtimes = [t for _, t in trials]

        summary = summarize_trials(distances, optimal_dist)
        summary["num_items"] = num_items
        summary["nn_distance"] = nn_dist
        summary["ortools_distance"] = or_dist
        summary["mean_qaoa_time_s"] = round(statistics.mean(runtimes), 2)
        results.append(summary)

    return results


def print_variability_table(results):
    header = (
        f"{'Items':<7}{'Trials':<8}{'Optimal':<9}{'NN':<6}{'ORTools':<9}"
        f"{'QAOA best':<11}{'QAOA worst':<12}{'QAOA mean':<11}{'StdDev':<9}{'Success%':<10}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['num_items']:<7}{r['num_trials']:<8}{r['optimal_distance']:<9}"
            f"{r['nn_distance']:<6}{r['ortools_distance']:<9}"
            f"{r['best_distance']:<11}{r['worst_distance']:<12}"
            f"{r['mean_distance']:<11}{r['stdev_distance']:<9}{r['success_rate_pct']:<10}"
        )


def save_variability_csv(results, path=None):
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "qaoa_variability_results.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved variability results to {path}")


def plot_variability(results, path=None):
    """Box-plot style bar chart: best/mean/worst QAOA distance vs. optimal, NN, OR-Tools."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if path is None:
        path = os.path.join(os.path.dirname(__file__), "qaoa_variability_chart.png")

    sizes = [r["num_items"] for r in results]
    x = range(len(sizes))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, [r["optimal_distance"] for r in results], "o-", label="Optimal", color="black")
    ax.plot(x, [r["nn_distance"] for r in results], "s--", label="Nearest Neighbor")
    ax.plot(x, [r["ortools_distance"] for r in results], "^--", label="OR-Tools")
    ax.errorbar(
        x,
        [r["mean_distance"] for r in results],
        yerr=[
            [r["mean_distance"] - r["best_distance"] for r in results],
            [r["worst_distance"] - r["mean_distance"] for r in results],
        ],
        fmt="D-",
        label="QAOA (mean, best-worst range)",
        capsize=5,
    )

    ax.set_xlabel("Number of pick-list items")
    ax.set_ylabel("Route distance")
    ax.set_title("QAOA run-to-run variability vs. deterministic solvers")
    ax.set_xticks(list(x))
    ax.set_xticklabels(sizes)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"Saved variability chart to {path}")


if __name__ == "__main__":
    # Fewer trials at larger sizes since each QAOA run gets much slower.
    # Expect roughly: size 2 (~5s/run x5 = 25s) + size 3 (~220s/run x3 = ~11 min)
    results = run_variability_study(pick_list_sizes_and_trials=[(2, 5), (3, 3)])
    print()
    print_variability_table(results)
    save_variability_csv(results)
    plot_variability(results)