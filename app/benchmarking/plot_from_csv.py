"""
Regenerates comparison_chart.png from an already-saved results.csv,
without re-running the (slow) benchmark. Use this after benchmark.py
has already produced results.csv once.
"""
import csv
import os

from app.benchmarking.benchmark import plot_results


def load_results_csv(path=None):
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "results.csv")
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        results = []
        for row in reader:
            results.append({
                "num_items": int(row["num_items"]),
                "optimal_distance": float(row["optimal_distance"]),
                "nn_distance": float(row["nn_distance"]),
                "nn_time_s": float(row["nn_time_s"]),
                "nn_gap_pct": float(row["nn_gap_pct"]),
                "ortools_distance": float(row["ortools_distance"]),
                "ortools_time_s": float(row["ortools_time_s"]),
                "ortools_gap_pct": float(row["ortools_gap_pct"]),
                "qaoa_distance": float(row["qaoa_distance"]),
                "qaoa_time_s": float(row["qaoa_time_s"]),
                "qaoa_gap_pct": float(row["qaoa_gap_pct"]),
            })
        return results


if __name__ == "__main__":
    results = load_results_csv()
    plot_results(results)