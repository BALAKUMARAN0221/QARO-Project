"""
Event-Driven Order Simulation (the "E" in EQROS).

Models orders arriving unpredictably during a warehouse shift as a Poisson
process - the standard way to represent random arrivals in operations
research / logistics simulation. Each order is a random pick list that
arrives at a random simulated time; as soon as it "arrives," the system
computes its route immediately using a FAST solver.

Why OR-Tools and not QAOA here: our own benchmarking (Module 8 /
qaoa_variability.py) showed QAOA takes 5-220+ seconds per solve and depends
on random circuit sampling. An event-driven system needs to react to each
order quickly and reliably, so OR-Tools (fast, deterministic, near-optimal)
is the right solver for the "live" reactive part of this project. QAOA
remains the batch/offline solver studied separately for solution quality.

Reproducible via `seed`, matching the rest of this project's pattern.
"""
import csv
import os
import random

from app.warehouse.layout import generate_warehouse, generate_pick_list
from app.solvers.qubo_formulation import build_subproblem_graph
from app.solvers.classical_solvers import solve_with_ortools


def generate_order_arrivals(duration, arrival_rate, seed=1):
    """
    Generates order arrival times over a simulated shift using a Poisson
    process: inter-arrival times are drawn from an exponential distribution
    with the given average `arrival_rate` (orders per simulated time unit).

    Returns a sorted list of arrival times (floats) within [0, duration].
    """
    rng = random.Random(seed)
    arrivals = []
    t = 0.0
    while True:
        # Exponential inter-arrival time - the defining property of a
        # Poisson arrival process.
        inter_arrival = rng.expovariate(arrival_rate)
        t += inter_arrival
        if t > duration:
            break
        arrivals.append(round(t, 2))
    return arrivals


def run_event_driven_simulation(duration=8.0, arrival_rate=1.0, rows=5, cols=5,
                                 min_items=2, max_items=4, seed=1):
    """
    Runs the full simulated shift: generates order arrival times, then for
    each arriving order (in time order) generates a random pick list and
    immediately solves it with OR-Tools.

    duration: length of the simulated shift (e.g. 8.0 = an 8-hour shift,
    in simulated time units).
    arrival_rate: average number of orders arriving per time unit.

    Returns a list of event dicts, one per order, in arrival order.
    """
    rng = random.Random(seed)
    warehouse_graph = generate_warehouse(rows=rows, cols=cols)
    arrival_times = generate_order_arrivals(duration, arrival_rate, seed=seed)

    events = []
    for order_id, arrival_time in enumerate(arrival_times, start=1):
        num_items = rng.randint(min_items, max_items)
        # Each order needs its own pick list - vary the seed per order so
        # they aren't all identical, while the whole run stays reproducible.
        pick_list = generate_pick_list(warehouse_graph, num_items=num_items, seed=seed * 1000 + order_id)

        sub_graph, node_mapping = build_subproblem_graph(warehouse_graph, pick_list)
        route, distance = solve_with_ortools(sub_graph)

        print(f"[t={arrival_time:>6.2f}] Order {order_id}: {num_items} items -> route distance {distance}")

        events.append({
            "order_id": order_id,
            "arrival_time": arrival_time,
            "num_items": num_items,
            "pick_list": pick_list,
            "route": route,
            "distance": distance,
        })

    return events


def save_events_csv(events, path=None):
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "event_log.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["order_id", "arrival_time", "num_items", "pick_list", "route", "distance"])
        writer.writeheader()
        for e in events:
            writer.writerow(e)
    print(f"\nSaved event log to {path}")


def plot_event_timeline(events, path=None):
    """Saves a timeline chart: order arrivals over simulated time, sized by route distance."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if path is None:
        path = os.path.join(os.path.dirname(__file__), "event_timeline_chart.png")

    times = [e["arrival_time"] for e in events]
    distances = [e["distance"] for e in events]
    order_ids = [e["order_id"] for e in events]

    fig, ax = plt.subplots(figsize=(10, 5))
    scatter = ax.scatter(times, distances, s=80, c=order_ids, cmap="viridis")
    for e in events:
        ax.annotate(str(e["order_id"]), (e["arrival_time"], e["distance"]),
                    textcoords="offset points", xytext=(0, 6), fontsize=8, ha="center")

    ax.set_xlabel("Simulated shift time")
    ax.set_ylabel("Route distance")
    ax.set_title("EQROS: Order arrivals during a simulated shift (OR-Tools routing)")
    fig.colorbar(scatter, ax=ax, label="Order ID (arrival order)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"Saved event timeline chart to {path}")


if __name__ == "__main__":
    events = run_event_driven_simulation(duration=8.0, arrival_rate=1.0, seed=1)
    print(f"\nTotal orders during shift: {len(events)}")
    save_events_csv(events)
    plot_event_timeline(events)