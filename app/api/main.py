"""
Live Dashboard - full advanced version.
 
Combines four features on top of the original live dashboard:
 
1. Solver Selection: an "Auto" mode that picks a solver by a fixed rule,
   based on what our own benchmarking (Module 8) and variability study
   found - QAOA is only fast enough to be practical at very small sizes,
   Simulated Annealing is fast and solves the same QUBO, OR-Tools is the
   most reliable choice as size grows:
       num_items <= 2   -> QAOA
       num_items 3-4    -> Simulated Annealing
       num_items >= 5   -> OR-Tools
 
2. Compare-all: solves ONE order with all four solvers and returns every
   result, so they can be shown side by side.
 
3. Event-driven live feed: a WebSocket endpoint that generates orders on
   their own random schedule (a Poisson process, same idea as
   app/events/event_driven_simulation.py) and auto-solves each one using
   the Solver Selection rule above, pushing results to the browser as
   they're produced - no button click needed once it's started.
 
4. History log: every solve (manual, compare, or live) is appended to an
   in-memory list and returned so the page can show a running table.
 
Run with:
    uvicorn app.api.main:app --reload
Then open http://127.0.0.1:8000 in a browser.
"""
import asyncio
import random
from datetime import datetime
 
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
 
from app.warehouse.layout import generate_warehouse, generate_pick_list
from app.solvers.qubo_formulation import build_subproblem_graph, formulate_qubo
from app.solvers.classical_solvers import (
    solve_nearest_neighbor,
    solve_with_ortools,
    solve_with_simulated_annealing,
)
from app.solvers.qaoa_solver import solve_with_qaoa
 
app = FastAPI(title="EQROS Live Dashboard")
 
WAREHOUSE_GRAPH = generate_warehouse(rows=5, cols=5)
 
# QAOA's simulation cost grows exponentially with problem size - variables
# scale as (num_items+1)^2, and each extra item has been ~40x slower than
# the last in our own testing (2 items/9 vars: ~5s, 3 items/16 vars: ~220s).
# 4 items (25 vars) was tested and did not finish in a reasonable time, so
# the safe, VERIFIED limit for this demo is 3 items.
MAX_QAOA_ITEMS = 3
 
# In-memory session state - resets whenever the server restarts, which is
# fine for a demo/dev tool like this.
ORDER_HISTORY = []
_next_order_id = 1
 
 
class SolveRequest(BaseModel):
    solver: str          # "nearest_neighbor" | "ortools" | "simulated_annealing" | "qaoa" | "auto"
    num_items: int = 3
    seed: int = 1
 
 
class CompareRequest(BaseModel):
    num_items: int = 3
    seed: int = 1
 
 
def select_best_solver(num_items):
    """The Solver Selection rule (see module docstring for the reasoning)."""
    if num_items <= 2:
        return "qaoa"
    elif num_items <= 4:
        return "simulated_annealing"
    else:
        return "ortools"
 
 
def _rotate_route_to_start_at_depot(route):
    """See earlier version's docstring - QAOA/SA don't always decode a
    route starting at the depot (node 0); rotate so they display correctly."""
    if route[0] == 0:
        return route
    core = route[:-1]
    idx = core.index(0)
    rotated = core[idx:] + core[:idx]
    rotated.append(rotated[0])
    return rotated
 
 
def _run_one_solver(solver_name, pick_list):
    """Runs a single named solver on pick_list. Returns (route, distance, node_mapping)."""
    sub_graph, node_mapping = build_subproblem_graph(WAREHOUSE_GRAPH, pick_list)
 
    if solver_name == "nearest_neighbor":
        route, distance = solve_nearest_neighbor(sub_graph)
 
    elif solver_name == "ortools":
        route, distance = solve_with_ortools(sub_graph)
 
    elif solver_name == "simulated_annealing":
        qp, tsp, _ = formulate_qubo(WAREHOUSE_GRAPH, pick_list)
        route, distance = solve_with_simulated_annealing(qp, tsp, sub_graph, num_reads=200, seed=1)
 
    elif solver_name == "qaoa":
        if len(pick_list) > MAX_QAOA_ITEMS:
            raise ValueError(
                f"QAOA is limited to {MAX_QAOA_ITEMS} pick-list items in this demo. "
                f"Simulation cost grows exponentially with problem size (our own testing "
                f"measured 16 variables taking ~220s) - {len(pick_list)} items would be "
                f"impractically slow or exhaust memory. Try {MAX_QAOA_ITEMS} or fewer items, "
                f"or pick a different solver."
            )
        qp, tsp, _ = formulate_qubo(WAREHOUSE_GRAPH, pick_list)
        result = solve_with_qaoa(qp, reps=1)
        base_route = tsp.interpret(result.x)
        route = base_route + [base_route[0]]
        distance = result.fval
 
    else:
        raise ValueError(f"Unknown solver: {solver_name}")
 
    route = _rotate_route_to_start_at_depot(route)
    return route, distance, node_mapping
 
 
def _log_history(solver, distance, num_items, mode):
    global _next_order_id
    entry = {
        "order_id": _next_order_id,
        "time": datetime.now().strftime("%H:%M:%S"),
        "solver": solver,
        "distance": distance,
        "num_items": num_items,
        "mode": mode,
    }
    ORDER_HISTORY.append(entry)
    _next_order_id += 1
    return entry
 
 
@app.post("/api/solve")
def solve(req: SolveRequest):
    """Manual solve: one order, one solver (or 'auto' to apply the Solver Selection rule)."""
    pick_list = generate_pick_list(WAREHOUSE_GRAPH, num_items=req.num_items, seed=req.seed)
 
    solver_name = select_best_solver(req.num_items) if req.solver == "auto" else req.solver
 
    try:
        route, distance, node_mapping = _run_one_solver(solver_name, pick_list)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
 
    _log_history(solver_name, distance, req.num_items, mode="manual" if req.solver != "auto" else "auto")
 
    node_mapping_json = {str(i): list(coord) for i, coord in enumerate(node_mapping)}
    return JSONResponse({
        "solver": solver_name,
        "auto_selected": req.solver == "auto",
        "pick_list": pick_list,
        "route": route,
        "distance": distance,
        "node_mapping": node_mapping_json,
        "history": ORDER_HISTORY[-20:],
    })
 
 
@app.post("/api/compare")
def compare(req: CompareRequest):
    """
    Solves the SAME order with all four solvers, returns every result.
    If one solver fails (e.g. QAOA refused for being too large), its entry
    contains an "error" field instead of a route/distance, and the other
    three still complete normally.
    """
    pick_list = generate_pick_list(WAREHOUSE_GRAPH, num_items=req.num_items, seed=req.seed)
 
    results = {}
    node_mapping_json = None
    for solver_name in ["nearest_neighbor", "ortools", "simulated_annealing", "qaoa"]:
        try:
            route, distance, node_mapping = _run_one_solver(solver_name, pick_list)
        except ValueError as e:
            results[solver_name] = {"error": str(e)}
            continue
 
        if node_mapping_json is None:
            node_mapping_json = {str(i): list(coord) for i, coord in enumerate(node_mapping)}
        results[solver_name] = {"route": route, "distance": distance}
        _log_history(solver_name, distance, req.num_items, mode="compare")
 
    return JSONResponse({
        "pick_list": pick_list,
        "num_items": req.num_items,
        "results": results,
        "node_mapping": node_mapping_json,
        "history": ORDER_HISTORY[-20:],
    })
 
 
@app.get("/api/history")
def get_history():
    return JSONResponse({"history": ORDER_HISTORY[-20:]})
 
 
@app.websocket("/ws/live")
async def live_feed(websocket: WebSocket):
    """
    Event-driven live feed: generates orders on a Poisson process and
    auto-solves each with the Solver Selection rule, pushing every result
    to the browser as soon as it's ready. Runs until the client disconnects
    (i.e. clicks "Stop Live Feed" or closes the page).
    """
    await websocket.accept()
    rng = random.Random()
    arrival_rate = 0.2  # avg orders per second - tuned for a readable demo pace
    loop = asyncio.get_event_loop()
 
    try:
        while True:
            wait_time = rng.expovariate(arrival_rate)
            await asyncio.sleep(min(wait_time, 8.0))  # cap the wait so demos don't stall too long
 
            num_items = rng.randint(2, 5)
            seed = rng.randint(1, 1_000_000)
            pick_list = generate_pick_list(WAREHOUSE_GRAPH, num_items=num_items, seed=seed)
            solver_name = select_best_solver(num_items)
 
            # Run the (blocking, CPU-bound) solve in a worker thread so it
            # doesn't freeze the event loop for other connections.
            route, distance, node_mapping = await loop.run_in_executor(
                None, _run_one_solver, solver_name, pick_list
            )
 
            entry = _log_history(solver_name, distance, num_items, mode="live")
            node_mapping_json = {str(i): list(coord) for i, coord in enumerate(node_mapping)}
 
            await websocket.send_json({
                "order_id": entry["order_id"],
                "time": entry["time"],
                "solver": solver_name,
                "pick_list": pick_list,
                "route": route,
                "distance": distance,
                "node_mapping": node_mapping_json,
                "history": ORDER_HISTORY[-20:],
            })
    except WebSocketDisconnect:
        pass
 
 
@app.get("/", response_class=HTMLResponse)
def index():
    return _PAGE_HTML
 
 
_PAGE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>EQROS Route Console</title>
<link rel="icon" href="data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22><text y=%22.9em%22 font-size=%2290%22>%F0%9F%9A%9A</text></svg>">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #0F1420;
    --panel: #171E2E;
    --panel-alt: #1B2438;
    --border: #2A3550;
    --text: #E7ECF5;
    --text-muted: #8992A9;
    --amber: #FFB020;
    --amber-dim: #A87418;
    --blue: #3FA9F5;
    --red: #FF5C4D;
    --green: #34D399;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    background: var(--bg);
    color: var(--text);
    font-family: 'IBM Plex Sans', Arial, sans-serif;
  }
  .mono { font-family: 'IBM Plex Mono', 'Courier New', monospace; }
 
  header {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 18px 28px;
    border-bottom: 1px solid var(--border);
  }
  .live-dot {
    width: 9px; height: 9px; border-radius: 50%;
    background: var(--text-muted);
  }
  .live-dot.on {
    background: var(--amber);
    animation: pulse 2s infinite;
  }
  @keyframes pulse {
    0%   { box-shadow: 0 0 0 0 rgba(255,176,32,0.55); }
    70%  { box-shadow: 0 0 0 8px rgba(255,176,32,0); }
    100% { box-shadow: 0 0 0 0 rgba(255,176,32,0); }
  }
  header h1 { font-size: 17px; font-weight: 600; margin: 0; letter-spacing: 0.2px; }
  header .tag { font-size: 11px; color: var(--text-muted); margin-left: 4px; }
  header .status { margin-left: auto; font-size: 11px; color: var(--text-muted); }
  header .status.live { color: var(--amber); }
 
  main { max-width: 1200px; margin: 0 auto; padding: 24px 28px 60px; }
 
  .control-strip {
    display: flex;
    flex-wrap: wrap;
    align-items: end;
    gap: 18px;
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 16px 18px;
    margin-bottom: 22px;
  }
  .field label { display: block; font-size: 11px; color: var(--text-muted); margin-bottom: 6px; }
  select, input[type=number] {
    background: var(--panel-alt);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 8px 10px;
    font-size: 14px;
    font-family: 'IBM Plex Mono', monospace;
  }
  select:focus, input:focus { outline: none; border-color: var(--amber); }
  input[type=number] { width: 70px; }
 
  button {
    border: none;
    border-radius: 4px;
    padding: 10px 18px;
    font-family: 'IBM Plex Sans', sans-serif;
    font-weight: 600;
    font-size: 13px;
    cursor: pointer;
    transition: transform 0.08s ease, background 0.15s ease;
  }
  button:active { transform: translateY(1px); }
  button:disabled { opacity: 0.45; cursor: not-allowed; }
 
  button.primary { background: var(--amber); color: #1A1204; }
  button.primary:hover:not(:disabled) { background: #FFC24D; }
 
  button.secondary { background: var(--panel-alt); color: var(--text); border: 1px solid var(--border); }
  button.secondary:hover:not(:disabled) { border-color: var(--blue); }
 
  button.live-toggle { background: var(--panel-alt); color: var(--text); border: 1px solid var(--border); }
  button.live-toggle.active { background: var(--red); color: #2b0d09; border-color: var(--red); }
 
  .qaoa-warning { font-size: 12px; color: var(--amber); align-self: center; }
 
  .grid { display: grid; grid-template-columns: 1.3fr 0.9fr 0.9fr; gap: 18px; align-items: start; }
  @media (max-width: 980px) { .grid { grid-template-columns: 1fr; } }
 
  .panel { background: var(--panel); border: 1px solid var(--border); border-radius: 6px; padding: 16px; }
  .panel h2 {
    font-size: 12px; text-transform: uppercase; letter-spacing: 0.6px;
    color: var(--text-muted); margin: 0 0 12px; font-weight: 500;
  }
 
  #routeSvg { width: 100%; height: auto; display: block; }
 
  .stat-row {
    display: flex; justify-content: space-between; padding: 8px 0;
    border-bottom: 1px solid var(--border); font-size: 13px;
  }
  .stat-row:last-child { border-bottom: none; }
  .stat-row .val { font-family: 'IBM Plex Mono', monospace; color: var(--amber); font-weight: 600; }
 
  ol#stopList { list-style: none; margin: 12px 0 0; padding: 0; font-size: 13px; }
  ol#stopList li {
    display: flex; gap: 10px; padding: 7px 0;
    border-bottom: 1px dashed var(--border); color: var(--text);
  }
  ol#stopList li:last-child { border-bottom: none; }
  ol#stopList .step-num { font-family: 'IBM Plex Mono', monospace; color: var(--blue); width: 18px; flex-shrink: 0; }
  ol#stopList .coord { color: var(--text-muted); font-family: 'IBM Plex Mono', monospace; margin-left: auto; }
 
  .empty-state { color: var(--text-muted); font-size: 13px; padding: 20px 0; text-align: center; }
 
  .compare-bar-row { margin-bottom: 10px; }
  .compare-bar-label {
    display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 4px;
  }
  .compare-bar-track { background: var(--panel-alt); border-radius: 3px; height: 8px; overflow: hidden; }
  .compare-bar-fill { height: 100%; background: var(--blue); border-radius: 3px; }
  .compare-bar-fill.best { background: var(--green); }
 
  table.history { width: 100%; border-collapse: collapse; font-size: 12px; }
  table.history th {
    text-align: left; color: var(--text-muted); font-weight: 500;
    padding: 6px 8px; border-bottom: 1px solid var(--border); font-size: 11px; text-transform: uppercase;
  }
  table.history td { padding: 6px 8px; border-bottom: 1px solid var(--border); font-family: 'IBM Plex Mono', monospace; }
  table.history tr:last-child td { border-bottom: none; }
  .mode-tag {
    font-family: 'IBM Plex Sans', sans-serif; font-size: 10px; padding: 2px 6px; border-radius: 3px;
  }
  .mode-tag.manual { background: #2A3550; color: var(--text-muted); }
  .mode-tag.auto { background: #3a2f13; color: var(--amber); }
  .mode-tag.live { background: #2f1313; color: var(--red); }
  .mode-tag.compare { background: #13302a; color: var(--green); }
 
  .history-scroll { max-height: 340px; overflow-y: auto; }
</style>
</head>
<body>
 
<header>
  <span class="live-dot" id="liveDot"></span>
  <h1>EQROS Route Console</h1>
  <span class="tag mono">warehouse routing / solver comparison</span>
  <span class="status" id="liveStatus">IDLE</span>
</header>
 
<main>
  <div class="control-strip">
    <div class="field">
      <label>Solver</label>
      <select id="solver">
        <option value="auto">Auto (rule-based)</option>
        <option value="ortools">OR-Tools</option>
        <option value="nearest_neighbor">Nearest Neighbor</option>
        <option value="simulated_annealing">Simulated Annealing</option>
        <option value="qaoa">QAOA (quantum)</option>
      </select>
    </div>
    <div class="field">
      <label>Pick-list size</label>
      <input type="number" id="numItems" value="3" min="2" max="6">
    </div>
    <button class="primary" id="solveBtn" onclick="solveOrder()">Generate &amp; Solve</button>
    <button class="secondary" id="compareBtn" onclick="compareAll()">Compare All Solvers</button>
    <button class="live-toggle" id="liveBtn" onclick="toggleLiveFeed()">Start Live Feed</button>
    <div class="qaoa-warning mono" id="qaoaWarning" style="display:none;">may take several minutes</div>
  </div>
 
  <div class="grid">
    <div class="panel">
      <h2>Route map</h2>
      <svg id="routeSvg" viewBox="-40 -40 540 540"></svg>
    </div>
 
    <div class="panel">
      <h2>Run summary</h2>
      <div id="summaryEmpty" class="empty-state">Run a solve to see results here.</div>
      <div id="summaryBox" style="display:none;">
        <div class="stat-row"><span>Solver</span><span class="val mono" id="solverUsed"></span></div>
        <div class="stat-row"><span>Total distance</span><span class="val mono" id="distance"></span></div>
        <div class="stat-row"><span>Stops</span><span class="val mono" id="numStops"></span></div>
      </div>
      <div id="compareBox" style="display:none; margin-top: 14px;"></div>
      <ol id="stopList"></ol>
    </div>
 
    <div class="panel">
      <h2>Order history</h2>
      <div id="historyEmpty" class="empty-state">No orders solved yet.</div>
      <div class="history-scroll">
        <table class="history" id="historyTable" style="display:none;">
          <thead><tr><th>#</th><th>Time</th><th>Solver</th><th>Items</th><th>Dist</th><th>Mode</th></tr></thead>
          <tbody id="historyBody"></tbody>
        </table>
      </div>
    </div>
  </div>
</main>
 
<script>
let liveSocket = null;
 
document.getElementById('solver').addEventListener('change', function() {
  const isQaoa = this.value === 'qaoa';
  document.getElementById('qaoaWarning').style.display = isQaoa ? 'block' : 'none';
  const numItems = document.getElementById('numItems');
  if (isQaoa) {
    numItems.dataset.prevMax = numItems.max;
    numItems.max = 3;
    if (parseInt(numItems.value) > 3) numItems.value = 3;
  } else if (numItems.dataset.prevMax) {
    numItems.max = numItems.dataset.prevMax;
  }
});
 
function setManualControlsEnabled(enabled) {
  document.getElementById('solveBtn').disabled = !enabled;
  document.getElementById('compareBtn').disabled = !enabled;
  document.getElementById('solver').disabled = !enabled;
  document.getElementById('numItems').disabled = !enabled;
}
 
async function solveOrder() {
  const btn = document.getElementById('solveBtn');
  const solver = document.getElementById('solver').value;
  const numItems = parseInt(document.getElementById('numItems').value);
 
  btn.disabled = true;
  btn.textContent = 'Solving…';
  document.getElementById('compareBox').style.display = 'none';
 
  try {
    const resp = await fetch('/api/solve', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({solver: solver, num_items: numItems, seed: Math.floor(Math.random() * 100000)})
    });
    const data = await resp.json();
    if (!resp.ok) {
      alert(data.error || 'Solve failed - please try again.');
      return;
    }
    renderRoute(data.route, data.node_mapping, data.solver, data.distance);
    renderHistory(data.history);
  } catch (err) {
    alert('Solve failed: could not reach the server.');
  } finally {
    btn.disabled = false;
    btn.textContent = 'Generate & Solve';
  }
}
 
async function compareAll() {
  const btn = document.getElementById('compareBtn');
  const numItems = parseInt(document.getElementById('numItems').value);
 
  btn.disabled = true;
  btn.textContent = 'Comparing…';
 
  try {
    const resp = await fetch('/api/compare', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({num_items: numItems, seed: Math.floor(Math.random() * 100000)})
    });
    const data = await resp.json();
    if (!resp.ok) {
      alert(data.error || 'Compare failed - please try again.');
      return;
    }
    renderCompare(data);
    renderHistory(data.history);
  } catch (err) {
    alert('Compare failed: could not reach the server.');
  } finally {
    btn.disabled = false;
    btn.textContent = 'Compare All Solvers';
  }
}
 
function renderCompare(data) {
  const names = {nearest_neighbor: 'Nearest Neighbor', ortools: 'OR-Tools',
                 simulated_annealing: 'Simulated Annealing', qaoa: 'QAOA'};
 
  const validEntries = Object.entries(data.results).filter(([k, r]) => !r.error);
  const distances = validEntries.map(([k, r]) => r.distance);
  const best = Math.min(...distances);
  const maxD = Math.max(...distances);
 
  let html = '';
  let bestSolver = null;
  for (const [key, res] of Object.entries(data.results)) {
    if (res.error) {
      html += `<div class="compare-bar-row">
        <div class="compare-bar-label"><span>${names[key]}</span><span class="mono" style="color: var(--text-muted);">skipped</span></div>
      </div>`;
      continue;
    }
    const pct = maxD > 0 ? Math.round((res.distance / maxD) * 100) : 0;
    const isBest = res.distance === best;
    if (isBest && !bestSolver) bestSolver = key;
    html += `<div class="compare-bar-row">
      <div class="compare-bar-label"><span>${names[key]}</span><span class="mono">${res.distance}</span></div>
      <div class="compare-bar-track"><div class="compare-bar-fill ${isBest ? 'best' : ''}" style="width:${pct}%"></div></div>
    </div>`;
  }
  document.getElementById('compareBox').innerHTML = html;
  document.getElementById('compareBox').style.display = 'block';
 
  if (bestSolver) {
    const bestResult = data.results[bestSolver];
    renderRoute(bestResult.route, data.node_mapping, bestSolver + ' (best of comparison)', bestResult.distance);
  }
}
 
function toggleLiveFeed() {
  const btn = document.getElementById('liveBtn');
  if (liveSocket) {
    liveSocket.close();
    return;
  }
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  liveSocket = new WebSocket(`${protocol}//${window.location.host}/ws/live`);
 
  liveSocket.onopen = () => {
    btn.textContent = 'Stop Live Feed';
    btn.classList.add('active');
    document.getElementById('liveDot').classList.add('on');
    document.getElementById('liveStatus').textContent = 'LIVE';
    document.getElementById('liveStatus').classList.add('live');
    setManualControlsEnabled(false);
    document.getElementById('compareBox').style.display = 'none';
  };
 
  liveSocket.onmessage = (event) => {
    const data = JSON.parse(event.data);
    renderRoute(data.route, data.node_mapping, data.solver + ' (auto)', data.distance);
    renderHistory(data.history);
  };
 
  liveSocket.onclose = () => {
    btn.textContent = 'Start Live Feed';
    btn.classList.remove('active');
    document.getElementById('liveDot').classList.remove('on');
    document.getElementById('liveStatus').textContent = 'IDLE';
    document.getElementById('liveStatus').classList.remove('live');
    setManualControlsEnabled(true);
    liveSocket = null;
  };
}
 
function renderRoute(route, nodeMapping, solverLabel, distance) {
  document.getElementById('summaryEmpty').style.display = 'none';
  document.getElementById('summaryBox').style.display = 'block';
  document.getElementById('solverUsed').textContent = solverLabel;
  document.getElementById('distance').textContent = distance;
  document.getElementById('numStops').textContent = route.length;
 
  const coords = route.map(i => nodeMapping[i]);
  const scale = 100;
  const svg = document.getElementById('routeSvg');
  svg.innerHTML = '';
 
  for (let r = 0; r < 5; r++) {
    for (let c = 0; c < 5; c++) {
      const dot = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
      dot.setAttribute('cx', c * scale);
      dot.setAttribute('cy', r * scale);
      dot.setAttribute('r', 2.5);
      dot.setAttribute('fill', '#2A3550');
      svg.appendChild(dot);
    }
  }
 
  const pathPoints = coords.map(([r, c]) => `${c * scale},${r * scale}`).join(' ');
  const polyline = document.createElementNS('http://www.w3.org/2000/svg', 'polyline');
  polyline.setAttribute('points', pathPoints);
  polyline.setAttribute('fill', 'none');
  polyline.setAttribute('stroke', '#FFB020');
  polyline.setAttribute('stroke-width', '3');
  polyline.setAttribute('stroke-linejoin', 'round');
  svg.appendChild(polyline);
 
  const len = polyline.getTotalLength();
  polyline.style.strokeDasharray = len;
  polyline.style.strokeDashoffset = len;
  polyline.getBoundingClientRect();
  polyline.style.transition = 'stroke-dashoffset 0.9s ease';
  requestAnimationFrame(() => { polyline.style.strokeDashoffset = 0; });
 
  coords.forEach(([r, c], step) => {
    const isStart = step === 0;
    const isEnd = step === coords.length - 1;
    const isDepot = isStart || isEnd;
    const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    circle.setAttribute('cx', c * scale);
    circle.setAttribute('cy', r * scale);
    circle.setAttribute('r', isDepot ? 10 : 8);
    circle.setAttribute('fill', isDepot ? '#FF5C4D' : '#3FA9F5');
    circle.style.opacity = 0;
    circle.style.transition = `opacity 0.3s ease ${step * 60}ms`;
    svg.appendChild(circle);
    requestAnimationFrame(() => { circle.style.opacity = 1; });
 
    // Start and end depot share a coordinate - offset their labels apart.
    const yOffset = isStart ? -18 : isEnd ? 20 : -15;
    const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    label.setAttribute('x', c * scale);
    label.setAttribute('y', r * scale + yOffset);
    label.setAttribute('text-anchor', 'middle');
    label.setAttribute('font-size', '12');
    label.setAttribute('font-family', 'IBM Plex Mono, monospace');
    label.setAttribute('font-weight', '600');
    label.setAttribute('fill', '#E7ECF5');
    label.textContent = step;
    label.style.opacity = 0;
    label.style.transition = `opacity 0.3s ease ${step * 60}ms`;
    svg.appendChild(label);
    requestAnimationFrame(() => { label.style.opacity = 1; });
  });
 
  const list = document.getElementById('stopList');
  list.innerHTML = '';
  coords.forEach(([r, c], step) => {
    const li = document.createElement('li');
    let label = `Pick item ${step}`;
    if (step === 0) label = 'Depot (start)';
    if (step === coords.length - 1) label = 'Depot (return)';
    li.innerHTML = `<span class="step-num">${String(step).padStart(2,'0')}</span><span>${label}</span><span class="coord">(${r}, ${c})</span>`;
    list.appendChild(li);
  });
}
 
function renderHistory(history) {
  if (!history || history.length === 0) return;
  document.getElementById('historyEmpty').style.display = 'none';
  document.getElementById('historyTable').style.display = 'table';
 
  const body = document.getElementById('historyBody');
  body.innerHTML = '';
  [...history].reverse().forEach(h => {
    const tr = document.createElement('tr');
    tr.innerHTML = `<td>${h.order_id}</td><td>${h.time}</td><td>${h.solver}</td>` +
      `<td>${h.num_items}</td><td>${h.distance}</td>` +
      `<td><span class="mode-tag ${h.mode}">${h.mode}</span></td>`;
    body.appendChild(tr);
  });
}
 
// Load any existing history on page load (e.g. after a refresh)
fetch('/api/history').then(r => r.json()).then(data => renderHistory(data.history));
</script>
</body>
</html>
"""