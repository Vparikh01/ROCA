import os
import pickle
import numpy as np
import networkx as nx
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.aco.engine import MaxMinACO
import plotly.io as pio

# ---------------- CONFIG ----------------
NUM_RUNS = 1
I_MAX = 200
H_WINDOW = 20
CV_THRESH = 0.25
DELTA_FACTOR = 0.05
SEEDS = list(range(1, NUM_RUNS+1))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GRAPH_PATH = os.path.join(BASE_DIR, "..", "tsplib_graphs", "bier127.pkl")
pio.renderers.default = "browser"

# ---------------- LOAD GRAPH ----------------
def load_graph(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    coords = np.array(data["coordinates"])
    n = len(coords)
    G = nx.Graph()
    for i in range(n):
        for j in range(i+1, n):
            w = float(np.linalg.norm(coords[i] - coords[j]))
            G.add_edge(i, j, weight=w)
    return G, coords

# ---------------- RUN ONE INSTANCE ----------------
def run_single(G, coords, seed, i_max=I_MAX):
    num_nodes = len(coords)
    cost_matrix = np.zeros((num_nodes, num_nodes), float)
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                cost_matrix[i][j] = float(np.linalg.norm(coords[i]-coords[j]))

    required_nodes = list(range(num_nodes))
    index_map = {i:i for i in required_nodes}

    engine = MaxMinACO(
        cost_matrix, start_node=0,
        reducedGraph=G, completeGraph=G,
        shortest_paths={}, required_nodes=required_nodes,
        index_map=index_map, seed=seed
    )

    cv_list = []
    eps = 1e-12
    
    for iteration in range(i_max):
        engine.run(iterations=1)
        tau = getattr(engine.pheromones, "matrix", None)
        if tau is None:
            tau = getattr(engine, "tau", None)
        if tau is None:
            cv_list.append(np.nan)
            continue

        n = tau.shape[0]
        pvals = np.array([tau[i,j] for i in range(n) for j in range(i+1, n)], dtype=float)
        pvals = np.nan_to_num(pvals, nan=0.0, posinf=0.0, neginf=0.0)
        
        # CV on top-k edges only (based on solution tour size)
        top_k = int(n * 2)  # solution has n edges, look at 2x that
        pvals_sorted = np.sort(pvals)[::-1]
        pvals_topk = pvals_sorted[:min(top_k, len(pvals_sorted))]
        mean_topk = np.mean(pvals_topk)
        std_topk = np.std(pvals_topk)
        cv_topk = std_topk / (mean_topk + eps)
        cv_list.append(cv_topk)

    return cv_list

# ---------------- EVALUATE STABILITY ----------------
def evaluate_cv_stability(cv_list, h_window=H_WINDOW, thresh=CV_THRESH, delta_factor=DELTA_FACTOR):
    cv = np.array(cv_list, dtype=float)
    successes = 0
    total_candidates = 0
    candidates = []

    if np.isnan(cv).all() or len(cv) < h_window:
        return successes, total_candidates, candidates

    for N in range(len(cv)-h_window+1):
        cv_start = cv[N]
        cv_end = cv[N+h_window-1]
        if np.isnan(cv_start) or np.isnan(cv_end):
            continue
        if cv_start >= thresh:
            total_candidates += 1
            delta = abs(cv_end - cv_start)
            passed = delta <= delta_factor * max(cv_start, 1e-12)
            if passed: successes += 1
            candidates.append((N, cv_start, cv_end, delta, passed))
    return successes, total_candidates, candidates

# ---------------- VISUALIZATION ----------------
def visualize(all_cv, proportion):
    fig = go.Figure()
    n_runs = len(all_cv)

    # Compute mean/std if more than 1 run
    if n_runs > 1:
        min_len = min(len(cv) for cv in all_cv)
        stacked = np.array([cv[:min_len] for cv in all_cv], dtype=float)
        mean_cv = np.mean(stacked, axis=0)
        std_cv = np.std(stacked, axis=0)

        # Plot the ±std band first
        fig.add_trace(go.Scatter(
            x=list(range(min_len)) + list(range(min_len))[::-1],
            y=list(mean_cv + std_cv) + list((mean_cv - std_cv)[::-1]),
            fill='toself',
            fillcolor='rgba(0,100,255,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=True,
            name="Mean ± Std"
        ))

    # Plot each individual run on top
    for cv in all_cv:
        if not cv: continue
        fig.add_trace(go.Scatter(
            x=list(range(len(cv))),
            y=cv,
            mode="lines",
            opacity=0.5,  # higher so visible
            line=dict(color="blue", width=1),
            showlegend=False
        ))

    # Plot mean line on top
    if n_runs > 1:
        fig.add_trace(go.Scatter(
            x=list(range(min_len)),
            y=mean_cv,
            mode="lines",
            line=dict(color="red", width=4),
            name="Mean CV"
        ))

    fig.update_layout(
        title=f"CV over iterations — Mean stability: {proportion*100:.1f}%",
        xaxis_title="Iteration",
        yaxis_title="Coefficient of Variation (Top-K)",
        template="plotly_white",
        height=600,
        width=900
    )
    fig.show()


# ---------------- MAIN BENCHMARK ----------------
def run_benchmark():
    print(f"Loading {GRAPH_PATH}...")
    G, coords = load_graph(GRAPH_PATH)
    print(f"Loaded graph with {len(coords)} nodes.")

    all_cv = []
    all_ratios = []
    all_candidates = []

    print(f"Running {NUM_RUNS} CV stability tests (Top-K Edges)...")
    for seed in SEEDS:
        cv_list = run_single(G, coords, seed, i_max=I_MAX)
        successes, total, candidates = evaluate_cv_stability(cv_list)
        all_cv.append(cv_list)
        all_ratios.append((successes, total))
        all_candidates.append(candidates)
        ratio_percent = (successes / total * 100.0) if total > 0 else 0.0
        print(f" Seed {seed:2d} → candidates={total}, pass_rate={ratio_percent:.1f}%")

    global_successes = sum(s for s, t in all_ratios)
    global_total = sum(t for s, t in all_ratios)
    proportion = (global_successes / global_total) if global_total else 0.0

    print("\n==============================================")
    print(" Pheromone Stability (CV on Top-K Edges) ")
    print("==============================================")
    print(f"Total candidate windows: {global_total}")
    print(f"Stable windows: {global_successes}")
    print(f"Global stability rate: {proportion*100:.1f}%")
    print(f"Pass (≥70% stable windows): {'PASS' if proportion >= 0.70 else 'FAIL'}")

    visualize(all_cv, proportion)

# ---------------- ENTRY ----------------
if __name__ == "__main__":
    run_benchmark()