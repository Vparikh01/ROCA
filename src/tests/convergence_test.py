import os
import pickle
import numpy as np
import networkx as nx
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.aco.engine import MaxMinACO

# ============================================================
# CONFIG
# ============================================================
NUM_RUNS = 2
I_MAX = 200
WINDOW = 50
STABILITY_THRESHOLD = 0.04  # 4%
SEEDS = list(range(1, NUM_RUNS + 1))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GRAPH_PATH = os.path.join(BASE_DIR, "..", "tsplib_graphs", "bier127.pkl")

# ============================================================
# LOAD GRAPH
# ============================================================
def load_graph(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    coords = np.array(data["coordinates"])
    n = len(coords)

    G = nx.Graph()
    for i in range(n):
        for j in range(i + 1, n):
            w = float(np.linalg.norm(coords[i] - coords[j]))
            G.add_edge(i, j, weight=w)

    return G, coords


# ============================================================
# RUN ONE ACO INSTANCE
# ============================================================
def run_single(G, coords, seed):
    num_nodes = len(coords)

    # build cost matrix
    cost_matrix = np.zeros((num_nodes, num_nodes), float)
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                cost_matrix[i][j] = float(np.linalg.norm(coords[i] - coords[j]))

    required_nodes = list(range(num_nodes))
    index_map = {i: i for i in required_nodes}

    # create ACO engine
    aco = MaxMinACO(
        cost_matrix,
        start_node=0,
        reducedGraph=G,
        completeGraph=G,
        shortest_paths={},
        required_nodes=required_nodes,
        index_map=index_map,
        seed=seed
    )

    mean_costs = []

    # ----- run for up to I_MAX iterations -----
    for it in range(I_MAX):
        aco.run(iterations=1, n=it)

        # gather all ant costs for μ_t (mean cost)
        costs = []
        for ant in aco.ants:
            if ant is None or len(ant.tour) != num_nodes:
                continue

            total = 0
            valid = True
            for k in range(num_nodes):
                u = ant.tour[k]
                v = ant.tour[(k + 1) % num_nodes]
                c = cost_matrix[u][v]
                if np.isinf(c):
                    valid = False
                    break
                total += c

            if valid:
                costs.append(total)

        μ_t = float(np.mean(costs)) if costs else np.nan
        mean_costs.append(μ_t)

    return mean_costs


# ============================================================
# COMPUTE t_conv FOR ONE RUN
# ============================================================
def compute_convergence(mean_costs):
    mean_costs = np.array(mean_costs)

    for t in range(0, len(mean_costs) - WINDOW):
        window = mean_costs[t:t + WINDOW]

        if np.any(np.isnan(window)):
            continue

        change = np.max(window) - np.min(window)
        change_rel = change / window[0]

        if change_rel <= STABILITY_THRESHOLD:
            return t  # first stable iteration

    return None  # failed to converge


# ============================================================
# MAIN BENCHMARK
# ============================================================
def run_benchmark():
    print(f"Loading {GRAPH_PATH}...")
    G, coords = load_graph(GRAPH_PATH)
    print(f"Loaded graph with {len(coords)} nodes.")

    all_mean_costs = []
    all_tconv = []

    print(f"Running {NUM_RUNS} convergence tests...")

    for seed in SEEDS:
        μ_list = run_single(G, coords, seed)
        t_conv = compute_convergence(μ_list)

        all_mean_costs.append(μ_list)
        all_tconv.append(t_conv)

        print(f"  Seed {seed:2d} → t_conv = {t_conv}")

    # pass/fail
    successes = sum(t is not None and t <= I_MAX for t in all_tconv)
    proportion = successes / NUM_RUNS

    print("\n==========================")
    print(" Convergence Test Results ")
    print("==========================")
    print(f"Passes: {successes}/{NUM_RUNS} ({proportion*100:.1f}%)")
    print(f"Pass (≥90% runs): {'PASS' if proportion >= 0.90 else 'FAIL'}")

    visualize(all_mean_costs, all_tconv, proportion)


# ============================================================
# VISUALIZATION
# ============================================================
def visualize(all_mean_costs, all_tconv, proportion_pass):
    NUM_RUNS = len(all_mean_costs)
    max_len = max(len(r) for r in all_mean_costs)

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=[
            "Mean Route Cost per Iteration (All Runs)",
            "Distribution of Convergence Iterations"
        ],
        vertical_spacing=0.15
    )

    # ========== Plot 1: Cost trajectories ==========
    arr = np.full((NUM_RUNS, max_len), np.nan)
    for i, L in enumerate(all_mean_costs):
        arr[i, :len(L)] = L
        fig.add_trace(
            go.Scatter(
                x=list(range(len(L))),
                y=L,
                mode='lines',
                opacity=0.3,
                line=dict(color="blue"),
                showlegend=False
            ),
            row=1, col=1
        )

    # average curve
    avg = np.nanmean(arr, axis=0)
    fig.add_trace(
        go.Scatter(
            x=list(range(len(avg))),
            y=avg,
            mode="lines",
            line=dict(color="red", width=3),
            name="Mean"
        ),
        row=1, col=1
    )

    fig.update_xaxes(title_text="Iteration", row=1, col=1)
    fig.update_yaxes(title_text="Mean Cost", row=1, col=1)

    # ========== Plot 2: Histogram of t_conv ==========
    finite_t = [t for t in all_tconv if t is not None]

    fig.add_trace(
        go.Histogram(
            x=finite_t,
            nbinsx=20,
            marker_color="blue",
        ),
        row=2, col=1
    )

    fig.update_xaxes(title_text="t_conv (iteration)", row=2, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=1)

    # layout
    fig.update_layout(
        height=900,
        width=1000,
        title_text=f"Convergence Stability Test — Pass Rate: {proportion_pass*100:.1f}%",
        template="plotly_white"
    )

    fig.show()

if __name__ == "__main__":
    run_benchmark()
