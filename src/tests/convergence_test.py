import os
import pickle
from pathlib import Path
import numpy as np
import networkx as nx
from itertools import permutations
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.aco.engine import MaxMinACO

# ------------------ Configuration ------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TSPLIB_PKL_PATH = os.path.join(BASE_DIR, "..", "tsplib_graphs", "att48.pkl")

NUM_RUNS = 10
NUM_ITERATIONS = 200
SEEDS = list(range(1, NUM_RUNS + 1))
CONVERGENCE_PATIENCE = 50

TSPLIB_OPTIMAL = {
    # Small / low complexity (20–50 nodes)
    'burma14': 3323,
    'ulysses16': 6859,
    'att48': 33600,
    'berlin52': 7542,
    'eil51': 426,

    # Medium complexity (100–300 nodes)
    'kroA100': 21282,
    'lin105': 14379,
    'pr107': 44303,
    'a280': 2579,
    'bier127': 118282,

    # Complex (500–2000 nodes)
    'pcb442': 50778,
    'rat575': 6773,
    'd657': 48912,
    'fl1400': 20127,
    # 'pla85900': 142382641,  # skip if strictly ≤2000 nodes
}

FORCE_OPTIMAL_COST = float(TSPLIB_OPTIMAL.get(Path(TSPLIB_PKL_PATH).stem, np.nan))  # set to None to compute

# ------------------ Helpers -------------------
def load_tsplib_graph(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    coords = np.asarray(data["coordinates"])
    n = len(coords)
    G = nx.Graph()
    for i in range(n):
        for j in range(i + 1, n):
            dist = float(np.linalg.norm(coords[i] - coords[j]))
            G.add_edge(i, j, weight=dist)
    return G, coords

def compute_optimal(G, num_nodes):
    if num_nodes <= 10:
        cost_matrix = nx.to_numpy_array(G, weight="weight")
        best_cost = float("inf")
        for perm in permutations(range(1, num_nodes)):
            tour = [0] + list(perm)
            tour_cost = sum(cost_matrix[tour[i]][tour[i + 1]] for i in range(num_nodes - 1))
            tour_cost += cost_matrix[tour[-1]][0]
            if tour_cost < best_cost:
                best_cost = tour_cost
        return best_cost
    else:
        approx_tour = nx.approximation.traveling_salesman_problem(
            G, cycle=True, weight="weight", method=nx.approximation.christofides
        )
        cost_matrix = nx.to_numpy_array(G, weight="weight")
        tour_cost = sum(cost_matrix[approx_tour[i]][approx_tour[(i + 1) % num_nodes]] for i in range(num_nodes))
        return float(tour_cost)

# ------------------ Single ACO run ------------------
def run_aco_single(G, required_nodes, cost_matrix, seed, max_iterations=NUM_ITERATIONS, patience=CONVERGENCE_PATIENCE):
    num_nodes = len(required_nodes)
    index_map = {n: i for i, n in enumerate(required_nodes)}

    G_indexed = nx.Graph()
    for u, v, d in G.edges(data=True):
        if u in index_map and v in index_map:
            G_indexed.add_edge(index_map[u], index_map[v], **d)

    aco = MaxMinACO(
        cost_matrix,
        start_node=0,
        reducedGraph=G_indexed,
        completeGraph=G,
        shortest_paths={},
        required_nodes=required_nodes,
        index_map=index_map,
        seed=seed
    )

    iteration_mean_costs = []
    iteration_best_costs = []  # Track best-so-far each iteration
    best_seen = np.inf
    last_improve_iter = 0
    iterations_run = 0

    for it in range(max_iterations):
        iterations_run += 1
        aco.run(iterations=1, n=it)

        # Compute mean cost across all ants this iteration
        ant_costs = []
        for ant in aco.ants:
            if ant is not None and hasattr(ant, "tour") and len(ant.tour) == num_nodes:
                tour_cost = 0.0
                valid = True
                for k in range(num_nodes):
                    u = ant.tour[k]
                    v = ant.tour[(k + 1) % num_nodes]
                    c = cost_matrix[u][v]
                    if np.isinf(c):
                        valid = False
                        break
                    tour_cost += c
                if valid:
                    ant_costs.append(tour_cost)

        if ant_costs:
            iteration_mean_costs.append(float(np.mean(ant_costs)))
        else:
            iteration_mean_costs.append(np.nan)

        # Track best-so-far cost each iteration
        if aco.best_length is not None and np.isfinite(aco.best_length):
            iteration_best_costs.append(float(aco.best_length))
            if aco.best_length + 1e-12 < best_seen:
                best_seen = aco.best_length
                last_improve_iter = it
        else:
            iteration_best_costs.append(best_seen if np.isfinite(best_seen) else np.nan)

        if (it - last_improve_iter) >= patience:
            break

    # Get best tour cost
    best_cost = np.nan
    if aco.best_tour is not None:
        try:
            best_cost = float(sum(cost_matrix[aco.best_tour[i]][aco.best_tour[(i + 1) % num_nodes]] for i in range(num_nodes)))
        except:
            best_cost = np.nan

    return iteration_mean_costs, iteration_best_costs, best_cost, iterations_run

# ------------------ Benchmark ------------------
def run_benchmark():
    print(f"Loading {TSPLIB_PKL_PATH}...")
    G, coords = load_tsplib_graph(TSPLIB_PKL_PATH)
    required_nodes = list(G.nodes())
    num_nodes = len(required_nodes)
    print(f"Loaded: {num_nodes} nodes")

    # Build cost matrix
    cost_matrix = np.zeros((num_nodes, num_nodes), dtype=float)
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i == j:
                cost_matrix[i][j] = 0.0
            else:
                cost_matrix[i][j] = float(np.linalg.norm(coords[i] - coords[j]))

    # Get optimal
    if FORCE_OPTIMAL_COST is not None:
        optimal_cost = float(FORCE_OPTIMAL_COST)
        print(f"Using forced optimal: {optimal_cost:.2f}")
    else:
        print("Computing optimal...")
        optimal_cost = compute_optimal(G, num_nodes)
        print(f"Optimal: {optimal_cost:.2f}")

    all_iteration_mean_costs = []
    all_iteration_best_costs = []
    all_best_costs = []
    all_deviations = []
    all_final20_means = []

    print(f"\nRunning {NUM_RUNS} runs...")
    for seed in SEEDS:
        iter_mean_costs, iter_best_costs, best_cost, T = run_aco_single(G, required_nodes, cost_matrix, seed)
        all_iteration_mean_costs.append(iter_mean_costs)
        all_iteration_best_costs.append(iter_best_costs)
        all_best_costs.append(best_cost)

        # Get last 20 finite values of BEST cost (not mean across ants)
        finite_best_costs = [c for c in iter_best_costs if np.isfinite(c)]
        if len(finite_best_costs) >= 20:
            final20 = finite_best_costs[-20:]
        else:
            final20 = finite_best_costs
        
        mean_final20 = float(np.mean(final20)) if final20 else np.nan
        all_final20_means.append(mean_final20)
        
        deviation = (mean_final20 - optimal_cost) / optimal_cost if np.isfinite(mean_final20) else np.nan
        all_deviations.append(deviation)
        
        print(f"  Seed {seed}: T={T}, Best={best_cost:.2f}, Final20={mean_final20:.2f}, Dev={deviation:.4f}")

    # Stats
    all_deviations = np.array(all_deviations)
    finite_mask = np.isfinite(all_deviations)
    n_finite = np.sum(finite_mask)

    mean_dev = float(np.mean(all_deviations[finite_mask])) if n_finite else np.nan
    std_dev = float(np.std(all_deviations[finite_mask], ddof=1)) if n_finite > 1 else 0.0
    ci95 = 1.96 * std_dev / np.sqrt(n_finite) if n_finite else np.nan
    num_pass = np.sum((all_deviations <= 0.10) & finite_mask)
    proportion_pass = num_pass / NUM_RUNS

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Optimal: {optimal_cost:.2f}")
    print(f"Mean deviation: {mean_dev:.4f} ({mean_dev*100:.2f}%)")
    print(f"Std deviation: {std_dev:.4f}")
    print(f"95% CI: [{mean_dev - ci95:.4f}, {mean_dev + ci95:.4f}]")
    print(f"Runs ≤10% deviation: {num_pass}/{NUM_RUNS} ({proportion_pass:.1%})")
    print(f"Pass (≥90%): {'✓ PASS' if proportion_pass >= 0.90 else '✗ FAIL'}")
    print("=" * 60)

    visualize_results(all_iteration_best_costs, all_best_costs, all_deviations, optimal_cost, mean_dev, proportion_pass)

def visualize_results(all_iteration_best_costs, all_best_costs, all_deviations, optimal_cost, mean_dev, proportion_pass):
    """
    Visualize Path Optimality Test results:
    - Top-left: Best-so-far convergence curves for all runs + mean + optimal
    - Top-right: Best cost per run (bar chart) + optimal
    - Bottom-left: Deviation percent histogram + 10% threshold
    """
    NUM_RUNS = len(all_best_costs)

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Best-So-Far Convergence (All Runs)",
            "Best Cost per Run",
            "Deviation (%) Across Runs",
            None  # bottom-right empty
        ),
        specs=[[{"type":"scatter"}, {"type":"bar"}],
               [{"type":"histogram"}, None]]
    )

    # ------------------ Top-left: Best-so-far curves ------------------
    max_len = max(len(costs) for costs in all_iteration_best_costs)
    arr = np.full((NUM_RUNS, max_len), np.nan)
    for i, costs in enumerate(all_iteration_best_costs):
        arr[i, :len(costs)] = costs
        # individual run line
        fig.add_trace(
            go.Scatter(
                x=list(range(1, len(costs)+1)),
                y=costs,
                mode='lines',
                line=dict(color='blue', width=1),
                opacity=0.3,
                name=f"Run {i+1}",
                showlegend=False,
                hovertemplate="Iteration %{x}<br>Cost %{y:.2f}"
            ),
            row=1, col=1
        )
    # mean best-so-far line
    mean_best = np.nanmean(arr, axis=0)
    fig.add_trace(
        go.Scatter(
            x=list(range(1, len(mean_best)+1)),
            y=mean_best,
            mode='lines',
            line=dict(color='red', width=3),
            name="Mean Best",
            hovertemplate="Iteration %{x}<br>Mean Best %{y:.2f}"
        ),
        row=1, col=1
    )
    # optimal line
    fig.add_hline(
        y=optimal_cost, line_dash="dash", line_color="green",
        annotation_text="Optimal", row=1, col=1
    )

    fig.update_xaxes(title_text="Iteration", row=1, col=1)
    fig.update_yaxes(title_text="Cost", row=1, col=1)

    # ------------------ Top-right: Best cost per run ------------------
    fig.add_trace(
        go.Bar(
            x=list(range(1, NUM_RUNS+1)),
            y=all_best_costs,
            name="Best Cost per Run",
            marker_color='blue',
            hovertemplate="Run %{x}<br>Best Cost %{y:.2f}"
        ),
        row=1, col=2
    )
    # optimal line
    fig.add_hline(
        y=optimal_cost, line_dash="dash", line_color="green",
        annotation_text="Optimal", row=1, col=2
    )
    fig.update_xaxes(title_text="Run", row=1, col=2)
    fig.update_yaxes(title_text="Cost", row=1, col=2)

    # ------------------ Bottom-left: Deviation histogram ------------------
    finite_devs = [d*100 for d in all_deviations if np.isfinite(d)]
    fig.add_trace(
        go.Histogram(
            x=finite_devs,
            nbinsx=20,
            marker_color='blue',
            name="Deviation (%)",
            hovertemplate="%{x:.2f}%"
        ),
        row=2, col=1
    )
    # 10% threshold
    fig.add_vline(
        x=10, line_dash="dash", line_color="red",
        annotation_text="10% Threshold", row=2, col=1
    )
    fig.update_xaxes(title_text="Deviation (%)", row=2, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=1)

    # ------------------ Layout and annotations ------------------
    fig.update_layout(
        height=900,
        width=1100,
        title_text=f"Path Optimality Test — Mean Dev: {mean_dev*100:.2f}%, Pass: {proportion_pass*100:.1f}%",
        template="plotly_white"
    )
    
    # annotate mean deviation and proportion passing
    fig.add_annotation(
        text=f"Mean Deviation: {mean_dev*100:.2f}%",
        xref="paper", yref="paper",
        x=0.5, y=1.05, showarrow=False,
        font=dict(size=14, color="black")
    )
    fig.add_annotation(
        text=f"Proportion ≤10%: {proportion_pass*100:.1f}%",
        xref="paper", yref="paper",
        x=0.5, y=1.0, showarrow=False,
        font=dict(size=14, color="black")
    )

    fig.show()

if __name__ == "__main__":
    run_benchmark()