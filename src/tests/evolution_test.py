import os
import pickle
from pathlib import Path
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from src.aco.engine import MaxMinACO

# ------------------ Configuration ------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TSPLIB_PKL_PATH = os.path.join(BASE_DIR, "..", "tsplib_graphs", "bier127.pkl")

NUM_RUNS = 2
NUM_ITERATIONS = 200
SEEDS = list(range(1, NUM_RUNS + 1))

TSPLIB_OPTIMAL = {
    'burma14': 3323,
    'ulysses16': 6859,
    'att48': 33600,
    'berlin52': 7542,
    'eil51': 426,
    'kroA100': 21282,
    'lin105': 14379,
    'pr107': 44303,
    'a280': 2579,
    'bier127': 118282,
}

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

# ------------------ Single ACO run ------------------
def run_aco_single(G, required_nodes, cost_matrix, seed, use_evolution=True, useQlearning=False):
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
        isQlearning=useQlearning,
        seed=seed
    )

    if not use_evolution:
        aco.macro_iter_size = NUM_ITERATIONS + 1
        for ant in aco.ants:
            ant.alpha = aco.alpha
            ant.beta = aco.beta

    for it in range(NUM_ITERATIONS):
        aco.run(iterations=1, n=it)

    best_cost = np.nan
    if aco.best_tour is not None:
        best_cost = float(
            sum(cost_matrix[aco.best_tour[i]][aco.best_tour[(i + 1) % num_nodes]]
                for i in range(num_nodes))
        )

    return {
        'best_cost': best_cost,
        'history': aco.best_length_history,
        'final_alphas': [ant.alpha for ant in aco.ants],
        'final_betas': [ant.beta for ant in aco.ants]
    }

# ------------------ Evolution Comparison ------------------
def run_evolution_comparison():
    print(f"Loading {TSPLIB_PKL_PATH}...")
    G, coords = load_tsplib_graph(TSPLIB_PKL_PATH)
    required_nodes = list(G.nodes())
    num_nodes = len(required_nodes)

    cost_matrix = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            cost_matrix[i][j] = 0 if i == j else np.linalg.norm(coords[i] - coords[j])

    results = {
        'q_evolved': [],
        'evolved': [],
        'fixed': []
    }

    for seed in SEEDS:
        print(f"\nSeed {seed}")

        print("  Q-learning + Evolution")
        results['q_evolved'].append(
            run_aco_single(G, required_nodes, cost_matrix, seed, use_evolution=True, useQlearning=True)
        )

        print("  Evolution only")
        results['evolved'].append(
            run_aco_single(G, required_nodes, cost_matrix, seed, use_evolution=True)
        )

        print("  Fixed")
        results['fixed'].append(
            run_aco_single(G, required_nodes, cost_matrix, seed, use_evolution=False)
        )

    analyze_results(results, NUM_RUNS)
    visualize_results(results, NUM_RUNS)

# ------------------ Analysis ------------------
def analyze_results(results, n_trials):
    q_costs = [r['best_cost'] for r in results['q_evolved']]
    e_costs = [r['best_cost'] for r in results['evolved']]
    f_costs = [r['best_cost'] for r in results['fixed']]

    print("\nFINAL RESULTS")
    print(f"Q + Evo:  μ={np.mean(q_costs):.2f} ± {np.std(q_costs):.2f}")
    print(f"Evo:      μ={np.mean(e_costs):.2f} ± {np.std(e_costs):.2f}")
    print(f"Fixed:    μ={np.mean(f_costs):.2f} ± {np.std(f_costs):.2f}")

# ------------------ Visualization ------------------
def visualize_results(results, n_trials):
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. Convergence curves
    ax1 = fig.add_subplot(gs[0, :2])
    for i, (q, e, f) in enumerate(zip(results['q_evolved'], results['evolved'], results['fixed'])):
        ax1.plot(q['history'], color='green', alpha=0.4, linewidth=1.5, label='Q + Evo' if i==0 else '')
        ax1.plot(e['history'], color='blue', alpha=0.4, linewidth=1.5, label='Evolution' if i==0 else '')
        ax1.plot(f['history'], color='red', alpha=0.4, linestyle='--', linewidth=1.5, label='Fixed' if i==0 else '')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Best Tour Cost')
    ax1.set_title('Convergence Curves (All Trials)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Box plot
    ax2 = fig.add_subplot(gs[1, 0])
    q_costs = [r['best_cost'] for r in results['q_evolved']]
    e_costs = [r['best_cost'] for r in results['evolved']]
    f_costs = [r['best_cost'] for r in results['fixed']]
    ax2.boxplot([q_costs, e_costs, f_costs], labels=['Q + Evo', 'Evolution', 'Fixed'], showmeans=True)
    ax2.set_ylabel('Final Tour Cost')
    ax2.set_title('Cost Distribution')
    ax2.grid(True)

    # 3. Delta histogram (Fixed - Q+Evo)
    ax3 = fig.add_subplot(gs[1, 1])
    deltas = [f - q for f, q in zip(f_costs, q_costs)]
    ax3.hist(deltas, bins=10, color='green', alpha=0.7, edgecolor='black')
    ax3.axvline(0, color='red', linestyle='--')
    ax3.set_xlabel('Δ = Fixed − Q+Evo')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Improvement Distribution')

    plt.show()

if __name__ == "__main__":
    run_evolution_comparison()
