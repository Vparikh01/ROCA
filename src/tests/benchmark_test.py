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
NUM_RUNS = 10
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

# All optimization modes to test
OPTIMIZATION_MODES = [
    'plain',
    'gradient',
    'evolution',
    'qlearning',
    'qlearning+gradient',
    'qlearning+evolution'
]

MODE_LABELS = {
    'plain': 'Plain ACO',
    'gradient': 'Gradient',
    'evolution': 'Evolution',
    'qlearning': 'Q-Learning',
    'qlearning+gradient': 'Q + Gradient',
    'qlearning+evolution': 'Q + Evolution'
}

MODE_COLORS = {
    'plain': '#000000',          # Black
    'gradient': '#E74C3C',       # Bright Red
    'evolution': '#3498DB',      # Bright Blue
    'qlearning': '#F39C12',      # Bright Orange
    'qlearning+gradient': '#9B59B6',  # Purple
    'qlearning+evolution': '#2ECC71'  # Bright Green
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
def run_aco_single(G, required_nodes, cost_matrix, seed, optimization_mode):
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
        optimization_mode=optimization_mode,
        seed=seed
    )
    
    # For plain mode, disable evolution
    if optimization_mode == 'plain':
        aco.macro_iter_size = NUM_ITERATIONS + 1
    
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

# ------------------ Main Comparison ------------------
def run_comprehensive_comparison():
    print(f"Loading {TSPLIB_PKL_PATH}...")
    G, coords = load_tsplib_graph(TSPLIB_PKL_PATH)
    required_nodes = list(G.nodes())
    num_nodes = len(required_nodes)
    
    cost_matrix = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            cost_matrix[i][j] = 0 if i == j else np.linalg.norm(coords[i] - coords[j])
    
    # Store results for each mode
    results = {mode: [] for mode in OPTIMIZATION_MODES}
    
    for seed in SEEDS:
        print(f"\n{'='*60}\nSeed {seed}\n{'='*60}")
        for mode in OPTIMIZATION_MODES:
            print(f"  Running: {MODE_LABELS[mode]}")
            results[mode].append(
                run_aco_single(G, required_nodes, cost_matrix, seed, mode)
            )
    
    analyze_results(results, NUM_RUNS)
    visualize_results(results, NUM_RUNS)

# ------------------ Analysis ------------------
def analyze_results(results, n_trials):
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)
    print(f"{'Mode':<25} {'Mean Cost':<15} {'Std Cost':<15} {'Best Cost':<15}")
    print("-"*80)
    
    for mode in OPTIMIZATION_MODES:
        costs = [r['best_cost'] for r in results[mode]]
        print(f"{MODE_LABELS[mode]:<25} {np.mean(costs):>12.2f}    "
              f"{np.std(costs):>12.2f}    {np.min(costs):>12.2f}")
    
    print("="*80)
    
    # Pairwise comparison with plain
    print("\nIMPROVEMENT vs PLAIN ACO:")
    print("-"*80)
    plain_costs = [r['best_cost'] for r in results['plain']]
    plain_mean = np.mean(plain_costs)
    
    for mode in OPTIMIZATION_MODES:
        if mode == 'plain':
            continue
        costs = [r['best_cost'] for r in results[mode]]
        improvement = ((plain_mean - np.mean(costs)) / plain_mean) * 100
        wins = sum(1 for p, c in zip(plain_costs, costs) if c < p)
        win_rate = (wins / n_trials) * 100
        print(f"{MODE_LABELS[mode]:<25} Improvement: {improvement:>6.2f}%  |  Win Rate: {win_rate:>5.1f}%")

# ------------------ Visualization ------------------
def visualize_results(results, n_trials):
    # Create two separate full-screen windows
    
    # ==================== WINDOW 1: Convergence & Performance ====================
    fig1 = plt.figure(figsize=(19, 10))
    fig1.canvas.manager.set_window_title('ACO Comparison - Convergence & Performance')
    gs1 = fig1.add_gridspec(2, 2, hspace=0.3, wspace=0.3, 
                            left=0.08, right=0.95, top=0.93, bottom=0.08)
    
    # --------------------------------------------------
    # 1. Convergence curves (all trials, all modes)
    # --------------------------------------------------
    ax1 = fig1.add_subplot(gs1[0, :])
    for mode in OPTIMIZATION_MODES:
        for i, r in enumerate(results[mode]):
            ax1.plot(
                r['history'],
                color=MODE_COLORS[mode],
                alpha=0.6,
                linewidth=2.0,
                linestyle='--' if mode == 'plain' else '-',
                label=MODE_LABELS[mode] if i == 0 else ""
            )
    ax1.set_title("Convergence Curves - All Modes & Trials", fontweight='bold', fontsize=14)
    ax1.set_xlabel("Iteration", fontsize=12)
    ax1.set_ylabel("Best Tour Cost", fontsize=12)
    ax1.legend(loc='upper right', fontsize=11, framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    
    # --------------------------------------------------
    # 2. Mean ± Std convergence
    # --------------------------------------------------
    ax2 = fig1.add_subplot(gs1[1, :])
    max_len = max(len(r['history']) for group in results.values() for r in group)
    
    for mode in OPTIMIZATION_MODES:
        histories = [
            r['history'] + [r['history'][-1]] * (max_len - len(r['history']))
            for r in results[mode]
        ]
        mean = np.mean(histories, axis=0)
        std = np.std(histories, axis=0)
        
        ax2.plot(mean, color=MODE_COLORS[mode], linewidth=3.0, label=MODE_LABELS[mode])
        ax2.fill_between(
            range(len(mean)),
            mean - std,
            mean + std,
            color=MODE_COLORS[mode],
            alpha=0.2
        )
    
    ax2.set_title(f"Average Convergence with Std Dev (n={n_trials})", fontweight='bold', fontsize=14)
    ax2.set_xlabel("Iteration", fontsize=12)
    ax2.set_ylabel("Best Tour Cost", fontsize=12)
    ax2.legend(loc='upper right', fontsize=11, framealpha=0.9, ncol=2)
    ax2.grid(True, alpha=0.3)
    
    fig1.suptitle(f"ACO Optimization Mode Comparison - Convergence Analysis\n{NUM_ITERATIONS} Iterations, {NUM_RUNS} Run(s)", 
                  fontsize=15, fontweight='bold')
    
    # ==================== WINDOW 2: Statistics & Analysis ====================
    fig2 = plt.figure(figsize=(19, 10))
    fig2.canvas.manager.set_window_title('ACO Comparison - Statistics & Analysis')
    gs2 = fig2.add_gridspec(2, 3, hspace=0.35, wspace=0.35,
                            left=0.08, right=0.95, top=0.93, bottom=0.08)
    
    plain_costs = [r['best_cost'] for r in results['plain']]
    plain_mean = np.mean(plain_costs)
    
    # --------------------------------------------------
    # 3. Box plot of final costs
    # --------------------------------------------------
    ax3 = fig2.add_subplot(gs2[0, :2])
    all_costs = [[r['best_cost'] for r in results[mode]] for mode in OPTIMIZATION_MODES]
    
    bp = ax3.boxplot(
        all_costs,
        labels=[MODE_LABELS[m] for m in OPTIMIZATION_MODES],
        showmeans=True,
        patch_artist=True,
        widths=0.6
    )
    
    for patch, mode in zip(bp['boxes'], OPTIMIZATION_MODES):
        patch.set_facecolor(MODE_COLORS[mode])
        patch.set_alpha(0.7)
        patch.set_linewidth(1.5)
    
    ax3.set_title("Final Cost Distribution by Mode", fontweight='bold', fontsize=14)
    ax3.set_ylabel("Tour Cost", fontsize=12)
    ax3.set_xticklabels([MODE_LABELS[m] for m in OPTIMIZATION_MODES], rotation=30, ha='right', fontsize=10)
    ax3.grid(True, axis='y', alpha=0.3)
    
    # --------------------------------------------------
    # 4. Improvement over Plain (bar chart)
    # --------------------------------------------------
    ax4 = fig2.add_subplot(gs2[0, 2])
    
    improvements = []
    labels_short = []
    for mode in OPTIMIZATION_MODES:
        if mode == 'plain':
            continue
        costs = [r['best_cost'] for r in results[mode]]
        improvement = ((plain_mean - np.mean(costs)) / plain_mean) * 100
        improvements.append(improvement)
        labels_short.append(MODE_LABELS[mode])
    
    colors_bar = [MODE_COLORS[m] for m in OPTIMIZATION_MODES if m != 'plain']
    bars = ax4.barh(range(len(improvements)), improvements, color=colors_bar, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax4.axvline(0, color='red', linestyle='--', linewidth=2)
    ax4.set_yticks(range(len(improvements)))
    ax4.set_yticklabels(labels_short, fontsize=10)
    ax4.set_title("% Improvement\nvs Plain ACO", fontweight='bold', fontsize=13)
    ax4.set_xlabel("Improvement (%)", fontsize=11)
    ax4.grid(True, axis='x', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, improvements)):
        width = bar.get_width()
        ax4.text(width, bar.get_y() + bar.get_height()/2.,
                f' {val:.1f}%', ha='left' if val > 0 else 'right', va='center', fontsize=10, fontweight='bold')
    
    # --------------------------------------------------
    # 5. Parameter scatter (Alpha vs Beta)
    # --------------------------------------------------
    ax5 = fig2.add_subplot(gs2[1, 0])
    for mode in OPTIMIZATION_MODES:
        for i, r in enumerate(results[mode]):
            ax5.scatter(
                r['final_alphas'],
                r['final_betas'],
                alpha=0.6,
                s=60,
                color=MODE_COLORS[mode],
                label=MODE_LABELS[mode] if i == 0 else "",
                marker='o' if 'qlearning' not in mode else '^',
                edgecolors='black',
                linewidths=0.5
            )
    
    ax5.set_xlabel("Alpha (α)", fontsize=12)
    ax5.set_ylabel("Beta (β)", fontsize=12)
    ax5.set_title("Final Parameter Distribution", fontweight='bold', fontsize=13)
    ax5.legend(loc='best', fontsize=9, framealpha=0.9)
    ax5.grid(True, alpha=0.3)
    
    # --------------------------------------------------
    # 6. Win rates vs Plain (pie chart)
    # --------------------------------------------------
    ax6 = fig2.add_subplot(gs2[1, 1])
    
    win_counts = []
    for mode in OPTIMIZATION_MODES:
        if mode == 'plain':
            continue
        costs = [r['best_cost'] for r in results[mode]]
        wins = sum(1 for p, c in zip(plain_costs, costs) if c < p)
        win_counts.append(wins)
    
    colors_pie = [MODE_COLORS[m] for m in OPTIMIZATION_MODES if m != 'plain']
    labels_pie = [MODE_LABELS[m] for m in OPTIMIZATION_MODES if m != 'plain']
    
    wedges, texts, autotexts = ax6.pie(
        win_counts,
        labels=labels_pie,
        autopct='%1.1f%%',
        startangle=90,
        colors=colors_pie,
        textprops={'fontsize': 10}
    )
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(11)
    
    ax6.set_title("Win Distribution\nvs Plain ACO", fontweight='bold', fontsize=13)
    
    # --------------------------------------------------
    # 7. Summary statistics table
    # --------------------------------------------------
    ax7 = fig2.add_subplot(gs2[1, 2])
    ax7.axis('off')
    
    table_data = [['Mode', 'Mean', 'Std', 'Best', 'vs Plain']]
    
    for mode in OPTIMIZATION_MODES:
        costs = [r['best_cost'] for r in results[mode]]
        mean_cost = np.mean(costs)
        std_cost = np.std(costs)
        best_cost = np.min(costs)
        
        if mode == 'plain':
            vs_plain = '-'
        else:
            improvement = ((plain_mean - mean_cost) / plain_mean) * 100
            vs_plain = f"{improvement:+.1f}%"
        
        table_data.append([
            MODE_LABELS[mode],
            f"{mean_cost:.1f}",
            f"{std_cost:.1f}",
            f"{best_cost:.1f}",
            vs_plain
        ])
    
    table = ax7.table(
        cellText=table_data,
        cellLoc='center',
        loc='center',
        colWidths=[0.30, 0.20, 0.15, 0.15, 0.20]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # Style header row
    for i in range(5):
        table[(0, i)].set_facecolor('#2C3E50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color rows by mode
    for i, mode in enumerate(OPTIMIZATION_MODES, start=1):
        table[(i, 0)].set_facecolor(MODE_COLORS[mode])
        table[(i, 0)].set_alpha(0.4)
        table[(i, 0)].set_text_props(weight='bold')
    
    ax7.set_title("Comprehensive Statistics", fontweight='bold', fontsize=13, pad=20)
    
    fig2.suptitle(f"ACO Optimization Mode Comparison - Statistical Analysis\n{NUM_ITERATIONS} Iterations, {NUM_RUNS} Run(s)", 
                  fontsize=15, fontweight='bold')
    
    plt.show()

if __name__ == "__main__":
    run_comprehensive_comparison()