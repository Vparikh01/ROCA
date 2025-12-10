import os
import pickle
from pathlib import Path
import numpy as np
import networkx as nx
from itertools import permutations
import plotly as plt
from src.aco.engine import MaxMinACO
import plotly.io as pio

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
    'pcb442': 50778,
    'rat575': 6773,
    'd657': 48912,
    'fl1400': 20127,
}

FORCE_OPTIMAL_COST = float(TSPLIB_OPTIMAL.get(Path(TSPLIB_PKL_PATH).stem, np.nan))
pio.renderers.default = "browser"

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
def run_aco_single(G, required_nodes, cost_matrix, seed, use_evolution=True, max_iterations=NUM_ITERATIONS):
    """Run ACO with or without evolution."""
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
    
    # Disable evolution if requested
    if not use_evolution:
        aco.macro_iter_size = max_iterations + 1
        # Reset all ants to baseline config values
        for ant in aco.ants:
            ant.alpha = aco.alpha  # Use the class-level config values
            ant.beta = aco.beta

    for it in range(max_iterations):
        aco.run(iterations=1, n=it)

    # Get best tour cost
    best_cost = np.nan
    if aco.best_tour is not None:
        try:
            best_cost = float(sum(cost_matrix[aco.best_tour[i]][aco.best_tour[(i + 1) % num_nodes]] for i in range(num_nodes)))
        except:
            best_cost = np.nan
    
    return {
        'best_cost': best_cost,
        'history': aco.best_length_history,
        'final_alphas': [ant.alpha for ant in aco.ants],
        'final_betas': [ant.beta for ant in aco.ants]
    }

# ------------------ Evolution Comparison ------------------
def run_evolution_comparison():
    """Compare ACO with evolution vs without evolution."""
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

    results = {
        'evolved': [],
        'fixed': []
    }

    print("\n" + "="*70)
    print("EVOLUTION COMPARISON")
    print("="*70)
    print(f"Runs: {NUM_RUNS}")
    print(f"Iterations per run: {NUM_ITERATIONS}")
    print()

    for run_idx, seed in enumerate(SEEDS):
        print(f"\n{'─'*70}")
        print(f"Trial {run_idx+1}/{NUM_RUNS} (seed={seed})")
        print(f"{'─'*70}")
        
        # Run WITH evolution
        print("  WITH Evolution:")
        result_evo = run_aco_single(G, required_nodes, cost_matrix, seed, use_evolution=True)
        results['evolved'].append(result_evo)
        print(f"     Final cost: {result_evo['best_cost']:.2f}")
        
        # Run WITHOUT evolution
        print("  WITHOUT Evolution (Fixed):")
        result_fix = run_aco_single(G, required_nodes, cost_matrix, seed, use_evolution=False)
        results['fixed'].append(result_fix)
        print(f"     Final cost: {result_fix['best_cost']:.2f}")
        
        # Show comparison
        delta = result_fix['best_cost'] - result_evo['best_cost']
        improvement = (delta / result_fix['best_cost']) * 100
        winner = "Evolution" if delta > 0 else "Fixed"
        print(f"  → Winner: {winner} (Δ = {delta:+.2f}, {improvement:+.2f}%)")

    # Analyze and visualize
    analyze_results(results, NUM_RUNS)
    visualize_results(results, NUM_RUNS)

def analyze_results(results, n_trials):
    """Analyze and print comparison statistics."""
    evolved_costs = [r['best_cost'] for r in results['evolved']]
    fixed_costs = [r['best_cost'] for r in results['fixed']]
    
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    
    print(f"\nBest Tour Cost (over {n_trials} trials):")
    print(f"  WITH Evolution:    μ={np.mean(evolved_costs):.2f} ± {np.std(evolved_costs):.2f}")
    print(f"                     range=[{np.min(evolved_costs):.2f}, {np.max(evolved_costs):.2f}]")
    print(f"  WITHOUT Evolution: μ={np.mean(fixed_costs):.2f} ± {np.std(fixed_costs):.2f}")
    print(f"                     range=[{np.min(fixed_costs):.2f}, {np.max(fixed_costs):.2f}]")
    
    deltas = [f - e for f, e in zip(fixed_costs, evolved_costs)]
    mean_improvement = (np.mean(fixed_costs) - np.mean(evolved_costs)) / np.mean(fixed_costs) * 100
    
    print(f"\nImprovement (Fixed - Evolved):")
    print(f"  Mean Δ: {np.mean(deltas):+.2f} ({mean_improvement:+.2f}%)")
    print(f"  Median Δ: {np.median(deltas):+.2f}")
    print(f"  Std Δ: {np.std(deltas):.2f}")
    
    wins = sum(1 for d in deltas if d > 0)
    ties = sum(1 for d in deltas if abs(d) < 1e-6)
    losses = sum(1 for d in deltas if d < 0)
    
    print(f"\nHead-to-Head:")
    print(f"  Evolution wins: {wins}/{n_trials} ({100*wins/n_trials:.1f}%)")
    print(f"  Ties: {ties}/{n_trials} ({100*ties/n_trials:.1f}%)")
    print(f"  Evolution losses: {losses}/{n_trials} ({100*losses/n_trials:.1f}%)")
    
    evolved_alphas = [a for r in results['evolved'] for a in r['final_alphas']]
    evolved_betas = [b for r in results['evolved'] for b in r['final_betas']]
    fixed_alphas = [a for r in results['fixed'] for a in r['final_alphas']]
    fixed_betas = [b for r in results['fixed'] for b in r['final_betas']]
    
    print(f"\nFinal Parameter Diversity:")
    print(f"  Evolution - Alpha: μ={np.mean(evolved_alphas):.3f} ± {np.std(evolved_alphas):.3f}")
    print(f"              Beta:  μ={np.mean(evolved_betas):.3f} ± {np.std(evolved_betas):.3f}")
    print(f"  Fixed     - Alpha: μ={np.mean(fixed_alphas):.3f} ± {np.std(fixed_alphas):.3f}")
    print(f"              Beta:  μ={np.mean(fixed_betas):.3f} ± {np.std(fixed_betas):.3f}")
    
    print("\n" + "="*70)
    if wins >= n_trials * 0.7:
        print("✅ VERDICT: Evolution SIGNIFICANTLY BETTER (≥70% win rate)")
    elif wins > losses:
        print("✓ VERDICT: Evolution slightly better")
    elif wins == losses:
        print("≈ VERDICT: No clear winner")
    else:
        print("✗ VERDICT: Fixed parameters better")
    print("="*70)

def visualize_results(results, n_trials):
    """Create comparison visualizations."""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Convergence curves
    ax1 = fig.add_subplot(gs[0, :2])
    for i, (r_evo, r_fix) in enumerate(zip(results['evolved'], results['fixed'])):
        ax1.plot(r_evo['history'], color='blue', alpha=0.4, linewidth=1.5,
                label='Evolution' if i == 0 else '')
        ax1.plot(r_fix['history'], color='red', alpha=0.4, linewidth=1.5, 
                linestyle='--', label='Fixed' if i == 0 else '')
    ax1.set_xlabel('Iteration', fontsize=11)
    ax1.set_ylabel('Best Tour Cost', fontsize=11)
    ax1.set_title('Convergence Curves (All Trials)', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 2. Average convergence
    ax2 = fig.add_subplot(gs[1, :2])
    max_len = max(len(r['history']) for r in results['evolved'])
    evo_histories = [r['history'] + [r['history'][-1]] * (max_len - len(r['history'])) 
                     for r in results['evolved']]
    fix_histories = [r['history'] + [r['history'][-1]] * (max_len - len(r['history'])) 
                     for r in results['fixed']]
    
    evo_mean = np.mean(evo_histories, axis=0)
    evo_std = np.std(evo_histories, axis=0)
    fix_mean = np.mean(fix_histories, axis=0)
    fix_std = np.std(fix_histories, axis=0)
    
    iterations = range(len(evo_mean))
    ax2.plot(iterations, evo_mean, color='blue', linewidth=2.5, label='Evolution')
    ax2.fill_between(iterations, evo_mean - evo_std, evo_mean + evo_std, 
                     color='blue', alpha=0.2)
    ax2.plot(iterations, fix_mean, color='red', linewidth=2.5, linestyle='--', label='Fixed')
    ax2.fill_between(iterations, fix_mean - fix_std, fix_mean + fix_std, 
                     color='red', alpha=0.2)
    ax2.set_xlabel('Iteration', fontsize=11)
    ax2.set_ylabel('Best Tour Cost', fontsize=11)
    ax2.set_title(f'Average Convergence (n={n_trials})', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # 3. Box plot
    ax3 = fig.add_subplot(gs[0, 2])
    evolved_costs = [r['best_cost'] for r in results['evolved']]
    fixed_costs = [r['best_cost'] for r in results['fixed']]
    bp = ax3.boxplot([evolved_costs, fixed_costs], labels=['Evolution', 'Fixed'],
                      patch_artist=True, showmeans=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightcoral')
    ax3.set_ylabel('Final Tour Cost', fontsize=11)
    ax3.set_title('Cost Distribution', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Delta histogram
    ax4 = fig.add_subplot(gs[1, 2])
    deltas = [f - e for f, e in zip(fixed_costs, evolved_costs)]
    ax4.hist(deltas, bins=15, color='green', alpha=0.7, edgecolor='black')
    ax4.axvline(0, color='red', linestyle='--', linewidth=2, label='No difference')
    ax4.axvline(np.mean(deltas), color='blue', linestyle='-', linewidth=2, label='Mean Δ')
    ax4.set_xlabel('Δ = Fixed - Evolution', fontsize=11)
    ax4.set_ylabel('Frequency', fontsize=11)
    ax4.set_title('Improvement Distribution', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Parameter scatter
    ax5 = fig.add_subplot(gs[2, 0])
    for r in results['evolved']:
        ax5.scatter(r['final_alphas'], r['final_betas'], 
                   color='blue', alpha=0.5, s=30, label='Evolution' if r == results['evolved'][0] else '')
    for r in results['fixed']:
        ax5.scatter(r['final_alphas'], r['final_betas'], 
                   color='red', alpha=0.5, s=30, marker='x', label='Fixed' if r == results['fixed'][0] else '')
    ax5.set_xlabel('Alpha (α)', fontsize=11)
    ax5.set_ylabel('Beta (β)', fontsize=11)
    ax5.set_title('Final Parameters', fontsize=12, fontweight='bold')
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)
    
    # 6. Win/Loss pie
    ax6 = fig.add_subplot(gs[2, 1])
    wins = sum(1 for d in deltas if d > 0)
    ties = sum(1 for d in deltas if abs(d) < 1e-6)
    losses = n_trials - wins - ties
    colors = ['#4CAF50', '#FFC107', '#F44336']
    ax6.pie([wins, ties, losses], labels=['Evo Wins', 'Ties', 'Evo Losses'],
           colors=colors, autopct='%1.1f%%', startangle=90)
    ax6.set_title('Head-to-Head', fontsize=12, fontweight='bold')
    
    # 7. Summary table
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.axis('off')
    summary_data = [
        ['Metric', 'Evolution', 'Fixed'],
        ['Mean Cost', f"{np.mean(evolved_costs):.2f}", f"{np.mean(fixed_costs):.2f}"],
        ['Std Cost', f"{np.std(evolved_costs):.2f}", f"{np.std(fixed_costs):.2f}"],
        ['Best Cost', f"{np.min(evolved_costs):.2f}", f"{np.min(fixed_costs):.2f}"],
        ['Worst Cost', f"{np.max(evolved_costs):.2f}", f"{np.max(fixed_costs):.2f}"],
        ['Win Rate', f"{100*wins/n_trials:.1f}%", f"{100*losses/n_trials:.1f}%"],
    ]
    table = ax7.table(cellText=summary_data, cellLoc='center', loc='center',
                     colWidths=[0.4, 0.3, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    for i in range(3):
        table[(0, i)].set_facecolor('#E0E0E0')
        table[(0, i)].set_text_props(weight='bold')
    ax7.set_title('Summary Statistics', fontsize=12, fontweight='bold', pad=20)
    
    # plt.savefig('evolution_comparison.png', dpi=150, bbox_inches='tight')
    # print("\nPlots saved as 'evolution_comparison.png'")
    plt.show()

if __name__ == "__main__":
    run_evolution_comparison()