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
T_CHANGE = 80               # iteration where weight modification happens
P_VALUES = [5, 10, 20]      # percent improvements to target
FOUND_ALLOWANCE = 1.12      # route ≤ 112% of new optimal
SEEDS = list(range(1, NUM_RUNS + 1))
EXPECTED_FACTOR = 1.75      # expected allowed iterations = round(1.75 * P)
MIN_ITER = 1

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GRAPH_PATH = os.path.join(BASE_DIR, "..", "tsplib_graphs", "eil51.pkl")

# Known optimal tour for eil51 (from TSPLIB) - 0-indexed
KNOWN_OPTIMAL_TOUR = [
    0, 21, 7, 25, 30, 27, 2, 35, 34, 19, 1, 28, 20, 15, 49, 33, 29, 8, 48, 9,
    38, 32, 44, 14, 43, 41, 39, 18, 40, 12, 24, 13, 23, 42, 6, 22, 47, 5, 26,
    50, 45, 11, 46, 17, 3, 16, 36, 4, 37, 10, 31
]

KNOWN_OPTIMAL_COST = 426


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
            # Use ROUNDED Euclidean distance (TSPLIB standard)
            w = float(np.round(np.linalg.norm(coords[i] - coords[j])))
            G.add_edge(i, j, weight=w)

    return G, coords


# ============================================================
# COMPUTE COST OF A TOUR
# ============================================================
def compute_tour_cost(tour, cost_matrix):
    """Calculate total cost of a tour."""
    n = len(tour)
    return sum(cost_matrix[tour[i]][tour[(i + 1) % n]] for i in range(n))


# ============================================================
# CREATE EDGE MODIFICATION FOR ADAPTIVITY TEST
# ============================================================
def create_edge_modification(cost_matrix, target_P):
    """
    Simple strategy:
    1. Find an edge NOT in the optimal tour
    2. Reduce it drastically, see what happens to optimal
    3. Adjust iteratively until we get close to target_P% improvement
    
    Returns: (u, v, new_weight, new_optimal_tour, new_optimal_cost, actual_P)
    """
    n = len(cost_matrix)
    original_tour = KNOWN_OPTIMAL_TOUR.copy()
    
    # Verify original optimal cost
    original_cost = compute_tour_cost(original_tour, cost_matrix)
    print(f"  Original optimal cost: {original_cost:.2f} (expected: {KNOWN_OPTIMAL_COST})")
    assert abs(original_cost - KNOWN_OPTIMAL_COST) < 1.0, "Original tour cost doesn't match!"
    
    target_new_optimal = original_cost * (1 - target_P / 100)
    print(f"  Target P: {target_P}% → Target new optimal: {target_new_optimal:.2f}")
    
    # Build set of edges in original optimal tour
    original_edges = set()
    for idx in range(len(original_tour)):
        u, v = original_tour[idx], original_tour[(idx + 1) % len(original_tour)]
        original_edges.add((min(u, v), max(u, v)))
    
    # Find an edge NOT in the optimal tour with reasonable weight
    candidate_edges = []
    for i in range(n):
        for j in range(i+1, n):
            edge_key = (min(i, j), max(i, j))
            if edge_key not in original_edges:
                weight = cost_matrix[i][j]
                if weight > 10:  # Pick edges with some weight to modify
                    candidate_edges.append((i, j, weight))
    
    # Sort by weight (higher weight = more impact when reduced)
    candidate_edges.sort(key=lambda x: x[2], reverse=True)
    
    # Try different edges and reductions until we get close to target_P
    best_result = None
    best_diff = float('inf')
    
    for u, v, original_weight in candidate_edges[:20]:  # Try top 20 candidates
        # Try different reduction amounts
        for reduction_pct in [0.99, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20]:
            new_weight = max(0.1, original_weight * (1 - reduction_pct))
            
            # Apply modification
            modified_cost_matrix = cost_matrix.copy()
            modified_cost_matrix[u][v] = new_weight
            modified_cost_matrix[v][u] = new_weight
            
            # Find best tour with modified costs by trying ALL 2-opt swaps exhaustively
            best_tour = original_tour
            best_cost = compute_tour_cost(original_tour, modified_cost_matrix)
            
            # Try ALL 2-opt swaps exhaustively
            for i in range(1, n-1):
                for k in range(i+1, n):
                    swap_tour = original_tour[:i] + original_tour[i:k+1][::-1] + original_tour[k+1:]
                    swap_cost = compute_tour_cost(swap_tour, modified_cost_matrix)
                    if swap_cost < best_cost:
                        best_cost = swap_cost
                        best_tour = swap_tour
            
            # Check if this gets us close to target
            actual_P = ((original_cost - best_cost) / original_cost) * 100
            diff = abs(actual_P - target_P)
            
            if diff < best_diff:
                best_diff = diff
                best_result = {
                    'edge': (u, v),
                    'original_weight': original_weight,
                    'new_weight': new_weight,
                    'new_tour': best_tour,
                    'new_cost': best_cost,
                    'actual_P': actual_P
                }
                
                print(f"    Trying edge ({u},{v}), reduction {reduction_pct*100:.0f}% → P={actual_P:.2f}%, diff={diff:.2f}")
                
                # If we're close enough, stop searching
                if diff < 0.5:
                    break
        
        if best_result and abs(best_result['actual_P'] - target_P) < 0.5:
            break
    
    if best_result is None:
        print("  ERROR: Could not find suitable modification!")
        return None, None, None, None, None, None
    
    u, v = best_result['edge']
    new_weight = best_result['new_weight']
    original_weight = best_result['original_weight']
    new_tour = best_result['new_tour']
    new_cost = best_result['new_cost']
    actual_P = best_result['actual_P']
    
    # Verify
    modified_cost_matrix = cost_matrix.copy()
    modified_cost_matrix[u][v] = new_weight
    modified_cost_matrix[v][u] = new_weight
    
    # Check BOTH the original tour and new tour with modified costs
    original_cost_with_new_weights = compute_tour_cost(original_tour, modified_cost_matrix)
    new_tour_with_new_weights = compute_tour_cost(new_tour, modified_cost_matrix)
    
    # The new optimal is whichever is ACTUALLY better after modification
    if new_tour_with_new_weights < original_cost_with_new_weights:
        final_new_optimal = new_tour_with_new_weights
        final_new_tour = new_tour
        is_new_better = True
    else:
        # Modification didn't help - optimal stays the same
        final_new_optimal = original_cost_with_new_weights
        final_new_tour = original_tour
        is_new_better = False
    
    # Calculate actual P based on original cost vs final new optimal
    actual_P = ((original_cost - final_new_optimal) / original_cost) * 100
    
    print(f"\n  === MODIFICATION SUMMARY ===")
    print(f"  Edge to modify: ({u}, {v})")
    print(f"  Original weight: {original_weight:.2f} → New weight: {new_weight:.2f}")
    print(f"  Reduction: {original_weight - new_weight:.2f} ({(1 - new_weight/original_weight)*100:.1f}%)")
    print(f"  Original tour cost (with new weights): {original_cost_with_new_weights:.2f}")
    print(f"  Alternative tour cost (with new weights): {new_tour_with_new_weights:.2f}")
    print(f"  Did modification create new optimal? {is_new_better}")
    print(f"  Final new optimal cost: {final_new_optimal:.2f}")
    print(f"  Actual P achieved: {actual_P:.2f}%")
    print(f"  Target was: {target_P}%")
    print(f"  Difference: {abs(actual_P - target_P):.2f}%")
    print(f"  ===========================\n")
    
    return u, v, new_weight, final_new_tour, final_new_optimal, actual_P


# ============================================================
# RUN A SINGLE ADAPTIVITY TEST
# ============================================================
def run_single(G, coords, seed, target_P):
    n = len(coords)

    # Build initial cost matrix with ROUNDED distances
    cost_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                cost_matrix[i][j] = float(np.round(np.linalg.norm(coords[i] - coords[j])))

    required_nodes = list(range(n))
    index_map = {i: i for i in required_nodes}

    # Determine edge to modify and new optimal
    u, v, new_weight, new_optimal_tour, new_optimal_cost, actual_P = create_edge_modification(
        cost_matrix.copy(), target_P
    )

    # Initialize ACO
    aco = MaxMinACO(
        cost_matrix,
        start_node=0,
        reducedGraph=G,
        completeGraph=G,
        shortest_paths={},
        required_nodes=required_nodes,
        index_map=index_map,
        seed=seed,
    )

    global_best_costs = []  # Track global best cost per iteration
    global_best = np.inf
    recorded_t_found = None

    # ========================================================
    # RUN ITERATIONS
    # ========================================================
    for it in range(I_MAX):
        aco.run(iterations=1, n=it)

        # Find best ant cost in this iteration
        iter_best = np.inf
        best_ant_tour = None
        for ant in aco.ants:
            if ant is None or len(ant.tour) != n:
                continue
            ant_cost = compute_tour_cost(ant.tour, cost_matrix)
            if ant_cost < iter_best:
                iter_best = ant_cost
                best_ant_tour = ant.tour
        
        # Update global best
        if iter_best < global_best:
            global_best = iter_best
            # Debug: print when we find improvement
            if it >= T_CHANGE:
                print(f"    [Seed {seed}, Iter {it}] New best found: {global_best:.2f} (threshold: {FOUND_ALLOWANCE * new_optimal_cost:.2f})")
        
        global_best_costs.append(global_best)

        # ====================================================
        # APPLY WEIGHT CHANGE AT t_change
        # ====================================================
        if it == T_CHANGE:
            # Update cost matrix
            cost_matrix[u][v] = new_weight
            cost_matrix[v][u] = new_weight
            
            # Update graph
            if G.has_edge(u, v):
                G[u][v]["weight"] = new_weight
            
            # Update ACO's internal cost matrix
            aco.cost_matrix[u][v] = new_weight
            aco.cost_matrix[v][u] = new_weight
            
            # Update ACO's heuristic matrix (η = 1/cost)
            if new_weight > 0:
                aco.heuristic[u][v] = 1.0 / new_weight
                aco.heuristic[v][u] = 1.0 / new_weight
            else:
                aco.heuristic[u][v] = 1e10
                aco.heuristic[v][u] = 1e10

        # ====================================================
        # CHECK FOR ADAPTATION AFTER t_change
        # ====================================================
        if it >= T_CHANGE and recorded_t_found is None:
            threshold = FOUND_ALLOWANCE * new_optimal_cost
            if global_best <= threshold:
                recorded_t_found = it

    # If never found, mark as infinity
    if recorded_t_found is None:
        recorded_t_found = np.inf

    iterations_to_adapt = recorded_t_found - T_CHANGE if recorded_t_found != np.inf else np.inf
    
    return global_best_costs, iterations_to_adapt, new_optimal_cost, actual_P


# ============================================================
# MAIN BENCHMARK
# ============================================================
def run_benchmark():
    print(f"Loading {GRAPH_PATH}...")
    G, coords = load_graph(GRAPH_PATH)
    print(f"Loaded graph with {len(coords)} nodes.")
    print(f"Known optimal tour cost: {KNOWN_OPTIMAL_COST}")

    all_results = {}  # P → results

    for target_P in P_VALUES:
        print(f"\n===============================")
        print(f"   Testing P = {target_P}% improvement")
        print(f"===============================")

        adapt_times = []
        all_cost_curves = []
        new_optimal = None
        actual_P = None

        for seed in SEEDS:
            costs, adapt_time, new_opt, act_P = run_single(G, coords, seed, target_P)
            adapt_times.append(adapt_time)
            all_cost_curves.append(costs)
            if new_optimal is None:
                new_optimal = new_opt
                actual_P = act_P
            
            status = f"{adapt_time}" if adapt_time != np.inf else "NOT FOUND"
            print(f"  Seed {seed:2d} → adapt time = {status}")

        # Use actual P for allowed iterations calculation
        allowed = max(MIN_ITER, int(round(EXPECTED_FACTOR * actual_P)))
        passes = sum(t <= allowed for t in adapt_times if t != np.inf)
        prop = passes / NUM_RUNS

        print("\n------------------------------------")
        print(f" Original optimal: {KNOWN_OPTIMAL_COST}")
        print(f" New optimal: {new_optimal:.1f}")
        print(f" Actual P achieved: {actual_P:.2f}%")
        print(f" Allowed iterations: {allowed}")
        print(f" Passes: {passes}/{NUM_RUNS} ({prop*100:.1f}%)")
        print(f" Pass (≥90%): {'PASS' if prop >= 0.90 else 'FAIL'}")
        print("------------------------------------")

        all_results[target_P] = (all_cost_curves, adapt_times, prop, new_optimal, actual_P)

        visualize(target_P, actual_P, all_cost_curves, adapt_times, prop, allowed, new_optimal)

    return all_results


# ============================================================
# VISUALIZATION
# ============================================================
def visualize(target_P, actual_P, all_curves, adapt_times, prop, allowed, new_optimal):
    num_runs = len(all_curves)
    max_len = max(len(c) for c in all_curves)

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=[
            f"Global Best Cost per Iteration (Target P={target_P}%, Actual P={actual_P:.1f}%)",
            f"Adaptation Time Distribution (Pass Rate={prop*100:.1f}%)"
        ],
        vertical_spacing=0.13,
        row_heights=[0.6, 0.4]
    )

    # ---------- Cost curve plot ----------
    # Individual runs (light)
    for i, curve in enumerate(all_curves):
        fig.add_trace(
            go.Scatter(
                y=curve, 
                x=list(range(len(curve))),
                mode='lines',
                opacity=0.25,
                line=dict(color="lightblue", width=1),
                showlegend=(i == 0),
                name="Individual runs" if i == 0 else None,
            ),
            row=1, col=1
        )

    # Mean curve
    arr = np.full((num_runs, max_len), np.nan)
    for i, L in enumerate(all_curves):
        arr[i, :len(L)] = L
    
    avg = np.nanmean(arr, axis=0)
    fig.add_trace(
        go.Scatter(
            y=avg, 
            x=list(range(len(avg))),
            line=dict(color="darkblue", width=3),
            name="Mean best cost",
        ),
        row=1, col=1
    )

    # Add vertical line at t_change
    fig.add_vline(
        x=T_CHANGE, 
        line_dash="dash", 
        line_color="red",
        annotation_text=f"Weight change (t={T_CHANGE})",
        annotation_position="top",
        row=1, col=1
    )

    # Add horizontal line for new optimal
    fig.add_hline(
        y=new_optimal,
        line_dash="dot",
        line_color="green",
        annotation_text=f"New optimal = {new_optimal:.0f}",
        annotation_position="right",
        row=1, col=1
    )

    # Add horizontal line for threshold (1.12 × new_optimal)
    threshold = FOUND_ALLOWANCE * new_optimal
    fig.add_hline(
        y=threshold,
        line_dash="dot",
        line_color="orange",
        annotation_text=f"Threshold (1.12×) = {threshold:.0f}",
        annotation_position="right",
        row=1, col=1
    )

    fig.update_yaxes(title_text="Best Cost Found", row=1, col=1)
    fig.update_xaxes(title_text="Iteration", row=1, col=1)

    # ---------- Histogram ----------
    finite = [t for t in adapt_times if t != np.inf]
    
    if finite:
        fig.add_trace(
            go.Histogram(
                x=finite,
                nbinsx=min(20, max(5, len(finite))),
                marker_color="steelblue",
                name="Adaptation times",
            ),
            row=2, col=1
        )

        # Add vertical line for allowed threshold
        fig.add_vline(
            x=allowed,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Allowed = {allowed}",
            annotation_position="top",
            row=2, col=1
        )

    fig.update_xaxes(title_text="Iterations to Adapt", row=2, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=1)

    # Overall layout
    fig.update_layout(
        height=900,
        width=1100,
        title_text=f"Edge Weight Adaptivity Test — Target P={target_P}%, Actual={actual_P:.1f}%, Allowed ≤ {allowed} iterations",
        template="plotly_white",
        showlegend=True
    )

    fig.show()


if __name__ == "__main__":
    run_benchmark()