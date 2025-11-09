import os
import networkx as nx
import numpy as np
import pickle
import random
from itertools import permutations
from src.aco.visualizer import draw_aco_graph
from src.aco.engine import MaxMinACO
from src.aco.astar import build_metric_closure
from src.nlp.matrix_visualizer import visualize_intent_matrix

# ------------------ Paths ------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PALO_ALTO_PKL = os.path.join(BASE_DIR, "..", "datasets", "sanjose_graph.pkl")

def load_graph(pkl_path):
    with open(pkl_path, "rb") as f:
        G = pickle.load(f)
    if G.is_directed():
        G = G.to_undirected()
    return G

# ------------------ Load and preprocess graph ------------------
G = load_graph(PALO_ALTO_PKL)

# Filter out very low-degree nodes
node_degrees = dict(G.degree())
significant_nodes = [n for n, deg in node_degrees.items() if deg > 0]
if len(significant_nodes) > 10:
    G = G.subgraph(significant_nodes).copy()

# Extract coordinates for visualization
pos = {}
for n, data in G.nodes(data=True):
    coords = None
    for key_x, key_y in [('x','y'), ('Lon','Lat'), ('longitude','latitude'), ('lon','lat')]:
        if key_x in data and key_y in data:
            coords = (data[key_x], data[key_y])
            break
    if coords:
        pos[n] = coords
if not pos:
    pos = nx.spring_layout(G, seed=42)

# ----------------- Required Nodes -----------------
# Pick the largest connected component
largest_cc = max(nx.connected_components(G), key=len)
subgraph_nodes = list(largest_cc)

# Sample 29 nodes from this component
required_nodes = random.sample(subgraph_nodes, 29)
start_node = required_nodes[0]

# ----------------- Cost Matrix & Reduced Graph -----------------
cost_matrix, reduced_graph, shortest_paths = build_metric_closure(G, required_nodes)

# Re-index reduced graph to 0..n-1 for ACO
index_map = {n: i for i, n in enumerate(required_nodes)}
G_indexed = nx.Graph()
for u, v, d in reduced_graph.edges(data=True):
    G_indexed.add_edge(index_map[u], index_map[v], **d)

num_nodes = len(required_nodes)
edge_weights = {
    (i, j): cost_matrix[i][j]
    for i in range(num_nodes)
    for j in range(num_nodes)
    if i != j
}

# ----------------- Run ACO -----------------
aco = MaxMinACO(cost_matrix, start_node=0, reducedGraph=G_indexed, completeGraph=G, shortest_paths=shortest_paths, required_nodes=required_nodes, index_map=index_map)
NUM_ITERATIONS = 200
iteration_paths = []

# Helper function to get the correct graph for path computation
def get_active_graph():
    """Returns the filtered graph if exclusions were applied, otherwise the original graph."""
    return getattr(aco, 'completeGraph_filtered', G)

for n in range(NUM_ITERATIONS):
    aco.run(iterations=1, n=n)
    
    # Check if we have a valid tour before processing
    if aco.best_iter_tour is not None:
        best_iter_nodes = [required_nodes[i] for i in aco.best_iter_tour]
        full_path = []
        active_graph = get_active_graph()  # Use filtered graph if available
        
        for k in range(len(best_iter_nodes)):
            u = best_iter_nodes[k]
            v = best_iter_nodes[(k+1) % len(best_iter_nodes)]
            
            # First try cached shortest_paths (which are updated by ACO)
            if (u, v) in aco.shortest_paths and aco.shortest_paths[(u, v)]:
                sp = aco.shortest_paths[(u, v)]
            else:
                # Fall back to computing on active graph
                try:
                    sp = nx.shortest_path(active_graph, u, v, weight='weight')
                except nx.NetworkXNoPath:
                    print(f"Warning: No path between {u} and {v}")
                    sp = []
            
            if not sp:
                full_path = []
                break
            full_path.extend(sp[:-1])
        
        if full_path:
            full_path.append(start_node)
            iteration_paths.append(full_path)
        else:
            # No valid path, append empty or use last valid path
            if iteration_paths:
                iteration_paths.append(iteration_paths[-1])  # Repeat last valid
            else:
                iteration_paths.append([])
    else:
        # No valid tour this iteration
        print(f"Skipping path construction for iteration {n+1} (no valid tour)")
        if iteration_paths:
            iteration_paths.append(iteration_paths[-1])  # Repeat last valid path
        else:
            iteration_paths.append([])
    
    if n == 50:
        NUM_ITERATIONS += n
        aco.apply_instruction("include libraries")
        # Update cost_matrix and G_indexed to reflect exclusions
        cost_matrix = aco.cost_matrix  # Get updated cost matrix
        
        # Rebuild G_indexed with updated costs
        G_indexed = nx.Graph()
        for i in range(num_nodes):
            for j in range(i+1, num_nodes):
                if not np.isinf(cost_matrix[i][j]):
                    G_indexed.add_edge(i, j, weight=cost_matrix[i][j])

# Final best path - use active graph
final_best_nodes = [required_nodes[i] for i in aco.best_tour]
final_full_path = []
active_graph = get_active_graph()

for k in range(len(final_best_nodes)):
    u = final_best_nodes[k]
    v = final_best_nodes[(k+1) % len(final_best_nodes)]
    
    if (u, v) in aco.shortest_paths and aco.shortest_paths[(u, v)]:
        sp = aco.shortest_paths[(u, v)]
    else:
        try:
            sp = nx.shortest_path(active_graph, u, v, weight='weight')
        except nx.NetworkXNoPath:
            print(f"Warning: No path between {u} and {v} in final path")
            sp = []
    
    if sp:
        final_full_path.extend(sp[:-1])
    else:
        break

if final_full_path:
    final_full_path.append(start_node)

# ----------------- Compute True Optimal -----------------
if num_nodes <= 9:
    # Small: brute-force
    best_cost = float('inf')
    best_tour = None
    for perm in permutations(range(1, num_nodes)):
        tour = [0] + list(perm)
        tour_cost = sum(cost_matrix[tour[i]][tour[i+1]] for i in range(num_nodes-1)) + cost_matrix[tour[-1]][0]
        if tour_cost < best_cost:
            best_cost = tour_cost
            best_tour = tour
    opt_tour_idx = best_tour
    opt_cost = best_cost
else:
    # Large: use NetworkX TSP approximation
    approx_tour = nx.approximation.traveling_salesman_problem(
        G_indexed, cycle=True, weight="weight", method=nx.approximation.christofides, 
    )
    opt_tour_idx = approx_tour
    opt_cost = sum(cost_matrix[approx_tour[i]][approx_tour[(i+1)%num_nodes]] for i in range(num_nodes))

# Map optimal tour back to node labels - use active graph
optimal_nodes = [required_nodes[i] for i in opt_tour_idx]
optimal_full_path = []

for k in range(len(optimal_nodes)):
    u = optimal_nodes[k]
    v = optimal_nodes[(k+1) % len(optimal_nodes)]
    
    if (u, v) in aco.shortest_paths and aco.shortest_paths[(u, v)]:
        sp = aco.shortest_paths[(u, v)]
    else:
        try:
            sp = nx.shortest_path(active_graph, u, v, weight='weight')
        except nx.NetworkXNoPath:
            print(f"Warning: No path between {u} and {v} in optimal path")
            sp = []
    
    if sp:
        optimal_full_path.extend(sp[:-1])
    else:
        break

if optimal_full_path:
    optimal_full_path.append(start_node)

# ----------------- Output -----------------
print(f"ACO Best Length: {sum(cost_matrix[aco.best_tour[i]][aco.best_tour[(i+1)%num_nodes]] for i in range(num_nodes)):.2f}")
print(f"ACO Post-Instruction Best Length: {sum(cost_matrix[aco.post_instruction_best_tour[i]][aco.post_instruction_best_tour[(i+1)%num_nodes]] for i in range(num_nodes)):.2f}")
print(f"Optimal Tour Length: {opt_cost:.2f}")

# ----------------- Visualize -----------------
draw_aco_graph(
    G,
    pos,
    iteration_paths,
    final_best_path=final_full_path,
    required_nodes=required_nodes,
    edge_weights=edge_weights,
    optimal_path=optimal_full_path,
    added_nodes=list(aco.added_nodes_set) if hasattr(aco, 'added_nodes_set') else [],
    removed_nodes=list(aco.safe) if hasattr(aco, 'safe') else []
)

if aco.negative_intent is not None:
    visualize_intent_matrix(
        aco.negative_intent.matrix,
        num_nodes,
        aco.intent_type
    )
if aco.positive_intent is not None:
    visualize_intent_matrix(
        aco.positive_intent.matrix,
        num_nodes,
        aco.intent_type
    )