# ----------------- main.py -----------------
import networkx as nx
import numpy as np
from itertools import permutations
from src.aco.visualizer import draw_aco_graph
from src.aco.engine import MaxMinACO

GRID_SIZE = 10
G = nx.grid_2d_graph(GRID_SIZE, GRID_SIZE)
pos = {n: (n[0], n[1]) for n in G.nodes()}

# ----------------- Required Nodes -----------------
required_nodes = [(0,0), (0,9), (9,0), (9,9), (4,4), (2,5), (6,3), (7,2), (5,8)]
start_node = required_nodes[0]

# ----------------- Randomized Cost Matrix (complete subgraph) -----------------
num_nodes = len(required_nodes)
cost_matrix = np.zeros((num_nodes, num_nodes))
edge_weights = {}  # store for plotting

for i in range(num_nodes):
    for j in range(num_nodes):
        if i != j:
            sp_len = nx.shortest_path_length(G, required_nodes[i], required_nodes[j])
            # Add small random variation
            weight = sp_len + np.random.uniform(0.1, 2.0)
            cost_matrix[i][j] = weight
            edge_weights[(i,j)] = weight

# ----------------- Reduced Graph -----------------
G_indexed = nx.Graph()
for i in range(num_nodes):
    G_indexed.add_node(i)
for i in range(num_nodes):
    for j in range(num_nodes):
        if i != j:
            G_indexed.add_edge(i, j, weight=cost_matrix[i][j])

# ----------------- Run ACO -----------------
aco = MaxMinACO(cost_matrix, start_node=0, reducedGraph=G_indexed)
NUM_ITERATIONS = 20
iteration_paths = []

for it in range(NUM_ITERATIONS):
    aco.run(iterations=1)
    best_iter_nodes = [required_nodes[i] for i in aco.best_iter_tour]
    # Expand full path
    full_path = []
    for k in range(len(best_iter_nodes)):
        u = best_iter_nodes[k]
        v = best_iter_nodes[(k+1) % len(best_iter_nodes)]
        sp = nx.shortest_path(G, u, v)
        full_path.extend(sp[:-1])
    full_path.append(start_node)
    iteration_paths.append(full_path)

# Final best path
final_best_nodes = [required_nodes[i] for i in aco.best_tour]
final_full_path = []
for k in range(len(final_best_nodes)):
    u = final_best_nodes[k]
    v = final_best_nodes[(k+1) % len(final_best_nodes)]
    sp = nx.shortest_path(G, u, v)
    final_full_path.extend(sp[:-1])
final_full_path.append(start_node)

# ----------------- Compute True Optimal -----------------
def compute_optimal_tour(required_nodes, cost_matrix):
    n = len(required_nodes)
    best_cost = float('inf')
    best_tour = None
    for perm in permutations(range(1, n)):
        tour = [0] + list(perm)
        tour_cost = sum(cost_matrix[tour[i]][tour[i+1]] for i in range(n-1))
        tour_cost += cost_matrix[tour[-1]][0]
        if tour_cost < best_cost:
            best_cost = tour_cost
            best_tour = tour
    return best_tour, best_cost

opt_tour_idx, opt_cost = compute_optimal_tour(required_nodes, cost_matrix)
optimal_nodes = [required_nodes[i] for i in opt_tour_idx]
optimal_full_path = []
for k in range(len(optimal_nodes)):
    u = optimal_nodes[k]
    v = optimal_nodes[(k+1) % len(optimal_nodes)]
    sp = nx.shortest_path(G, u, v)
    optimal_full_path.extend(sp[:-1])
optimal_full_path.append(start_node)

print(f"ACO Best Length: {sum(cost_matrix[aco.best_tour[i]][aco.best_tour[(i+1)%num_nodes]] for i in range(num_nodes)):.2f}")
print(f"Optimal Tour Length: {opt_cost:.2f}")

# ----------------- Visualize -----------------
draw_aco_graph(
    G,
    pos,
    iteration_paths,
    final_best_path=final_full_path,
    required_nodes=required_nodes,
    edge_weights=edge_weights,
    optimal_path=optimal_full_path
)
