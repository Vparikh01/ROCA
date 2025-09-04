# ----------------- mmas_test.py -----------------
import os
import networkx as nx
import numpy as np
import pickle
import random
from itertools import permutations
from src.aco.visualizer import draw_aco_graph
from src.aco.engine import MaxMinACO
from src.aco.astar import build_metric_closure

# ------------------ Paths ------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EURO_AIRPORT_PKL = os.path.join(BASE_DIR, "..", "datasets", "europe_airports_graph.pkl")

def load_graph(pkl_path):
    with open(pkl_path, "rb") as f:
        G = pickle.load(f)
    if G.is_directed():
        G = G.to_undirected()
    return G

# ------------------ Load and preprocess graph ------------------
G = load_graph(EURO_AIRPORT_PKL)

# Filter out very low-degree nodes
node_degrees = dict(G.degree())
significant_nodes = [n for n, deg in node_degrees.items() if deg > 1]
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
required_nodes = random.sample(list(pos.keys()), 19)
start_node = required_nodes[0]  # first in the random list as start

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
aco = MaxMinACO(cost_matrix, start_node=0, reducedGraph=G_indexed)
NUM_ITERATIONS = 200
iteration_paths = []

for _ in range(NUM_ITERATIONS):
    aco.run(iterations=1)
    best_iter_nodes = [required_nodes[i] for i in aco.best_iter_tour]
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
        G_indexed, cycle=True, weight="weight", method=nx.approximation.christofides
    )
    opt_tour_idx = approx_tour
    opt_cost = sum(cost_matrix[approx_tour[i]][approx_tour[(i+1)%num_nodes]] for i in range(num_nodes))

# Map optimal tour back to node labels
optimal_nodes = [required_nodes[i] for i in opt_tour_idx]
optimal_full_path = []
for k in range(len(optimal_nodes)):
    u = optimal_nodes[k]
    v = optimal_nodes[(k+1) % len(optimal_nodes)]
    sp = nx.shortest_path(G, u, v)
    optimal_full_path.extend(sp[:-1])
optimal_full_path.append(start_node)

# ----------------- Output -----------------
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
