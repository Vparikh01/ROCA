import os
import pickle
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from src.aco.engine import MaxMinACO

# Load a TSPLIB instance (adjust path as needed)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
pkl_path = os.path.join(BASE_DIR, "..", "tsplib_graphs", "bier127.pkl")

with open(pkl_path, "rb") as f:
    data = pickle.load(f)

coords = np.asarray(data["coordinates"])
n = len(coords)

# Build graph
G = nx.Graph()
for i in range(n):
    for j in range(i + 1, n):
        dist = float(np.linalg.norm(coords[i] - coords[j]))
        G.add_edge(i, j, weight=dist)

# Build cost matrix
required_nodes = list(G.nodes())
cost_matrix = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        if i != j:
            cost_matrix[i][j] = float(np.linalg.norm(coords[i] - coords[j]))

# Create index map
index_map = {node: i for i, node in enumerate(required_nodes)}

# Run MaxMin ACO
aco = MaxMinACO(
    cost_matrix,
    start_node=0,
    reducedGraph=G,
    completeGraph=G,
    shortest_paths={},
    required_nodes=required_nodes,
    index_map=index_map,
    seed=42
)

# Optimize
aco.run(iterations=100, n=0)

print(f"Best tour length: {aco.best_length:.2f}")

# Visualize the tour
tour = aco.best_tour
tour_coords = coords[tour]

plt.figure(figsize=(10, 8))
# Plot edges
plt.plot(tour_coords[:, 0], tour_coords[:, 1], 'b-', linewidth=1.5, alpha=0.6)
plt.plot([tour_coords[-1, 0], tour_coords[0, 0]], 
         [tour_coords[-1, 1], tour_coords[0, 1]], 'b-', linewidth=1.5, alpha=0.6)
# Plot nodes
plt.scatter(tour_coords[:, 0], tour_coords[:, 1], c='blue', s=50, zorder=3)
# Highlight start node
plt.scatter(tour_coords[0, 0], tour_coords[0, 1], c='red', s=100, marker='*', 
            zorder=4, label='Start/End')
plt.title(f'Best Tour - Length: {aco.best_length:.2f}')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
