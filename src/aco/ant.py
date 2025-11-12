import numpy as np
import networkx as nx

class Ant:
    def __init__(self, start_node, num_nodes, graph: nx.Graph, seed=None):
        self.start_node = start_node
        self.current_node = start_node
        self.tour = [start_node]
        self.visited = set([start_node])
        self.tour_length = 0
        self.graph = graph
        self.num_nodes = num_nodes
        self.rng = np.random.default_rng(seed)

        # Precompute neighbors as NumPy arrays
        self.neighbor_indices = [
            np.array(list(self.graph.neighbors(n)), dtype=int)
            for n in range(self.num_nodes)
        ]

    def reset(self):
        self.current_node = self.start_node
        self.tour = [self.start_node]
        self.visited = set([self.start_node])
        self.tour_length = 0

    def choose_next_node(self, pheromone_matrix, heuristic_matrix, alpha, beta, intent_matrix=None, upsilon=None):
        # Use precomputed neighbors
        all_neighbors = self.neighbor_indices[self.current_node]

        # Boolean mask: neighbors that are not visited
        mask = np.isin(all_neighbors, list(self.visited), invert=True)
        neighbors = all_neighbors[mask]

        if len(neighbors) == 0:
            return False

        # Vectorized access
        tau = pheromone_matrix[self.current_node, neighbors]
        eta = heuristic_matrix[self.current_node, neighbors]

        if intent_matrix is not None and upsilon is not None:
            pi = intent_matrix[self.current_node, neighbors]
            probs = (tau ** alpha) * (eta ** beta) * (pi ** upsilon)
        else:
            probs = (tau ** alpha) * (eta ** beta)

        # Normalize safely
        total = probs.sum()
        if total > 0:
            probs /= total
        else:
            probs = np.ones_like(probs) / len(probs)

        # Roulette wheel selection
        r = self.rng.random()
        cum_probs = np.cumsum(probs)
        selected_index = np.searchsorted(cum_probs, r)
        next_node = neighbors[selected_index]

        # Update ant state
        self.current_node = next_node
        self.tour.append(next_node)
        self.visited.add(next_node)
        return True