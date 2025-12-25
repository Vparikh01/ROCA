import numpy as np
import networkx as nx
from src.aco.config import load_config
from src.rl.edgeQ import EdgeQ

cfg = load_config()


class Ant:
    def __init__(self, start_node, num_nodes, Qlearner, graph: nx.Graph, seed=None, lambda_q=1.0, lambda_start=0.0, lambda_end=1.25, iterations=200):
        self.start_node = start_node
        self.current_node = start_node
        self.tour = [start_node]
        self.visited = set([start_node])
        self.tour_length = 0
        self.graph = graph
        self.num_nodes = num_nodes
        self.alpha = cfg["alpha"]
        self.beta = cfg["beta"]
        self.mew_alpha = 0.0   # log-space mean offset
        self.mew_beta  = 0.0
        self.rng = np.random.default_rng(seed)
        self.Qlearner = Qlearner
        self.lambda_q = lambda_q
        self.current_iteration = 0
        self.max_iterations = iterations
        self.lambda_start = lambda_start
        self.lambda_end = lambda_end
        self.tour_sum = 0.0

        # Precompute neighbors as NumPy arrays
        self.neighbor_indices = [
            np.array(list(self.graph.neighbors(n)), dtype=int)
            for n in range(self.num_nodes)
        ]
    def update_lambda(self):
        self.current_iteration += 1
        frac = min(self.current_iteration / self.max_iterations, 1.0)
        self.lambda_q = self.lambda_start + frac * (self.lambda_end - self.lambda_start)

    def reset(self):
        self.tour_sum += self.tour_length
        self.current_node = self.start_node
        self.tour = [self.start_node]
        self.visited = set([self.start_node])
        self.tour_length = 0
        
    def reset_tour_sum(self):
        self.tour_sum = 0.0

    def choose_next_node(self, pheromone_matrix, heuristic_matrix, intent_matrix=None, upsilon=None):
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

        if self.Qlearner is not None:
            q_costs = self.Qlearner.Q[self.current_node, neighbors]
            #(lower cost = higher preference)
            q_pref = 1.0 / (q_costs + 1e-8)
            q_sum = q_pref.sum()
            if q_sum > 0:
                q_probs = q_pref / q_sum
            else:
                q_probs = np.ones_like(q_pref) / len(q_pref)
        else:
            q_probs = np.ones_like(neighbors)

        # Intent term
        if intent_matrix is not None and upsilon is not None:
            pi = intent_matrix[self.current_node, neighbors]
        else:
            pi = np.ones_like(neighbors)

        # Combine factors
        probs = (tau ** self.alpha) * (eta ** self.beta) * (pi ** (upsilon if upsilon else 1.0)) * (q_probs ** self.lambda_q)

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