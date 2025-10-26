import networkx as nx
import numpy as np

class Ant:
    def __init__(self, start_node, num_nodes, graph: nx.Graph, seed=None):
        # Initialize ant starting node
        self.start_node = start_node  # changed from hardcoded 0 to flexible start node
        self.current_node = start_node
        self.tour = [start_node]
        self.visited = set([start_node])
        self.tour_length = 0
        self.graph: nx.Graph = graph
        self.num_nodes = num_nodes
        self.rng = np.random.default_rng(seed)  # per-ant RNG for reproducibility

    def reset(self):
        # Reset the ant state at the beginning of each iteration
        self.current_node = self.start_node
        self.tour = [self.start_node]
        self.visited = set([self.start_node])
        self.tour_length = 0

    def choose_next_node(self, pheromone_matrix, heuristic_matrix, alpha, beta, intent_matrix=None, upsilon=None):
        """
        Select the next node based on pheromone and heuristic matrices.
        Implements MMAS probability formula with per-ant RNG and roulette wheel selection.
        """

        # Get unvisited neighbors of the current node
        neighbors = [n for n in self.graph.neighbors(self.current_node) if n not in self.visited]

        # Safety check: if no neighbors left, return False
        if not neighbors:
            return False  # prevents crash if ant is stuck

        # Compute transition probabilities for neighbors
        probabilities = {}  # local dict instead of instance variable
        for neighbor in neighbors:
            tau = pheromone_matrix[self.current_node][neighbor]  # pheromone value
            eta = heuristic_matrix[self.current_node][neighbor]   # heuristic (1/cost)
            if intent_matrix is not None and upsilon is not None:
                pi = intent_matrix[self.current_node][neighbor]
                probabilities[neighbor] = (tau ** alpha) * (eta ** beta) * (pi ** upsilon)
            else:
                probabilities[neighbor] = (tau ** alpha) * (eta ** beta)  # MMAS formula

        # Normalize probabilities to sum to 1
        probs = np.array(list(probabilities.values()))
        total = probs.sum()
        if total > 0:
            probs /= total
        else:
            # Fallback if all probabilities are zero
            probs = np.ones(len(neighbors)) / len(neighbors)

        # Roulette wheel (cumulative probability) selection using per-ant RNG
        cum_probs = np.cumsum(probs)
        r = self.rng.random()  # random float [0,1)
        selected_index = np.searchsorted(cum_probs, r)
        next_node = list(probabilities.keys())[selected_index]

        # Update ant state with chosen node
        self.current_node = next_node
        self.tour.append(next_node)
        self.visited.add(next_node)
        return True
