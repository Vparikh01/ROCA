import numpy as np
from src.aco.config import load_config
cfg = load_config()

class PheromoneMatrix:
    def __init__(self, num_nodes, rho, expected_length):
        self.num_nodes = num_nodes
        self.tau_max = min(cfg["tau"]["max"] / expected_length, 1e3)
        self.tau_min = max(cfg["tau"]["min"] * self.tau_max, 1e-6)
        self.rho = rho  # evaporation rate
        self.matrix = np.full((num_nodes, num_nodes), self.tau_max)  # init to tau_max
    
    def evaporate(self):
        self.matrix *= (1 - self.rho)
        np.clip(self.matrix, self.tau_min, self.tau_max, out=self.matrix)

    def deposit(self, tour, tour_length):
        """
        Deposit pheromone along tour: only global-best ant in MMAS
        Δτ = 1 / tour_length
        """
        delta = 1 / tour_length
        for i in range(len(tour) - 1):
            a, b = tour[i], tour[i+1]
            self.matrix[a][b] += delta
            self.matrix[b][a] += delta  # symmetric
        np.clip(self.matrix, self.tau_min, self.tau_max, out=self.matrix)
    def reset(self):
        self.matrix.fill(self.tau_max)
