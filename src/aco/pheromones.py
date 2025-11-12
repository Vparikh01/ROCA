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
        delta = 1 / tour_length
        tour_arr = np.array(tour, dtype=int)
        edges_a = tour_arr[:-1]
        edges_b = tour_arr[1:]
        self.matrix[edges_a, edges_b] += delta
        self.matrix[edges_b, edges_a] += delta  # symmetric
        np.clip(self.matrix, self.tau_min, self.tau_max, out=self.matrix)
    def reset(self):
        self.matrix.fill(self.tau_max)
