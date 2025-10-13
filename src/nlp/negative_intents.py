import numpy as np
from src.aco.config import load_config

class NegativeIntentMatrix:
    """
    Tracks negative intent influence between nodes for ACO paths.
    Initialized to 1.0 (neutral), values < 1 indicate avoidance.
    """
    def __init__(self, num_nodes):
        self.num_nodes = num_nodes
        self.matrix = np.ones((num_nodes, num_nodes), dtype=float)

    def apply_negative_nodes(self, node_indices, modifiers):
        """
        Apply negative influence to specific nodes.
        node_indices: list of node indices to update
        modifiers: list of same length with values < 1 to scale influence
        """
        for idx, modifier in zip(node_indices, modifiers):
            self.matrix[idx, :] *= modifier
            self.matrix[:, idx] *= modifier

    def reset(self):
        """Reset all values to neutral (1.0)."""
        self.matrix.fill(1.0)
    
    def get_matrix(self):
        return self.matrix.copy()
