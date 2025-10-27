import numpy as np

class PositiveIntentMatrix:
    """
    Tracks positive intent influence between nodes for ACO paths.
    Initialized to 1e-12(>0) (neutral), values > 0 indicate attraction.
    """
    def __init__(self, num_nodes):
        self.num_nodes = num_nodes
        self.matrix = np.full((num_nodes, num_nodes), 1e-12, dtype=float)

    def apply_positive_nodes(self, node_indices, modifiers):
        """
        Apply positive influence to specific nodes.
        node_indices: list of node indices to update
        modifiers: list of same length with values > 0 to scale influence
        """
        for idx, modifier in zip(node_indices, modifiers):
            self.matrix[idx, :] *= modifier
            self.matrix[:, idx] *= modifier

    def reset(self):
        """Reset all values to neutral (>0)."""
        self.matrix = np.full((self.num_nodes, self.num_nodes), 1e-12, dtype=float)

    def get_matrix(self):
        return self.matrix.copy()
