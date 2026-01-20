import numpy as np
from collections import deque

class EdgeQ:
    def __init__(self, n_nodes, avgCost, alpha=0.1, gamma=0.65, n_step=0):
        """
        n_nodes : number of cities
        alpha   : TD learning rate
        gamma   : discount factor
        n_step  : number of steps for TD(n)
        """
        self.n = n_nodes
        self.alpha = alpha
        self.gamma = gamma
        self.n_step = n_step

        # Q-table: expected remaining cost after taking edge i -> j
        self.Q = np.ones((n_nodes, n_nodes)) * avgCost
        self.buffer = deque()

    def observe(self, i, j, cost):
        self.buffer.append((i, j, cost))

        # Perform TD(n) update if buffer is full
        if len(self.buffer) > self.n_step:
            self._td_update()

    def _td_update(self):
        """
        Perform one TD(n) update from the buffer.
        """
        G = 0.0
        for t, (_, _, r) in enumerate(self.buffer):
            G += (self.gamma ** t) * r

        i0, j0, _ = self.buffer[0]

        if len(self.buffer) > self.n_step:  # When buffer has 4+
            i_next, j_next, _ = self.buffer[self.n_step]  # Gets buffer[3]
            G += (self.gamma ** self.n_step) * self.Q[i_next][j_next]

        # TD update
        self.Q[i0][j0] += self.alpha * (G - self.Q[i0][j0])

        # Remove oldest transition
        self.buffer.popleft()

    def flush(self):
        while self.buffer:
            G = 0.0
            for t, (_, _, r) in enumerate(self.buffer):
                G += (self.gamma ** t) * r

            i0, j0, _ = self.buffer[0]
            self.Q[i0][j0] += self.alpha * (G - self.Q[i0][j0])
            self.buffer.popleft()

    def get(self, i, j):
        return self.Q[i][j]
