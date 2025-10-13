import numpy as np
from src.aco.pheromones import PheromoneMatrix
from src.nlp.negative_intents import NegativeIntentMatrix
from src.aco.ant import Ant
from src.aco.config import load_config
from src.nlp.context_encoder import process_instruction
cfg = load_config()

class MaxMinACO:
    def __init__(self, cost_matrix, start_node, reducedGraph):
        self.cost_matrix = cost_matrix
        self.num_nodes = len(cost_matrix)
        self.num_ants = min(cfg["num_ants"], self.num_nodes)
        self.alpha = cfg["alpha"]
        self.beta = cfg["beta"]
        
        positive_costs = cost_matrix[cost_matrix > 0]
        expected_length = max(np.mean(positive_costs) * self.num_nodes, 1e-6)
        self.pheromones = PheromoneMatrix(self.num_nodes, cfg["rho"], expected_length)
        self.negative_intent = None  # to be set via instruction

        self.heuristic = np.zeros_like(cost_matrix, dtype=float)
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if i != j and cost_matrix[i][j] > 0:
                    self.heuristic[i][j] = 1 / cost_matrix[i][j]
                else:
                    self.heuristic[i][j] = 0  # prevents huge values
        self.best_tour = None
        self.best_length = float('inf')
        self.ants = [
            Ant(
                start_node=start_node,
                num_nodes=self.num_nodes,
                graph=reducedGraph,
                seed=(cfg.get("seed") or 0) + i  # per-ant reproducible RNG
            )
            for i in range(self.num_ants)
        ]
        np.random.seed(cfg.get("seed") or 0)

    def apply_instruction(self, instruction, model, intent_templates):
        intent_type, confidence, node_info_list = process_instruction(
            #TODO: NODE INFO
            instruction, self.node_info, model, intent_templates
        )

        # store for later use in ACO
        self.intent_type = intent_type
        self.confidence = confidence
        self.node_info_list = node_info_list  # store the full list with metadata & modifiers

        if intent_type == "avoid":
            # Initialize the matrix with the number of nodes
            self.negative_intent = NegativeIntentMatrix(self.num_nodes)

            # Apply the negative modifiers from node_info_list
            node_indices = [info["node_id"] for info in node_info_list]
            modifiers = [info["pheromone_modifier"] for info in node_info_list]
            self.negative_intent.apply_negative_nodes(node_indices, modifiers)

        print(f"Instruction applied: '{instruction}'")
        print(f"Intent: {intent_type}, Confidence: {confidence:.3f}")


    def run(self, iterations=100, n=0):
        for _ in range(iterations):
            self.best_iter_tour = None
            self.best_iter_length = float('inf')
            for ant in self.ants:
                ant.reset()
                # construct tour
                while len(ant.tour) < self.num_nodes:
                    if not ant.choose_next_node(self.pheromones.matrix, self.heuristic, self.alpha, self.beta):
                        break  # or implement a restart/backtrack mechanism
                ant.tour_length = self.calculate_tour_length(ant.tour)
                # update best
                if ant.tour_length < self.best_length:
                    self.best_tour = ant.tour.copy()
                    self.best_length = ant.tour_length
                if ant.tour_length < self.best_iter_length:
                    self.best_iter_tour = ant.tour.copy()
                    self.best_iter_length = ant.tour_length

            self.pheromones.evaporate()

            if np.random.rand() < 0.25:  # 25% iteration-best
                self.pheromones.deposit(self.best_iter_tour, self.best_iter_length)
            else:  # 75% global-best
                self.pheromones.deposit(self.best_tour, self.best_length)
            
            # Optional: stagnation check / dynamic tau adjustment could go here
            print(f"Iteration {n+1}: Best length {self.best_length}")

    def calculate_tour_length(self, tour):
        length = 0
        for i in range(len(tour)-1):
            length += self.cost_matrix[tour[i]][tour[i+1]]
        length += self.cost_matrix[tour[-1]][tour[0]]  # complete cycle
        return length
