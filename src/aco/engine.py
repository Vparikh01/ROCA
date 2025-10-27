import numpy as np
import networkx as nx
from src.aco.pheromones import PheromoneMatrix
from src.nlp.negative_intents import NegativeIntentMatrix
from src.nlp.positive_intents import PositiveIntentMatrix
from src.aco.ant import Ant
from src.aco.config import load_config
from src.nlp.context_encoder import process_instruction
from src.nlp.context_encoder import extract_node_info
from src.aco.astar import exclusion_closure_update

cfg = load_config()

class MaxMinACO:
    def __init__(self, cost_matrix, start_node, reducedGraph, completeGraph, shortest_paths, required_nodes, index_map):
        self.cost_matrix = cost_matrix
        self.start_node = start_node
        self.num_nodes = len(cost_matrix)
        self.num_ants = min(cfg["num_ants"], self.num_nodes)
        self.alpha = cfg["alpha"]
        self.beta = cfg["beta"]
        
        positive_costs = cost_matrix[cost_matrix > 0]
        expected_length = max(np.mean(positive_costs) * self.num_nodes, 1e-6)
        self.pheromones = PheromoneMatrix(self.num_nodes, cfg["rho"], expected_length)
        self.negative_intent = None  # to be set via instruction
        self.positive_intent = None  # to be set via instruction
        self.shortest_paths = shortest_paths
        unique_nodes = set().union(*self.shortest_paths.values())
        self.completeGraph = completeGraph
        subgraph = completeGraph.subgraph(unique_nodes).copy()
        self.node_info = extract_node_info(subgraph)
        self.shortest_paths = shortest_paths
        self.required_nodes = required_nodes
        self.index_map = index_map

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
                start_node=self.start_node,
                num_nodes=self.num_nodes,
                graph=reducedGraph,
                seed=(cfg.get("seed") or 0) + i  # per-ant reproducible RNG
            )
            for i in range(self.num_ants)
        ]
        np.random.seed(cfg.get("seed") or 0)

    def apply_instruction(self, instruction):

        if not self.node_info:
            print("[apply_instruction] No node_info available — skipping NLP instruction.")
            self.intent_type, self.confidence, self.significant_nodes = None, 0.0, []
            return
        self.instruction = instruction
        intent_type, confidence, significant_nodes = process_instruction(
            instruction, self.node_info
        )
        self.intent_type, self.confidence, self.significant_nodes = intent_type, confidence, significant_nodes

        if intent_type == "avoid":
            print(f"\n{'='*60}\nAPPLYING AVOID INSTRUCTION: '{instruction}'\n{'='*60}")
            self.negative_intent = NegativeIntentMatrix(self.num_nodes)

            node_modifier_map = {n["node_id"]: n["pheromone_modifier"] for n in significant_nodes}
            print(f"Significant nodes: {len(significant_nodes)} | Mod range: [{min(node_modifier_map.values()):.3f}, {max(node_modifier_map.values()):.3f}]")

            for i in range(self.num_nodes):
                for j in range(self.num_nodes):
                    if i == j: continue
                    u, v = self.required_nodes[i], self.required_nodes[j]
                    if (u, v) in self.shortest_paths:
                        path = self.shortest_paths[(u, v)]
                        mods = [node_modifier_map[n] for n in path if n in node_modifier_map]
                        self.negative_intent.matrix[i][j] = max(0.1, np.mean(mods)) if mods else 1.0

            threshold = 0.05
            excluded = [n["node_id"] for n in significant_nodes if n["pheromone_modifier"] <= threshold and n["node_id"] not in set(self.required_nodes)]
            print(f"\nExclusion threshold: {threshold} | Candidates: {len(excluded)}")

            if excluded:
                G = self.completeGraph.copy()
                safe = []
                for node in excluded:
                    G_temp = G.copy()
                    G_temp.remove_node(node)
                    if nx.is_connected(G_temp):
                        G = G_temp  # commit removal
                        safe.append(node)
                print(f"Safe exclusions (iterative): {len(safe)}/{len(excluded)}")

                if safe:
                    self.cost_matrix, self.shortest_paths = exclusion_closure_update(
                        self.completeGraph, self.required_nodes, self.cost_matrix, self.shortest_paths, safe, self.index_map
                    )
                    for i in range(self.num_nodes):
                        for j in range(self.num_nodes):
                            if i == j: self.heuristic[i][j] = 0
                            elif self.cost_matrix[i][j] > 0: self.heuristic[i][j] = 1 / self.cost_matrix[i][j]
                            else: self.heuristic[i][j] = 0
                    print("Connectivity OK → Hard exclusions applied")
                else:
                    print("No safe exclusions found → Soft penalties only")
                    print("\n--- Attempting Hard Exclusion Debug Info ---")
                    for node in excluded:
                        data = self.completeGraph.nodes[node]
                        print(f"\nNode: {node}")
                        for k, v in data.items():
                            print(f"  {k}: {v}")
                    print("--- End of Exclusion Node Attributes ---\n")
            else:
                print("No nodes to exclude")
            
            # Resynchronize best_length and tour after exclusion closure
            if self.best_tour is not None:
                new_cost = sum(
                    self.cost_matrix[self.best_tour[i]][self.best_tour[i+1]]
                    for i in range(len(self.best_tour)-1)
                ) + self.cost_matrix[self.best_tour[-1]][self.best_tour[0]]

                if abs(new_cost - self.best_length) > 1e-6:
                    print(f"[Sync] Adjusting best_length: {self.best_length:.3f} → {new_cost:.3f}")
                    self.best_length = new_cost

            print(f"Final ACO Best Length: {self.best_length:.3f}")

        else:
            print(f"\n{'='*60}\nAPPLYING PREFER INSTRUCTION: '{instruction}'\n{'='*60}")
            self.positive_intent = PositiveIntentMatrix(self.num_nodes)
            nodes_added = 0
            min_nodes_to_add = self.compute_num_nodes_to_add()
            threshold = 0.95

            current_opt_nodes = [self.required_nodes[i] for i in self.best_tour]
            current_opt_path = []
            for k in range(len(current_opt_nodes)):
                u = current_opt_nodes[k]
                v = current_opt_nodes[(k+1) % len(current_opt_nodes)]
                sp = nx.shortest_path(self.completeGraph, u, v)
                current_opt_path.extend(sp[:-1])
            current_opt_path.append(self.start_node)

            # create subset of significant_nodes whose node_id is in current_opt_path
            sig_map = {n["node_id"]: n for n in significant_nodes}
            subset = [
                sig_map[node_id]
                for node_id in current_opt_path
                if node_id in sig_map
            ]

            # nodes in current optimal path
            stage1_modifiers = {
                n["node_id"]: self.map_modifier_to_positive_value(n["pheromone_modifier"])
                for n in subset
            }
            print("Stage 1 Modifiers:")
            for node_id, modifier in stage1_modifiers.items():
                print(f"  Node {node_id}: {modifier:.3f} | Metadata: {sig_map[node_id]['metadata']}")

            # nodes in current shortest paths
            stage2_modifiers = {
                n["node_id"]: self.map_modifier_to_positive_value(n["pheromone_modifier"])
                for n in significant_nodes
            }

            for node_id, modifier in stage2_modifiers.items():
                if modifier >= threshold:
                    nodes_added += 1
                    print(f"Node considered: {node_id}")
                    print(f"Metadata: {sig_map[node_id]['metadata']}")
                    print(f"Modifier: {modifier:.3f}")
                    
                if nodes_added >= min_nodes_to_add:
                    # TODO: Rethink if this is too lenient
                    print("Enough nodes already present in current shortest paths.")
                    break
                
            #TODO: now create queue and start adding nodes until min_nodes_to_add is reached
            #Sorted by 1) cost in order closest to current optimal solution paths
            #          2) cost in order of closest to ANY shortest path part of the shortest paths list

            print(f"Significant nodes: {len(significant_nodes)} | Mod range: [{min(stage2_modifiers.values()):.3f}, {max(stage2_modifiers.values()):.3f}]")

        print(f"\nInstruction Summary:\n  Intent: {intent_type}\n  Confidence: {confidence:.3f}")

        # sorted_nodes = sorted(significant_nodes, key=lambda x: x["similarity"], reverse=True)[:100]
        # for node in sorted_nodes:
        #     print(
        #         f"Node {node['node_id']}: similarity {node['similarity']:.3f}, modifier {node['pheromone_modifier']:.3f}"
        #     )
        #     print("Metadata:", node["metadata"])
        #     print("---")


    def run(self, iterations=100, n=0):
        for _ in range(iterations):
            self.best_iter_tour = None
            self.best_iter_length = float('inf')
            
            for ant in self.ants:
                ant.reset()
                # construct tour
                while len(ant.tour) < self.num_nodes:
                    if self.negative_intent is not None:
                        modifier = self.confidence * 4 - 1.5 
                        if modifier < 1: modifier = 1
                        if not ant.choose_next_node(self.pheromones.matrix, self.heuristic, self.alpha, self.beta, self.negative_intent.matrix, upsilon=modifier):
                            break  # ant got stuck
                    else:
                        if not ant.choose_next_node(self.pheromones.matrix, self.heuristic, self.alpha, self.beta):
                            break  # ant got stuck
                
                # Only calculate length if tour is complete
                if len(ant.tour) == self.num_nodes:
                    ant.tour_length = self.calculate_tour_length(ant.tour)
                    
                    # Skip if tour uses impossible paths
                    if not np.isinf(ant.tour_length):
                        # update best
                        if ant.tour_length < self.best_length:
                            self.best_tour = ant.tour.copy()
                            self.best_length = ant.tour_length
                        if ant.tour_length < self.best_iter_length:
                            self.best_iter_tour = ant.tour.copy()
                            self.best_iter_length = ant.tour_length
                else:
                    ant.tour_length = float('inf')

            self.pheromones.evaporate()

            # Only deposit if we have valid tours
            if self.best_iter_tour is not None and self.best_tour is not None:
                if np.random.rand() < 0.25:  # 25% iteration-best
                    self.pheromones.deposit(self.best_iter_tour, self.best_iter_length)
                else:  # 75% global-best
                    self.pheromones.deposit(self.best_tour, self.best_length)
            elif self.best_iter_tour is not None:
                # Only iteration-best is valid
                self.pheromones.deposit(self.best_iter_tour, self.best_iter_length)
            elif self.best_tour is not None:
                # Only global-best is valid (shouldn't happen but just in case)
                self.pheromones.deposit(self.best_tour, self.best_length)
            else:
                # No valid tours at all
                print(f"Warning: No valid tours found in iteration {n+1}")
            
            print(f"Iteration {n+1}: Best length {self.best_length}")

    def calculate_tour_length(self, tour):
        if tour is None or len(tour) < self.num_nodes:
            return float('inf')
        
        length = 0
        for i in range(len(tour)-1):
            cost = self.cost_matrix[tour[i]][tour[i+1]]
            if np.isinf(cost):
                return float('inf')
            length += cost
        
        final_cost = self.cost_matrix[tour[-1]][tour[0]]
        if np.isinf(final_cost):
            return float('inf')
        
        return length + final_cost
    
    def compute_num_nodes_to_add(self):
        """Heuristic: how many nodes to add. Conservative growth with problem size."""
        return max(1, int(np.ceil(np.sqrt(self.num_nodes))+10))
    
    def map_modifier_to_positive_value(self, modifier):
        # map [1, 2] → [0, 1], anything <1 → 0
        value = max(0.0, min(modifier - 1.0, 1.0))
        return float(value)