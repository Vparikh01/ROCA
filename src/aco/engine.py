import numpy as np
import networkx as nx
from src.aco.pheromones import PheromoneMatrix
from src.nlp.negative_intents import NegativeIntentMatrix
from src.nlp.positive_intents import PositiveIntentMatrix
from src.rl.edgeQ import EdgeQ
from src.aco.ant import Ant
from src.aco.config import load_config
from src.nlp.context_encoder import process_instruction
from src.nlp.context_encoder import extract_node_info
from src.aco.astar import exclusion_closure_update
from src.aco.astar import inclusion_closure_update
from src.aco.astar import inclusion_filter

cfg = load_config()

class MaxMinACO:
    def __init__(self, cost_matrix, start_node, reducedGraph, completeGraph, shortest_paths, 
                 required_nodes, index_map, optimization_mode="plain", seed=None):
        """
        optimization_mode options:
        - "plain": Standard ACO with no parameter optimization
        - "gradient": Gradient-based parameter adaptation
        - "evolution": Evolutionary parameter optimization
        - "qlearning": Q-learning edge selection only
        - "qlearning+gradient": Q-learning + gradient adaptation
        - "qlearning+evolution": Q-learning + evolutionary optimization
        """
        self.ITERATION_BEST_RATIO = 0.325
        self.cost_matrix = cost_matrix
        self.start_node = start_node
        self.num_nodes = len(cost_matrix)
        self.num_ants = min(cfg["num_ants"], self.num_nodes)
        self.alpha = cfg["alpha"]
        self.beta = cfg["beta"]
        self.seed = seed if seed is not None else cfg["seed"] or 0
        self.total_iterations = 0
        self.optimization_mode = optimization_mode
        avgCost = np.mean(cost_matrix[cost_matrix > 0]) if len(cost_matrix[cost_matrix > 0]) > 0 else 1.0
        print(f"Average Cost: {avgCost}")

        # Initialize Q-learner if needed
        self.Qlearner = None
        if "qlearning" in optimization_mode:
            self.Qlearner = EdgeQ(self.num_nodes, avgCost, alpha=0.1, gamma=0.9, n_step=3)

        # Macro-evolution / population settings
        self.macro_iter_size = cfg.get("macro_iter_size", 20)
        self.best_length_history = []
        
        # Evolution-specific defaults
        if "evolution" in optimization_mode:
            self._evo_defaults = {
                "selection_ratio": cfg.get("evo_selection_ratio", 0.5),
                "elite_frac": cfg.get("evo_elite_frac", 0.2),
                "mutation_sigma": cfg.get("evo_mutation_sigma", 0.15),
                "reinit_frac": cfg.get("evo_reinit_frac", 0.1),
                "diversity_threshold": cfg.get("evo_diversity_threshold", 0.01),
            }
        
        positive_costs = cost_matrix[cost_matrix > 0]
        expected_length = max(np.mean(positive_costs) * self.num_nodes, 1e-6)
        self.pheromones = PheromoneMatrix(self.num_nodes, cfg["rho"], expected_length)
        self.negative_intent = None
        self.positive_intent = None
        self.shortest_paths = shortest_paths
        unique_nodes = set().union(*self.shortest_paths.values())
        self.completeGraph = completeGraph
        subgraph = completeGraph.subgraph(unique_nodes).copy()
        self.node_info = extract_node_info(subgraph)
        self.required_nodes = required_nodes
        self.index_map = index_map

        # Initialize heuristic matrix
        self.heuristic = np.zeros_like(cost_matrix, dtype=float)
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if i != j and cost_matrix[i][j] > 0:
                    self.heuristic[i][j] = 1 / cost_matrix[i][j]
                else:
                    self.heuristic[i][j] = 0
                    
        self.best_tour = None
        self.best_length = float('inf')
        self.post_instruction_best_tour = None
        self.post_instruction_best_length = float('inf')
        
        # Create ant population
        self.ants = [
            Ant(
                start_node=self.start_node,
                num_nodes=self.num_nodes,
                Qlearner=self.Qlearner,
                graph=reducedGraph,
                seed=self.seed + i
            )
            for i in range(self.num_ants)
        ]

        # Initialize population diversity based on optimization mode
        if optimization_mode in ["gradient", "qlearning+gradient"]:
            # Gradient mode: keep first 5 ants at baseline, diversify rest
            for i, ant in enumerate(self.ants):
                if i <= 4:
                    continue
                a, b = self._sample_random_params()
                ant.alpha = float(a)
                ant.beta = float(b)
                # Initialize gradient tracking
                ant.mew_alpha = 0.0
                ant.mew_beta = 0.0
        elif optimization_mode in ["evolution", "qlearning+evolution"]:
            # Evolution mode: keep first ant at baseline, diversify rest
            for i, ant in enumerate(self.ants):
                if i == 0:
                    continue
                a, b = self._sample_random_params()
                ant.alpha = float(a)
                ant.beta = float(b)
        
        np.random.seed(self.seed)

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
        self.significant_map = {n["node_id"]: n for n in self.significant_nodes}

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
                unsafe = []
                
                for node in excluded:
                    G_temp = G.copy()
                    G_temp.remove_node(node)
                    if nx.is_connected(G_temp):
                        safe.append(node)
                        G = G_temp
                    else:
                        unsafe.append(node)

                print(f"Safe exclusions: {len(safe)} | Unsafe exclusions: {len(unsafe)}")

                all_to_remove = safe + unsafe
                G_filtered = self.completeGraph.copy()
                G_filtered.remove_nodes_from(all_to_remove)

                new_cost, new_shortest_paths, _ = exclusion_closure_update(
                    self.completeGraph, self.required_nodes, self.cost_matrix, self.shortest_paths, all_to_remove, self.index_map
                )

                broken_pairs = [
                    (u, v) for u in self.required_nodes for v in self.required_nodes
                    if u != v and ((u, v) not in new_shortest_paths or not new_shortest_paths[(u, v)])
                ]
                for u, v in broken_pairs:
                    try:
                        sp = nx.shortest_path(G_filtered, u, v, weight='weight')
                        new_shortest_paths[(u, v)] = sp
                        new_cost[self.index_map[u], self.index_map[v]] = sum(
                            self.cost_matrix[self.index_map[sp[i]], self.index_map[sp[i+1]]]
                            for i in range(len(sp)-1)
                        )
                    except nx.NetworkXNoPath:
                        new_shortest_paths[(u, v)] = []
                        new_cost[self.index_map[u], self.index_map[v]] = float('inf')

                self.cost_matrix = new_cost
                self.shortest_paths = new_shortest_paths
                self.completeGraph_filtered = G_filtered

                for i in range(self.num_nodes):
                    for j in range(self.num_nodes):
                        if i == j:
                            self.heuristic[i][j] = 0
                        elif self.cost_matrix[i][j] > 0:
                            self.heuristic[i][j] = 1 / self.cost_matrix[i][j]
                        else:
                            self.heuristic[i][j] = 0

                print("Hard exclusions applied with partial fallback for broken paths")
                self.safe = safe + unsafe
        else:
            print(f"\n{'='*60}\nAPPLYING PREFER INSTRUCTION: '{instruction}'\n{'='*60}")
            self.positive_intent = PositiveIntentMatrix(self.num_nodes)
            nodes_added = 0
            min_nodes_to_add = self.compute_num_nodes_to_add()
            threshold = 0.95

            node_modifier_map = {n["node_id"]: n["pheromone_modifier"] for n in significant_nodes}
            print(f"Significant nodes: {len(significant_nodes)} | Mod range: [{min(node_modifier_map.values()):.3f}, {max(node_modifier_map.values()):.3f}]")

            for i in range(self.num_nodes):
                for j in range(self.num_nodes):
                    if i == j:
                        continue
                    u, v = self.required_nodes[i], self.required_nodes[j]
                    if (u, v) in self.shortest_paths:
                        path = self.shortest_paths[(u, v)]
                        mods = [node_modifier_map[n] for n in path if n in node_modifier_map]
                        if mods:
                            avg_mod = np.mean(mods)
                            self.positive_intent.matrix[i][j] = self.map_modifier_to_positive_value(avg_mod)
                        else:
                            self.positive_intent.matrix[i][j] = 1e-6

            current_opt_nodes = [self.required_nodes[i] for i in self.best_tour]
            current_opt_path = []
            full_path = []
            for k in range(len(current_opt_nodes)):
                u = current_opt_nodes[k]
                v = current_opt_nodes[(k+1) % len(current_opt_nodes)]
                sp = None
                if (u, v) in self.shortest_paths and self.shortest_paths[(u, v)]:
                    sp = self.shortest_paths[(u, v)]
                else:
                    graph_for_query = getattr(self, 'completeGraph_filtered', self.completeGraph)
                    try:
                        sp = nx.shortest_path(graph_for_query, u, v, weight='weight')
                    except nx.NetworkXNoPath:
                        sp = []
                if not sp:
                    print(f"Warning: No path between {u} and {v} after exclusion")
                    full_path = []
                    break
                full_path.extend(sp[:-1])
                current_opt_path.extend(sp[:-1])
            current_opt_path.append(self.start_node)

            sig_map = {n["node_id"]: n for n in significant_nodes}
            subset = [
                sig_map[node_id]
                for node_id in current_opt_path
                if node_id in sig_map
            ]

            stage2_modifiers = {
                n["node_id"]: self.map_modifier_to_positive_value(n["pheromone_modifier"])
                for n in significant_nodes
            }

            for node_id, modifier in stage2_modifiers.items():
                if modifier >= threshold:
                    nodes_added += 1
                    if not hasattr(self, 'added_nodes_set'):
                        self.added_nodes_set = set()
                    self.added_nodes_set.add(node_id)
                    
                if nodes_added >= min_nodes_to_add:
                    print("Enough nodes already present in current shortest paths: " + str(nodes_added))
                    if hasattr(self, 'positive_intent') and self.positive_intent is not None:
                        self.positive_intent.matrix.fill(1e-6)
                    return
                
            if nodes_added < min_nodes_to_add:
                num_additional = min_nodes_to_add - nodes_added
                print(f"\nNeed {num_additional} more nodes — building filtered candidate set...")

                filtered_candidates = inclusion_filter(self, current_opt_path)

                if not filtered_candidates:
                    print("No candidates passed the distance thresholds. Consider adjusting thresholds.")
                else:
                    temp_nodes_set = {nid for nid, _, _ in filtered_candidates}
                    temp_subgraph = self.completeGraph.subgraph(temp_nodes_set).copy()
                    temp_node_info = extract_node_info(temp_subgraph)

                    if not temp_node_info:
                        print("[Warning] No node metadata available for filtered candidates — skipping inference.")
                    else:
                        _, conf_sub, sig_nodes_sub = process_instruction(self.instruction, temp_node_info)
                        print(f"Inference returned {len(sig_nodes_sub)} significant nodes in filtered set (confidence: {conf_sub:.3f})")

                        if sig_nodes_sub:
                            sig_nodes_sub_sorted = sorted(sig_nodes_sub, key=lambda x: x.get("pheromone_modifier", 0.0), reverse=True)
                            dist_map_opt = {nid: d for nid, d, _ in filtered_candidates}
                            dist_map_sp = {nid: d for nid, _, d in filtered_candidates}

                            print("\nInferred Significant Candidates (post-filtering):")
                            print(f"{'Node ID':<12} {'Modifier':<10} {'d_opt':<10} {'d_sp':<10} {'Metadata'}")
                            print("-" * 80)
                            for nd in sig_nodes_sub_sorted[:15]:
                                nid = nd["node_id"]
                                mod = nd.get("pheromone_modifier", 0.0)
                                d1 = dist_map_opt.get(nid, float("inf"))
                                d2 = dist_map_sp.get(nid, float("inf"))
                                meta = temp_node_info.get(nid, {}).get("metadata", {})
                                print(f"{nid:<12} {mod:<10.3f} {d1:<10.2f} {d2:<10.2f} {meta}")

                            self.filtered_inferred_candidates = sig_nodes_sub_sorted
                            top_inferred = self.filtered_inferred_candidates[:num_additional]
                            included_ids = [node["node_id"] for node in top_inferred]
                            if not hasattr(self, 'added_nodes_set'):
                                self.added_nodes_set = set()
                            self.added_nodes_set.update(included_ids)
                            self.cost_matrix, self.shortest_paths = inclusion_closure_update(
                                self.completeGraph,
                                self.required_nodes,
                                self.cost_matrix,
                                self.shortest_paths,
                                included_ids,
                                self.index_map
                            )
                        else:
                            print("No significant nodes returned by inference on filtered set.")
                                    
            print(f"Significant nodes: {len(significant_nodes)} | Mod range: [{min(stage2_modifiers.values()):.3f}, {max(stage2_modifiers.values()):.3f}]")
            
        # Resynchronize best_length and tour after closure updates
        if self.best_tour is not None:
            new_cost = sum(
                self.cost_matrix[self.best_tour[i]][self.best_tour[i+1]]
                for i in range(len(self.best_tour)-1)
            ) + self.cost_matrix[self.best_tour[-1]][self.best_tour[0]]

            if abs(new_cost - self.best_length) > 1e-6:
                print(f"[Sync] Adjusting best_length: {self.best_length:.3f} → {new_cost:.3f}")
                self.best_length = new_cost
        print(f"Current ACO Best Length: {self.best_length:.3f}")
        print(f"\nInstruction Summary:\n  Intent: {intent_type}\n  Confidence: {confidence:.3f}")

    # ---------- Population evolution helpers ----------
    def _ant_fitness(self, ant):
        if not np.isfinite(ant.tour_length) or ant.tour_length <= 0:
            return 0.0
        return 1.0 / ant.tour_length

    def _lognormal(self, mean, sigma):
        return np.random.lognormal(mean=mean, sigma=sigma)

    def _sample_random_params(self, mew_alpha=0.0, mew_beta=0.0, sigma=0.25):
        """Sample new random alpha,beta near base cfg values (lognormal around base)."""
        base_a = cfg.get("alpha", 1.0)
        base_b = cfg.get("beta", 5.0)
        a = base_a * self._lognormal(mew_alpha, sigma)
        b = base_b * self._lognormal(mew_beta, sigma)
        a = np.clip(a, 0.1, 10.0)
        b = np.clip(b, 0.1, 10.0)
        return (a, b)

    def _reinit_ant_params(self, ant):
        a, b = self._sample_random_params()
        ant.alpha = float(a)
        ant.beta = float(b)

    def evolve_population_gradient(self):
        """Gradient-based parameter adaptation (first engine approach)."""
        # Compute baseline cost from first 5 ants
        baseline_cost = 0.0
        count = 0
        for i in range(5):
            ant = self.ants[i]
            baseline_cost += ant.tour_sum / self.macro_iter_size
            count += 1
        baseline_cost /= max(count, 1)

        base_alpha = cfg["alpha"]
        base_beta = cfg["beta"]
        eps = 1e-8

        # Update remaining ants via directional gradient
        for ant in self.ants[5:]:
            ant_cost = ant.tour_sum / self.macro_iter_size
            if ant_cost == 0.0:
                ant.reset_tour_sum()
                continue

            # Joint perturbation vector (LOG-SPACE)
            delta = np.array([
                np.log(ant.alpha / base_alpha + eps),
                np.log(ant.beta / base_beta + eps)
            ])

            norm = np.linalg.norm(delta)
            if norm < eps:
                ant.reset_tour_sum()
                continue

            # Directional derivative
            direction = delta / norm
            delta_cost = ant_cost - baseline_cost

            # Cost-scaled directional gradient
            grad = -(delta_cost / (baseline_cost + eps)) * direction

            # Shift log-normal mean
            if grad[0] > 0:
                ant.mew_alpha += min(0.05, grad[0])
            else:
                ant.mew_alpha += max(-0.05, grad[0])
            if grad[1] > 0:
                ant.mew_beta += min(0.05, grad[1])
            else:
                ant.mew_beta += max(-0.05, grad[1])

            print(f"\n[ANT {self.ants.index(ant)-4} GRADIENT]")
            print(f"  direction: {direction}")
            print(f"  delta_cost: {delta_cost:.4f}")
            print(f"  mew_alpha: {ant.mew_alpha:.4f}")
            print(f"  mew_beta: {ant.mew_beta:.4f}")

            # Resample new alpha,beta
            ant.alpha, ant.beta = self._sample_random_params(
                mew_alpha=ant.mew_alpha,
                mew_beta=ant.mew_beta,
                sigma=0.25
            )

            ant.alpha = max(0.1, ant.alpha)
            ant.beta = max(0.1, ant.beta)
            print(f"  New alpha: {ant.alpha:.4f}")
            print(f"  New beta: {ant.beta:.4f}")

            ant.reset_tour_sum()

    def evolve_population_evolution(self, selection_ratio=None, elite_frac=None, 
                                    mutation_sigma=None, reinit_frac=None, 
                                    diversity_threshold=None):
        """Evolutionary parameter optimization (second engine approach)."""
        # Resolve defaults
        selection_ratio = selection_ratio if selection_ratio is not None else self._evo_defaults["selection_ratio"]
        elite_frac = elite_frac if elite_frac is not None else self._evo_defaults["elite_frac"]
        mutation_sigma = mutation_sigma if mutation_sigma is not None else self._evo_defaults["mutation_sigma"]
        reinit_frac = reinit_frac if reinit_frac is not None else self._evo_defaults["reinit_frac"]
        diversity_threshold = diversity_threshold if diversity_threshold is not None else self._evo_defaults["diversity_threshold"]

        num = len(self.ants)
        if num == 0:
            return

        # Collect fitnesses
        fitnesses = np.array([self._ant_fitness(a) for a in self.ants])
        if fitnesses.sum() <= 0:
            for i in np.random.choice(range(num), max(1, int(np.ceil(num * reinit_frac))), replace=False):
                self._reinit_ant_params(self.ants[i])
            return

        # Ranking by tour_length
        ranked_idx = np.argsort([a.tour_length if np.isfinite(a.tour_length) else float('inf') for a in self.ants])
        n_elite = max(1, int(np.floor(elite_frac * num)))

        # Keep elites
        elites = [self.ants[i] for i in ranked_idx[:n_elite]]

        # Selection pool (roulette by fitness)
        parent_probs = fitnesses / fitnesses.sum()
        n_parents = max(1, int(np.ceil(selection_ratio * num)))
        parent_indices = np.random.choice(range(num), size=n_parents, p=parent_probs, replace=True)

        # Build new parameter list
        new_params = []

        # Keep elites first
        for e in elites:
            new_params.append((e.alpha, e.beta))

        # Generate children by sampling parents + mutation
        for _ in range(num - len(new_params)):
            parent_idx = np.random.choice(parent_indices)
            parent = self.ants[parent_idx]
            child_alpha = parent.alpha * self._lognormal(0.0, mutation_sigma)
            child_beta = parent.beta * self._lognormal(0.0, mutation_sigma)
            new_params.append((child_alpha, child_beta))

        # Reinit a small fraction
        n_reinit = max(1, int(np.floor(reinit_frac * num)))
        reinit_indices = np.random.choice(
            range(n_elite, len(new_params)),
            min(n_reinit, len(new_params) - n_elite),
            replace=False
        )

        for ri in reinit_indices:
            new_params[ri] = self._sample_random_params()
        
        # Apply new params to ants
        for ant, (a_p, b_p) in zip(self.ants, new_params):
            ant.alpha = float(a_p)
            ant.beta = float(b_p)

        # Diversity check
        alphas = np.array([ant.alpha for ant in self.ants])
        betas = np.array([ant.beta for ant in self.ants])
        if np.var(alphas) < diversity_threshold and np.var(betas) < diversity_threshold:
            n_force = max(1, int(np.ceil(0.25 * num)))
            idxs = np.random.choice(
                range(n_elite, num),
                min(n_force, num - n_elite),
                replace=False
            )
            for i in idxs:
                self._reinit_ant_params(self.ants[i])

    def run(self, iterations=100, n=0):
        for iteration in range(iterations):
            self.total_iterations += 1
            self.best_iter_tour = None
            self.best_iter_length = float('inf')

            # Reset all ants
            for ant in self.ants:
                ant.reset()
                ant.update_lambda()

            unfinished_ants = np.array([True] * self.num_ants)

            # Construct tours step by step
            for step in range(self.num_nodes - 1):
                active_ants = [ant for ant, active in zip(self.ants, unfinished_ants) if active]

                if not active_ants:
                    break

                for ant in active_ants:
                    prev_node = ant.current_node
                    
                    if self.negative_intent is not None:
                        modifier = self.confidence
                        ant.choose_next_node(
                            self.pheromones.matrix,
                            self.heuristic,
                            self.negative_intent.matrix,
                            upsilon=modifier
                        )
                    elif self.positive_intent is not None:
                        modifier = self.confidence
                        ant.choose_next_node(
                            self.pheromones.matrix,
                            self.heuristic,
                            self.positive_intent.matrix,
                            upsilon=modifier
                        )
                    else:
                        ant.choose_next_node(self.pheromones.matrix, self.heuristic)
                        
                    new_node = ant.current_node
                    if self.Qlearner is not None and prev_node != new_node:
                        cost = self.cost_matrix[prev_node, new_node]
                        if np.isfinite(cost):
                            self.Qlearner.observe(prev_node, new_node, cost)
                            
                for idx, ant in enumerate(self.ants):
                    if len(ant.tour) >= self.num_nodes:
                        unfinished_ants[idx] = False

            # Calculate tour lengths
            tour_lengths = []
            for ant in self.ants:
                ant.tour_length = self.calculate_tour_length(ant.tour)
                tour_lengths.append(ant.tour_length)
            tour_lengths = np.array(tour_lengths)

            # Update iteration-best and global-best
            finite_mask = np.isfinite(tour_lengths)
            if finite_mask.any():
                best_idx = np.argmin(tour_lengths[finite_mask])
                best_ant = np.array(self.ants)[finite_mask][best_idx]

                self.best_iter_tour = best_ant.tour.copy()
                self.best_iter_length = best_ant.tour_length

                if self.best_iter_length < self.best_length:
                    self.best_length = self.best_iter_length
                    self.best_tour = self.best_iter_tour.copy()

                if self.positive_intent is not None or self.negative_intent is not None:
                    post_instr_best_idx = np.argmin(tour_lengths[finite_mask])
                    post_instr_best_ant = np.array(self.ants)[finite_mask][post_instr_best_idx]
                    self.post_instruction_best_tour = post_instr_best_ant.tour.copy()
                    self.post_instruction_best_length = post_instr_best_ant.tour_length

            # Evaporate pheromones
            self.pheromones.evaporate()

            # Deposit pheromones
            if self.best_iter_tour is not None and self.best_tour is not None:
                if np.random.rand() < self.ITERATION_BEST_RATIO:
                    self.pheromones.deposit(self.best_iter_tour, self.best_iter_length)
                else:
                    self.pheromones.deposit(self.best_tour, self.best_length)
            elif self.best_iter_tour is not None:
                self.pheromones.deposit(self.best_iter_tour, self.best_iter_length)
            elif self.best_tour is not None:
                self.pheromones.deposit(self.best_tour, self.best_length)
            else:
                print(f"Warning: No valid tours found in iteration {iteration}")
            
            # Q-learning episodic reset
            if self.Qlearner is not None:
                self.Qlearner.flush()
            
            # Macro-evolution trigger based on optimization mode
            if self.total_iterations % self.macro_iter_size == 0:
                if self.optimization_mode in ["gradient", "qlearning+gradient"]:
                    self.evolve_population_gradient()
                elif self.optimization_mode in ["evolution", "qlearning+evolution"]:
                    self.evolve_population_evolution(
                        selection_ratio=self._evo_defaults["selection_ratio"],
                        elite_frac=self._evo_defaults["elite_frac"],
                        mutation_sigma=self._evo_defaults["mutation_sigma"],
                        reinit_frac=self._evo_defaults["reinit_frac"],
                        diversity_threshold=self._evo_defaults["diversity_threshold"]
                    )

            print(f"Iteration {self.total_iterations}: Best length {self.best_length:.3f}")
            self.best_length_history.append(self.best_length)
            
    def calculate_tour_length(self, tour):
        if tour is None or len(tour) < self.num_nodes:
            return float('inf')
        
        tour_np = np.array(tour, dtype=int)
        edges = self.cost_matrix[tour_np[:-1], tour_np[1:]]
        if np.isinf(edges).any():
            return float('inf')
        
        final_edge = self.cost_matrix[tour_np[-1], tour_np[0]]
        return float(np.sum(edges) + final_edge) if not np.isinf(final_edge) else float('inf')
    
    def compute_num_nodes_to_add(self):
        """Heuristic: how many nodes to add. Conservative growth with problem size."""
        return max(1, int(np.ceil(np.sqrt(self.num_nodes))+10))
    
    def map_modifier_to_positive_value(self, modifier):
        # map [1, 2] → [0, 1], anything <1 → 0
        value = max(0.0, min(modifier - 1.0, 1.0))
        return float(value)