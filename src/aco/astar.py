import networkx as nx
import numpy as np

def build_metric_closure(G, required_nodes):
    """
    Build metric closure for required nodes in G.
    
    Returns:
        cost_matrix: 2D NumPy array [i][j] = shortest distance from required_nodes[i] to required_nodes[j]
        reduced_graph: NetworkX graph with only required nodes as complete graph
        shortest_paths: dict[(u, v)] = list of nodes representing shortest path from u to v
    """
    n = len(required_nodes)
    cost_matrix = np.zeros((n, n))
    reduced_graph = nx.Graph()
    shortest_paths = {}

    # Add nodes to reduced graph
    reduced_graph.add_nodes_from(required_nodes)

    for i, u in enumerate(required_nodes):
        # Single-source shortest paths from u
        lengths, paths = nx.single_source_dijkstra(G, u, weight="weight")
        for j, v in enumerate(required_nodes):
            if i == j:
                cost_matrix[i][j] = 0
                continue
            cost_matrix[i][j] = lengths[v]
            reduced_graph.add_edge(u, v, weight=lengths[v])
            shortest_paths[(u, v)] = paths[v]

    return cost_matrix, reduced_graph, shortest_paths

def exclusion_closure_update(G, required_nodes, cost_matrix, shortest_paths, excluded_nodes, index_map):
    """
    Recomputes shortest paths and cost matrix after excluding nodes,
    ensuring all paths go around excluded nodes (if possible).
    """
    import numpy as np
    import networkx as nx

    print(f"\n{'='*60}")
    print("EXCLUSION CLOSURE UPDATE")
    print(f"{'='*60}")
    print(f"Total nodes in graph: {G.number_of_nodes()}")
    print(f"Required nodes: {len(required_nodes)}")
    print(f"Excluding nodes: {excluded_nodes}")

    excluded_set = set(excluded_nodes)
    required_set = set(required_nodes)

    # Prevent required nodes from being excluded
    overlap = excluded_set & required_set
    if overlap:
        print(f"Warning: {len(overlap)} required nodes attempted for exclusion, ignoring them.")
        excluded_set -= overlap
        excluded_nodes = list(excluded_set)

    if not excluded_nodes:
        print("No valid nodes to exclude. Returning original data.\n")
        return cost_matrix, shortest_paths

    # Create filtered graph without excluded nodes
    G_filtered = G.copy()
    G_filtered.remove_nodes_from(excluded_nodes)
    print(f"Remaining nodes after exclusion: {G_filtered.number_of_nodes()}")

    # Connectivity check
    if not nx.is_connected(G_filtered):
        print("Warning: Graph is disconnected after exclusion. Some paths may be unreachable.")

    # Identify paths affected by exclusions
    affected_pairs = [
        (u, v)
        for (u, v), path in shortest_paths.items()
        if any(node in excluded_set for node in path)
    ]
    print(f"Affected paths to recompute: {len(affected_pairs)}")

    updated_count, blocked_count = 0, 0

    for u, v in affected_pairs:
        i, j = index_map.get(u), index_map.get(v)
        if i is None or j is None:
            continue

        try:
            # Find new shortest path avoiding excluded nodes
            new_length, new_path = nx.single_source_dijkstra(G_filtered, u, v, weight="weight")
            shortest_paths[(u, v)] = new_path
            cost_matrix[i][j] = new_length
            cost_matrix[j][i] = new_length
            updated_count += 1

        except (nx.NetworkXNoPath, nx.NodeNotFound):
            shortest_paths[(u, v)] = []
            cost_matrix[i][j] = np.inf
            cost_matrix[j][i] = np.inf
            blocked_count += 1

    print(f"Paths successfully rerouted: {updated_count}")
    if blocked_count > 0:
        print(f"{blocked_count} paths became impossible due to disconnection.")
    print(f"{'='*60}\n")

    return cost_matrix, shortest_paths, G_filtered

import networkx as nx
import numpy as np

def inclusion_filter(self, current_opt_path):
    """
    Find candidate nodes near the current optimal tour paths and shortest paths.
    Uses precomputed distances for massive speedup.
    """
    G = self.completeGraph
    
    # Compute radius cutoff based on average edge weight
    edge_weights = [d.get("weight", 1.0) for _, _, d in G.edges(data=True)]
    avg_edge_weight = np.mean(edge_weights)
    radius_cutoff = avg_edge_weight * 2
    
    # Collect all unique nodes from optimal path and shortest paths
    # Filter to only nodes that exist in graph
    opt_path_nodes = set(n for n in current_opt_path if n in G)
    all_shortest_nodes = set()
    for path in self.shortest_paths.values():
        all_shortest_nodes.update(n for n in path if n in G)
    
    # Union of both sets - these are our "anchor" nodes
    anchor_nodes = opt_path_nodes | all_shortest_nodes
    
    print(f"Optimal path nodes: {len(opt_path_nodes)}")
    print(f"All shortest path nodes: {len(all_shortest_nodes)}")
    print(f"Total anchor nodes: {len(anchor_nodes)}")
    
    # Single multi-source Dijkstra from all anchor nodes with cutoff
    # This computes distances from ANY anchor node to all reachable nodes within cutoff
    distances = nx.multi_source_dijkstra_path_length(
        G, anchor_nodes, cutoff=radius_cutoff, weight="weight"
    )
    
    # Now compute separate distances for opt vs sp nodes
    # This is still fast because we're just doing 2 more multi-source calls with cutoff
    dist_from_opt = nx.multi_source_dijkstra_path_length(
        G, opt_path_nodes, cutoff=radius_cutoff, weight="weight"
    )
    dist_from_sp = nx.multi_source_dijkstra_path_length(
        G, all_shortest_nodes, cutoff=radius_cutoff, weight="weight"
    )
    
    filtered_candidates = []
    
    for node in distances.keys():  # Only check reachable nodes
        if node in set(self.required_nodes):
            continue  # skip required nodes
        if node in anchor_nodes:
            continue  # skip nodes already in paths
        if node in set(self.significant_map.keys()):
            continue  # skip already significant nodes
        
        d_opt = dist_from_opt.get(node, float("inf"))
        d_sp = dist_from_sp.get(node, float("inf"))
        
        # Include nodes within radius of both sets
        if d_opt <= radius_cutoff and d_sp <= radius_cutoff:
            filtered_candidates.append((node, d_opt, d_sp))
    
    print(f"Filtered candidate nodes (near paths): {len(filtered_candidates)}")
    
    return filtered_candidates

def inclusion_closure_update(G, required_nodes, cost_matrix, shortest_paths, included_nodes, index_map):
    """
    Forcefully appends included nodes into the closest required-node pair path,
    recomputing via two Dijkstra runs if necessary.
    Optimized to cache Dijkstra results for runtime reduction.
    """
    def safe_path_cost(G, path):
        total = 0.0
        for a, b in zip(path[:-1], path[1:]):
            total += G[a][b].get("weight", 1.0) if G.has_edge(a, b) else 1.0
        return total

    print(f"\n{'='*60}")
    print(f"INCLUSION CLOSURE UPDATE")
    print(f"{'='*60}")
    print(f"Nodes to include: {len(included_nodes)}")

    included_nodes = [n for n in included_nodes if n in G]
    if not included_nodes:
        print("No valid included nodes found in graph.")
        return cost_matrix, shortest_paths

    # Precompute Dijkstra results for all required nodes once
    dist_cache, path_cache = {}, {}
    for src in required_nodes:
        dist, paths = nx.single_source_dijkstra(G, src, weight="weight")
        dist_cache[src] = dist
        path_cache[src] = paths

    for node in included_nodes:
        best_pair = None
        best_path = None
        best_cost = float("inf")

        for u in required_nodes:
            if node not in dist_cache[u]:
                continue
            for v in required_nodes:
                if u == v or node not in dist_cache[v]:
                    continue

                total_cost = dist_cache[u][node] + dist_cache[v][node]
                if total_cost < best_cost:
                    try:
                        path_u_node = path_cache[u][node]
                        path_node_v = list(reversed(path_cache[v][node]))
                        full_path = path_u_node + path_node_v[1:]
                        best_cost = safe_path_cost(G, full_path)
                        best_pair = (u, v)
                        best_path = full_path
                    except KeyError:
                        continue

        if best_pair and best_path:
            u, v = best_pair
            shortest_paths[(u, v)] = best_path
            shortest_paths[(v, u)] = list(reversed(best_path))

            i, j = index_map.get(u), index_map.get(v)
            if i is not None and j is not None:
                cost_matrix[i][j] = best_cost
                cost_matrix[j][i] = best_cost

            print(f"Node {node} appended into path ({u}, {v}) | cost = {best_cost:.3f}")
        else:
            print(f"Node {node} unreachable from required nodes.")

    print(f"\nAll included nodes processed successfully.")
    print(f"{'='*60}\n")
    return cost_matrix, shortest_paths