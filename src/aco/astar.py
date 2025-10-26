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
    Updates shortest paths and cost matrix after excluding nodes.
    
    Args:
        G: Complete graph
        required_nodes: List of required node IDs
        cost_matrix: Current cost matrix (n x n)
        shortest_paths: Dict of (u,v) -> path
        excluded_nodes: List of node IDs to exclude
        index_map: Dict mapping node ID -> index in cost_matrix
    
    Returns:
        Updated cost_matrix, shortest_paths
    """
    print(f"\n{'='*60}")
    print(f"EXCLUSION CLOSURE UPDATE")
    print(f"{'='*60}")
    print(f"Graph stats:")
    print(f"  Total nodes: {G.number_of_nodes()}")
    print(f"  Required nodes: {len(required_nodes)}")
    print(f"  Nodes to exclude: {len(excluded_nodes)}")
    
    # CRITICAL: Ensure no required nodes are excluded
    excluded_set = set(excluded_nodes)
    required_set = set(required_nodes)
    overlap = excluded_set & required_set
    
    if overlap:
        print(f"  ERROR: Attempted to exclude {len(overlap)} required nodes!")
        print(f"  Removing them from exclusion list...")
        excluded_nodes = list(excluded_set - required_set)
        excluded_set = set(excluded_nodes)
        print(f"  Corrected exclusion list: {len(excluded_nodes)} nodes")
    
    if not excluded_nodes:
        print("  No nodes to exclude. Returning original matrices.")
        print(f"{'='*60}\n")
        return cost_matrix, shortest_paths
    
    # Create filtered graph
    G_filtered = G.copy()
    G_filtered.remove_nodes_from(excluded_nodes)
    
    print(f"  Nodes after exclusion: {G_filtered.number_of_nodes()}")
    
    # Verify all required nodes still exist
    missing_required = [n for n in required_nodes if n not in G_filtered]
    if missing_required:
        print(f"  FATAL ERROR: {len(missing_required)} required nodes missing from graph!")
        print(f"  This should never happen. Aborting exclusion.")
        return cost_matrix, shortest_paths
    
    # Check connectivity
    is_connected = nx.is_connected(G_filtered)
    print(f"  Graph connected: {is_connected}")
    
    if not is_connected:
        components = list(nx.connected_components(G_filtered))
        print(f"  WARNING: Graph is disconnected!")
        print(f"  Number of components: {len(components)}")
        
        # Check required nodes distribution
        req_per_component = []
        for comp in components:
            req_in_comp = [n for n in required_nodes if n in comp]
            req_per_component.append(len(req_in_comp))
        
        print(f"  Required nodes per component: {req_per_component}")
        
        if len([x for x in req_per_component if x > 0]) > 1:
            print(f"  ERROR: Required nodes split across multiple components!")
            print(f"  Some paths will be impossible.")
        
    # Find affected pairs (paths that go through excluded nodes)
    affected_pairs = [
        (u, v)
        for (u, v), path in shortest_paths.items()
        if any(node in excluded_set for node in path)
    ]
    
    print(f"\nProcessing affected paths:")
    print(f"  Total affected: {len(affected_pairs)}")
    
    blocked_count = 0
    updated_count = 0
    
    for u, v in affected_pairs:
        i = index_map.get(u)
        j = index_map.get(v)
        
        if i is None or j is None:
            continue
        
        try:
            # Try to find new path avoiding excluded nodes
            new_path = nx.shortest_path(G_filtered, u, v, weight="weight")
            new_length = nx.shortest_path_length(G_filtered, u, v, weight="weight")
            
            # Update with new path
            shortest_paths[(u, v)] = new_path
            cost_matrix[i][j] = new_length
            cost_matrix[j][i] = new_length
            updated_count += 1
            
        except nx.NetworkXNoPath:
            # No path exists - nodes in different components
            shortest_paths[(u, v)] = []
            cost_matrix[i][j] = np.inf
            cost_matrix[j][i] = np.inf
            blocked_count += 1
                
        except nx.NodeNotFound:
            # Node doesn't exist (shouldn't happen)
            shortest_paths[(u, v)] = []
            cost_matrix[i][j] = np.inf
            cost_matrix[j][i] = np.inf
            blocked_count += 1
    
    print(f"\nResults:")
    print(f"  Paths updated with new routes: {updated_count}")
    
    if blocked_count > 0:
        print(f"  WARNING: {blocked_count} paths are now impossible!")
        print(f"  This will make finding valid tours very difficult.")
    
    print(f"{'='*60}\n")
    
    return cost_matrix, shortest_paths