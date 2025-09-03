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
