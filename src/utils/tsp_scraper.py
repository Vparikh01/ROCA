import pickle
import requests
import numpy as np
from pathlib import Path

def parse_tsp_file(content):
    """Parse TSPLIB format file content."""
    lines = content.strip().split('\n')
    
    # Parse header
    metadata = {}
    i = 0
    while i < len(lines) and not lines[i].startswith('NODE_COORD_SECTION'):
        if ':' in lines[i]:
            key, value = lines[i].split(':', 1)
            metadata[key.strip()] = value.strip()
        i += 1
    
    # Parse coordinates
    i += 1  # Skip NODE_COORD_SECTION line
    coords = []
    while i < len(lines) and not lines[i].startswith('EOF'):
        parts = lines[i].strip().split()
        if len(parts) >= 3:
            # Format: node_id x y
            coords.append([float(parts[1]), float(parts[2])])
        i += 1
    
    return {
        'metadata': metadata,
        'coordinates': np.array(coords),
        'n_nodes': len(coords)
    }

def download_tsplib_graph(name):
    """Download a graph from TSPLIB."""
    base_url = "http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/"
    url = f"{base_url}{name}.tsp.gz"
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Decompress if gzipped
        import gzip
        content = gzip.decompress(response.content).decode('utf-8')
        return parse_tsp_file(content)
    except:
        # Try without .gz extension
        url = f"{base_url}{name}.tsp"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return parse_tsp_file(response.text)

def save_tsplib_graphs(graph_names, output_dir='tsplib_graphs'):
    """Download and save multiple TSPLIB graphs as pickle files."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    graphs = {}
    for name in graph_names:
        print(f"Downloading {name}...")
        try:
            graph = download_tsplib_graph(name)
            graphs[name] = graph
            
            # Save individual pickle file
            pickle_file = output_path / f"{name}.pkl"
            with open(pickle_file, 'wb') as f:
                pickle.dump(graph, f)
            print(f"  Saved to {pickle_file}")
        except Exception as e:
            print(f"  Error downloading {name}: {e}")
    
    # Save all graphs in one file
    all_graphs_file = output_path / "all_graphs.pkl"
    with open(all_graphs_file, 'wb') as f:
        pickle.dump(graphs, f)
    print(f"\nSaved all graphs to {all_graphs_file}")
    
    return graphs

# Example usage
if __name__ == "__main__":
    # Five examples per strict node range
    
    # Small / low complexity (20–50 nodes)
    graph_names_small = [
        'burma14',  # 14 nodes (slightly below 20, optional)
        'ulysses16',# 16 nodes (optional)
        'att48',    # 48 nodes
        'berlin52', # 52 nodes (just above 50)
        'eil51',    # 51 nodes
    ]
    
    # Medium complexity (100–300 nodes)
    graph_names_medium = [
        'kroA100',  # 100 nodes
        'lin105',   # 105 nodes
        'pr107',    # 107 nodes
        'a280',     # 280 nodes
        'bier127',  # 127 nodes
    ]
    
    # Complex (500–2000 nodes)
    graph_names_complex = [
        'pcb442',   # 442 nodes (slightly below 500, optional)
        'rat575',   # 575 nodes
        'd657',     # 657 nodes
        'fl1400',   # 1400 nodes
        'pla85900', # 85900 nodes (way above 2000, can skip if strict)
    ]
    
    graph_names = graph_names_small + graph_names_medium + graph_names_complex
    graphs = save_tsplib_graphs(graph_names)
    
    # Load example
    print("\nLoading example:")
    with open('tsplib_graphs/att48.pkl', 'rb') as f:
        graph = pickle.load(f)
    print(f"att48: {graph['n_nodes']} nodes")
    print(f"Coordinates shape: {graph['coordinates'].shape}")