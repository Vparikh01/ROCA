import os
import pickle
import random
import networkx as nx
import matplotlib.pyplot as plt

# ------------------ Paths ------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PALO_ALTO_PKL = os.path.join(BASE_DIR, "palo_alto_graph.pkl")
AIRPORTS_PKL = os.path.join(BASE_DIR, "europe_airports_graph.pkl")

# ------------------ Load Graphs ------------------
def load_graph(pkl_path):
    with open(pkl_path, "rb") as f:
        G = pickle.load(f)
    # Convert to undirected (NetworkX preserves all attributes)
    if G.is_directed():
        G = G.to_undirected()
    return G

G_map = load_graph(PALO_ALTO_PKL)
G_airports = load_graph(AIRPORTS_PKL)

print(f"Palo Alto graph: {len(G_map.nodes)} nodes, {len(G_map.edges)} edges")
print(f"European airports graph: {len(G_airports.nodes)} nodes, {len(G_airports.edges)} edges")

# ------------------ Visualizations ------------------
def visualize_palo_alto_network(G):
    fig, ax = plt.subplots(1, 1, figsize=(15, 15))
    pos = {n: (data['x'], data['y']) for n, data in G.nodes(data=True)}
    nx.draw(G, pos, ax=ax, node_size=0, edge_color='lightgray', width=0.5)

    # POIs
    poi_nodes = [n for n in G.nodes() if G.nodes[n].get('is_poi', False)]
    for poi in poi_nodes:
        x, y = G.nodes[poi]['x'], G.nodes[poi]['y']
        ax.scatter(x, y, s=100, c='red', zorder=3, alpha=0.8)
        ax.annotate(G.nodes[poi].get('name', 'unknown'), (x, y),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, ha='left', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

    # Edge labels (lengths)
    edge_labels = {(u, v): f"{int(d.get('length', 0))}m" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=4, ax=ax, alpha=0.7)
    
    ax.set_title("Palo Alto Street Network with POIs and Edge Costs", fontsize=16)
    ax.axis('equal')
    plt.tight_layout()
    return fig, ax

def visualize_europe_airports(G):
    # Filter out isolated/near-isolated airports (degree ≤ 1)
    node_degrees = dict(G.degree())
    significant_airports = [n for n, deg in node_degrees.items() if deg > 1]
    G_sub = G.subgraph(significant_airports).copy()
    
    fig, ax = plt.subplots(1, 1, figsize=(20, 15))
    
    # Use 'Lon' and 'Lat' keys; skip nodes missing them
    pos = {}
    for n, data in G_sub.nodes(data=True):
        if 'Lon' in data and 'Lat' in data:
            pos[n] = (data['Lon'], data['Lat'])
    
    nx.draw(G_sub, pos, ax=ax, node_size=50, node_color='red', alpha=0.7)
    
    # Labels
    labels = {}
    for node in pos.keys():
        iata = G_sub.nodes[node].get('IATA', '')
        if iata and iata != "\\N":
            labels[node] = iata
        else:
            labels[node] = G_sub.nodes[node].get('Name', f"Airport_{node}")[:8]
    nx.draw_networkx_labels(G_sub, pos, labels, font_size=6, ax=ax)

    # Edge labels (weights)
    edge_labels = {(u, v): f"{d.get('weight', 0):.0f}" 
                   for u, v, d in G_sub.edges(data=True) if u in pos and v in pos}
    nx.draw_networkx_edge_labels(G_sub, pos, edge_labels=edge_labels, font_size=4, ax=ax, alpha=0.8)

    ax.set_title("European Airports Network (Connected Airports Only)", fontsize=16)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.tight_layout()
    return fig, ax
# ------------------ Random Metadata Samples ------------------
random_poi = random.choice([n for n in G_map.nodes() if G_map.nodes[n].get('is_poi', False)])
print("\nRandom Palo Alto POI Metadata:")
print(G_map.nodes[random_poi])

random_airport = random.choice(list(G_airports.nodes()))
print("\nRandom European Airport Metadata:")
print(G_airports.nodes[random_airport])

# ------------------ Display Plots ------------------
fig1, ax1 = visualize_palo_alto_network(G_map)
plt.show()

fig2, ax2 = visualize_europe_airports(G_airports)
plt.show()