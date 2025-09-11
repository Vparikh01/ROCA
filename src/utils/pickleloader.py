import osmnx as ox
import pandas as pd
import networkx as nx
from geopy.distance import geodesic
import matplotlib.pyplot as plt
import pickle

# ------------------ Loader functions ------------------

def load_palo_alto_drive_with_pois():
    """
    Load Palo Alto road network as a graph, update POI nodes with rich metadata.
    POIs include hospitals, schools, universities, fire stations, and more.
    """
    # Load street network
    G = ox.graph_from_place("Palo Alto, California, USA", network_type="drive")

    # Ensure all edges have 'length'
    for u, v, k, data in G.edges(keys=True, data=True):
        if 'length' not in data:
            x1, y1 = G.nodes[u]['x'], G.nodes[u]['y']
            x2, y2 = G.nodes[v]['x'], G.nodes[v]['y']
            data['length'] = geodesic((y1, x1), (y2, x2)).meters

    # Load POIs
    poi_tags = {
        "amenity": [
            "hospital", "school", "university", "fire_station", "clinic", "pharmacy",
            "police", "townhall", "courthouse", "library", "post_office", "bank", 
            "fuel", "restaurant", "cafe", "fast_food", "bar", "pub", "theatre", 
            "cinema", "community_centre", "social_facility", "place_of_worship",
            "kindergarten", "college", "research_institute", "doctors", "dentist",
            "veterinary", "nursing_home", "retirement_home", "childcare"
        ]
    }

    try:
        pois = ox.features.features_from_place("Palo Alto, California, USA", tags=poi_tags)
        print(f"Found {len(pois)} POI features")

        for idx, row in pois.iterrows():
            if "geometry" not in row or row.geometry.is_empty:
                continue

            if row.geometry.geom_type in ["Polygon", "MultiPolygon"]:
                lat = row.geometry.centroid.y
                lon = row.geometry.centroid.x
            elif row.geometry.geom_type == "Point":
                lat = row.geometry.y
                lon = row.geometry.x
            else:
                continue

            if pd.isna(lat) or pd.isna(lon):
                continue

            try:
                nearest_node = ox.distance.nearest_nodes(G, X=lon, Y=lat)
                
                # Update the existing node directly
                node_attrs = dict(row)
                node_attrs['y'] = lat
                node_attrs['x'] = lon
                node_attrs['is_poi'] = True
                node_attrs['connected_node'] = nearest_node
                if pd.isna(node_attrs.get("name")) or not node_attrs.get("name"):
                    node_attrs["name"] = node_attrs.get("amenity", "unknown").title()
                
                G.nodes[nearest_node].update(node_attrs)

            except Exception as e:
                print(f"Error updating POI {row.get('name', 'unknown')}: {e}")
                continue

    except Exception as e:
        print(f"Warning: Could not load POIs: {e}")

    # Mark remaining nodes as non-POI
    for node in G.nodes():
        if 'is_poi' not in G.nodes[node]:
            G.nodes[node]['is_poi'] = False

    # Project graph for metric calculations
    G = ox.project_graph(G)
    return G


def load_openflights_europe_airports_rich():
    """Load European airports with metadata and geodesic edge weights."""
    airports_url = "https://raw.githubusercontent.com/jpatokal/openflights/master/data/airports.dat"
    routes_url = "https://raw.githubusercontent.com/jpatokal/openflights/master/data/routes.dat"

    cols_airports = ["AirportID", "Name", "City", "Country", "IATA", "ICAO", 
                     "Lat", "Lon", "Alt", "Timezone", "DST", "TzDatabase", "Type", "Source"]
    cols_routes = ["Airline", "AirlineID", "SourceAirport", "SourceID", 
                   "DestAirport", "DestID", "Codeshare", "Stops", "Equipment"]

    airports_df = pd.read_csv(airports_url, header=None, names=cols_airports)
    routes_df = pd.read_csv(routes_url, header=None, names=cols_routes)

    central_south_america_countries_with_airports = [
    # Central America
    "Belize",
    "Costa Rica",
    "El Salvador",
    "Guatemala",
    "Honduras",
    "Nicaragua",
    "Panama",

    # South America
    "Argentina",
    "Bolivia",
    "Brazil",
    "Chile",
    "Colombia",
    "Ecuador",
    "Guyana",
    "Paraguay",
    "Peru",
    "Suriname",
    "Uruguay",
    "Venezuela",

    # Caribbean South American states
    "Aruba",
    "Curacao",
    "Saint Martin",   # Dutch side separate in OpenFlights
    "Bonaire"
    ]
    


    airports_df = airports_df[
    (airports_df["Lat"] >= -60) & (airports_df["Lat"] <= 15)  # Approx latitude bounds for Central/South America
    ]
    airports_df = airports_df[airports_df["Country"].isin(central_south_america_countries_with_airports)]

    G = nx.Graph()
    for _, row in airports_df.iterrows():
        node_id = int(row["AirportID"])
        G.add_node(node_id, **row.to_dict())

    for _, row in routes_df.iterrows():
        try:
            src = int(row["SourceID"])
            dest = int(row["DestID"])
        except:
            continue
        if src in G.nodes and dest in G.nodes:
            src_coords = (G.nodes[src]["Lat"], G.nodes[src]["Lon"])
            dest_coords = (G.nodes[dest]["Lat"], G.nodes[dest]["Lon"])
            dist_km = geodesic(src_coords, dest_coords).km
            G.add_edge(src, dest, weight=dist_km)
    return G


# ------------------ Visualization functions ------------------

def visualize_palo_alto_network(G):
    fig, ax = plt.subplots(1, 1, figsize=(15, 15))
    ox.plot_graph(G, ax=ax, node_size=0, edge_color='lightgray', edge_linewidth=0.5, show=False, close=False)
    
    poi_nodes = [n for n in G.nodes() if G.nodes[n].get('is_poi', False)]
    for poi in poi_nodes:
        x, y = G.nodes[poi]['x'], G.nodes[poi]['y']
        ax.scatter(x, y, s=100, c='red', zorder=3, alpha=0.8)
        ax.annotate(str(G.nodes[poi].get('name', 'Unknown')), (x, y), xytext=(5, 5),
                    textcoords='offset points', fontsize=8, ha='left', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    edge_labels = {(u, v): f"{int(d['length'])}m" for u, v, d in G.edges(data=True) if d.get('length', 0) > 0}
    pos = {n: (data['x'], data['y']) for n, data in G.nodes(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=4, ax=ax, alpha=0.7)
    
    ax.set_title("Palo Alto Street Network with POIs and Edge Costs", fontsize=16)
    ax.axis('equal')
    plt.tight_layout()
    return fig, ax


def visualize_europe_airports(G):
    node_degrees = dict(G.degree())
    significant_airports = [node for node, degree in node_degrees.items() if degree > 1]
    G_sub = G.subgraph(significant_airports).copy()
    
    fig, ax = plt.subplots(1, 1, figsize=(20, 15))
    pos = {node: (G_sub.nodes[node]['Lon'], G_sub.nodes[node]['Lat']) for node in G_sub.nodes()}
    
    nx.draw_networkx_nodes(G_sub, pos, node_size=50, node_color='red', alpha=0.7, ax=ax)
    nx.draw_networkx_edges(G_sub, pos, alpha=0.2, width=0.5, ax=ax)
    
    labels = {}
    for node in G_sub.nodes():
        iata = G_sub.nodes[node].get('IATA', '')
        if iata and iata != "\\N":
            labels[node] = iata
        else:
            name = G_sub.nodes[node].get('Name', f"Airport_{node}")
            labels[node] = name[:8]
    
    nx.draw_networkx_labels(G_sub, pos, labels, font_size=6, ax=ax)
    edge_labels = {(u, v): f"{d['weight']:.0f}" for u, v, d in G_sub.edges(data=True)}
    nx.draw_networkx_edge_labels(G_sub, pos, edge_labels=edge_labels, font_size=4, ax=ax, alpha=0.8)
    
    ax.set_title("European Airports Network", fontsize=16)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.tight_layout()
    return fig, ax


# ------------------ Main ------------------

if __name__ == "__main__":
    # print("Loading Palo Alto street network...")
    # G_map = load_palo_alto_drive_with_pois()
    # print(f"Palo Alto graph: {len(G_map.nodes)} nodes, {len(G_map.edges)} edges")

    # print("Saving Palo Alto graph to disk...")
    # with open("palo_alto_graph.pkl", "wb") as f:
    #     pickle.dump(G_map, f)

    print("Loading European airports network...")
    G_airports = load_openflights_europe_airports_rich()
    print(f"European airports graph: {len(G_airports.nodes)} nodes, {len(G_airports.edges)} edges")

    print("Saving European airports graph to disk...")
    with open("europe_airports_graph.pkl", "wb") as f:
        pickle.dump(G_airports, f)

    # print("\nCreating visualizations...")
    # fig1, ax1 = visualize_palo_alto_network(G_map)
    # plt.figure(fig1.number)
    # plt.show()

    fig2, ax2 = visualize_europe_airports(G_airports)
    plt.figure(fig2.number)
    plt.show()

    print("Visualizations complete!")