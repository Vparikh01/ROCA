import pickle
import networkx as nx
import plotly.graph_objects as go
import heapq
import os

# ------------------ Paths ------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PALO_ALTO_PKL = os.path.join(BASE_DIR, "..", "datasets", "palo_alto_graph.pkl")
EURO_AIRPORT_PKL = os.path.join(BASE_DIR, "..", "datasets", "europe_airports_graph.pkl")

# ---------- Load Graph ----------
def load_graph(pkl_path):
    with open(pkl_path, "rb") as f:
        G = pickle.load(f)
    if G.is_directed():
        G = G.to_undirected()
    return G

G = load_graph(PALO_ALTO_PKL)

# Filter out very low-degree nodes (helps with both graph types)
# Street networks: removes dead ends and isolated segments
# Airport networks: removes isolated airports
node_degrees = dict(G.degree())
significant_nodes = [n for n, deg in node_degrees.items() if deg > 1]
if len(significant_nodes) > 10:  # Only filter if we have enough nodes left
    G = G.subgraph(significant_nodes).copy()

# Universal coordinate extraction - try all possible coordinate keys
pos = {}
for n, data in G.nodes(data=True):
    coords = None
    
    # Try different coordinate key combinations
    if 'x' in data and 'y' in data:
        coords = (data['x'], data['y'])
    elif 'Lon' in data and 'Lat' in data:
        coords = (data['Lon'], data['Lat'])
    elif 'longitude' in data and 'latitude' in data:
        coords = (data['longitude'], data['latitude'])
    elif 'lon' in data and 'lat' in data:
        coords = (data['lon'], data['lat'])
    
    if coords:
        pos[n] = coords

# If no coordinates found, use spring layout
if not pos:
    pos = nx.spring_layout(G, seed=42)

# Pick start/end nodes randomly from positioned nodes
import random
nodes_list = list(pos.keys())
start = random.choice(nodes_list)
end = random.choice([n for n in nodes_list if n != start])

print(f"Visualizing path from {start} to {end}")

# ---------- Dijkstra Step Generator ----------
def dijkstra_steps(G, start, end):
    """Yield step states with minimal information changes."""
    dist = {n: float("inf") for n in G.nodes}
    dist[start] = 0.0
    pq = [(0.0, start)]
    parent = {start: None}
    visited = set()
    
    step_count = 0
    
    while pq:
        d, u = heapq.heappop(pq)
        if u in visited:
            continue
            
        visited.add(u)
        step_count += 1
        
        # Current node expansion
        yield {
            "step": step_count,
            "event": "visit",
            "current": u,
            "visited": frozenset(visited),
            "distances": dict(dist),
            "parent": dict(parent),
            "relaxed_edges": []
        }
        
        if u == end:
            break
            
        # Collect all edge relaxations for this node in one step
        relaxed_edges = []
        for v in G.neighbors(u):
            if v in pos:  # Only consider positioned nodes
                # Try different edge weight keys in order of preference
                w = (G[u][v].get("weight") or 
                     G[u][v].get("length") or 
                     G[u][v].get("distance") or 
                     G[u][v].get("cost") or 
                     1.0)
                nd = dist[u] + float(w)
                if nd < dist[v]:
                    dist[v] = nd
                    parent[v] = u
                    heapq.heappush(pq, (nd, v))
                    relaxed_edges.append((u, v))
        
        if relaxed_edges:
            step_count += 1
            yield {
                "step": step_count,
                "event": "relax",
                "current": u,
                "visited": frozenset(visited),
                "distances": dict(dist),
                "parent": dict(parent),
                "relaxed_edges": relaxed_edges
            }

# ---------- Create Base Graph (Static) ----------
def create_base_graph():
    # Create base edges (static throughout animation)
    edge_x, edge_y = [], []
    for u, v in G.edges():
        if u in pos and v in pos:
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='lightgray'),
        mode="lines",
        hoverinfo="none",
        name="Graph Edges",
        showlegend=False
    )
    
    return edge_trace

# ---------- Create Node Traces ----------
def create_node_traces(current=None, visited=frozenset(), start_node=None, end_node=None):
    # Separate traces for different node types for better control
    unvisited_x, unvisited_y, unvisited_text = [], [], []
    visited_x, visited_y, visited_text = [], [], []
    current_x, current_y, current_text = [], [], []
    start_x, start_y, start_text = [], [], []
    end_x, end_y, end_text = [], [], []
    
    for n in pos.keys():
        x, y = pos[n]
        # Universal label creation - try multiple label sources
        text = (G.nodes[n].get('IATA') if G.nodes[n].get('IATA') and G.nodes[n].get('IATA') != "\\N" else
                G.nodes[n].get('name') or
                G.nodes[n].get('Name') or
                str(n))
        
        # Truncate long labels
        if len(str(text)) > 8:
            text = str(text)[:8]
        
        if n == start_node:
            start_x.append(x)
            start_y.append(y)
            start_text.append(text)
        elif n == end_node:
            end_x.append(x)
            end_y.append(y)
            end_text.append(text)
        elif n == current:
            current_x.append(x)
            current_y.append(y)
            current_text.append(text)
        elif n in visited:
            visited_x.append(x)
            visited_y.append(y)
            visited_text.append(text)
        else:
            unvisited_x.append(x)
            unvisited_y.append(y)
            unvisited_text.append(text)
    
    traces = []
    
    # Start node (green)
    if start_x:
        traces.append(go.Scatter(
            x=start_x, y=start_y, text=start_text,
            mode="markers+text", textposition="top center",
            marker=dict(size=15, color='green', line=dict(width=2, color='darkgreen')),
            name="Start", showlegend=True
        ))
    
    # End node (red)
    if end_x:
        traces.append(go.Scatter(
            x=end_x, y=end_y, text=end_text,
            mode="markers+text", textposition="top center",
            marker=dict(size=15, color='red', line=dict(width=2, color='darkred')),
            name="Target", showlegend=True
        ))
    
    # Current node (yellow)
    if current_x:
        traces.append(go.Scatter(
            x=current_x, y=current_y, text=current_text,
            mode="markers+text", textposition="top center",
            marker=dict(size=12, color='yellow', line=dict(width=2, color='orange')),
            name="Current", showlegend=True
        ))
    
    # Visited nodes (blue)
    if visited_x:
        traces.append(go.Scatter(
            x=visited_x, y=visited_y, text=visited_text,
            mode="markers+text", textposition="top center",
            marker=dict(size=8, color='lightblue', line=dict(width=1, color='blue')),
            name="Visited", showlegend=True
        ))
    
    # Unvisited nodes (gray)
    if unvisited_x:
        traces.append(go.Scatter(
            x=unvisited_x, y=unvisited_y, text=unvisited_text,
            mode="markers+text", textposition="top center",
            marker=dict(size=6, color='lightgray', line=dict(width=1, color='gray')),
            name="Unvisited", showlegend=True
        ))
    
    return traces

# ---------- Create Path Trace ----------
def create_path_trace(parent, start, end):
    if end not in parent:
        return None
    
    path = []
    cur = end
    while cur is not None:
        path.append(cur)
        cur = parent.get(cur)
    path.reverse()
    
    if not path or path[0] != start:
        return None
    
    x, y = [], []
    for i in range(len(path) - 1):
        if path[i] in pos and path[i+1] in pos:
            x0, y0 = pos[path[i]]
            x1, y1 = pos[path[i+1]]
            x += [x0, x1, None]
            y += [y0, y1, None]
    
    return go.Scatter(
        x=x, y=y,
        line=dict(width=6, color='red'),
        mode="lines",
        name="Shortest Path",
        showlegend=True
    )

# ---------- Create Relaxed Edges Trace ----------
def create_relaxed_edges_trace(relaxed_edges):
    if not relaxed_edges:
        return None
    
    x, y = [], []
    for u, v in relaxed_edges:
        if u in pos and v in pos:
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            x += [x0, x1, None]
            y += [y0, y1, None]
    
    return go.Scatter(
        x=x, y=y,
        line=dict(width=3, color='dodgerblue'),
        mode="lines",
        name="Relaxed Edges",
        showlegend=True
    )

# ---------- Build Animation Frames ----------
frames = []
steps_data = list(dijkstra_steps(G, start, end))

base_edge_trace = create_base_graph()
final_parent = {}

for idx, step in enumerate(steps_data):
    current = step.get("current")
    visited = step.get("visited", frozenset())
    parent = step.get("parent", {})
    relaxed_edges = step.get("relaxed_edges", [])
    final_parent = parent  # Keep updating to get the final parent mapping
    
    # Build frame data
    frame_data = [base_edge_trace]
    
    # Add relaxed edges if any
    relaxed_trace = create_relaxed_edges_trace(relaxed_edges)
    if relaxed_trace:
        frame_data.append(relaxed_trace)
    
    # Add nodes
    node_traces = create_node_traces(
        current=current, 
        visited=visited, 
        start_node=start, 
        end_node=end
    )
    frame_data.extend(node_traces)
    
    frames.append(go.Frame(
        name=str(idx),
        data=frame_data
    ))

# Add final frame with the complete shortest path highlighted
if steps_data and final_parent:
    final_step = steps_data[-1]
    final_visited = final_step.get("visited", frozenset())
    
    # Final frame data
    final_frame_data = [base_edge_trace]
    
    # Add the shortest path
    path_trace = create_path_trace(final_parent, start, end)
    if path_trace:
        final_frame_data.append(path_trace)
    
    # Add final node traces (no current node, just visited/unvisited)
    final_node_traces = create_node_traces(
        current=None, 
        visited=final_visited, 
        start_node=start, 
        end_node=end
    )
    final_frame_data.extend(final_node_traces)
    
    frames.append(go.Frame(
        name=str(len(frames)),
        data=final_frame_data
    ))

# ---------- Initial Frame Data ----------
if frames:
    initial_data = frames[0].data
else:
    initial_data = [base_edge_trace] + create_node_traces(start_node=start, end_node=end)

# ---------- Create Figure ----------
fig = go.Figure(
    data=initial_data,
    layout=go.Layout(
        title=f"Dijkstra's Algorithm: {start} → {end}",
        showlegend=True,
        hovermode="closest",
        margin=dict(b=50, l=50, r=50, t=80),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='white',
        updatemenus=[{
            "buttons": [
                {
                    "args": [None, {
                        "frame": {"duration": 1000, "redraw": False},
                        "fromcurrent": True,
                        "transition": {"duration": 200}
                    }],
                    "label": "▶ Play",
                    "method": "animate"
                },
                {
                    "args": [[None], {
                        "frame": {"duration": 0, "redraw": False},
                        "mode": "immediate",
                        "transition": {"duration": 0}
                    }],
                    "label": "⏸ Pause",
                    "method": "animate"
                }
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 10},
            "showactive": True,
            "type": "buttons",
            "x": 0.1,
            "xanchor": "right",
            "y": 1.02,
            "yanchor": "top"
        }],
        sliders=[{
            "steps": [
                {
                    "args": [[f.name], {
                        "frame": {"duration": 0, "redraw": False},
                        "mode": "immediate",
                        "transition": {"duration": 100}
                    }],
                    "label": f"Step {i+1}",
                    "method": "animate"
                }
                for i, f in enumerate(frames)
            ],
            "active": 0,
            "x": 0.1,
            "len": 0.85,
            "y": 0,
            "yanchor": "top",
            "currentvalue": {"prefix": "Step: ", "visible": True}
        }] if frames else []
    ),
    frames=frames
)

# Adaptive layout - use equal aspect ratio for small coordinate ranges, natural for large
coord_x = [pos[n][0] for n in pos]
coord_y = [pos[n][1] for n in pos]
x_range = max(coord_x) - min(coord_x) if coord_x else 1
y_range = max(coord_y) - min(coord_y) if coord_y else 1

# If coordinates are in a reasonable range (like street networks), use equal aspect
# If coordinates are geographic (like lat/lon), use natural scaling
if x_range < 1000 and y_range < 1000:
    # Likely street network coordinates
    fig.update_layout(
        xaxis=dict(scaleanchor="y", scaleratio=1),
        yaxis=dict(scaleanchor="x", scaleratio=1)
    )
else:
    # Likely geographic coordinates or very large street network
    fig.update_layout(height=600, width=900)

fig.show()