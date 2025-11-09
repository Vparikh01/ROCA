# ----------------- visualizer.py -----------------
import plotly.graph_objects as go

def draw_aco_graph(
    G, pos, iteration_paths, final_best_path=None,
    start=None, end=None, required_nodes=None, edge_weights=None,
    optimal_path=None, added_nodes=None, removed_nodes=None
):
    """
    Visualizes ACO iterative paths on a graph using Plotly.
    Highlights required nodes, shows reduced graph edge costs,
    final best ACO path, optimal path, and added/removed nodes.
    """
    # ---------- Base edges (full graph) ----------
    edge_x, edge_y = [], []
    for u, v in G.edges():
        if u in pos and v in pos:
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]

    base_edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='lightgray'),
        mode="lines",
        hoverinfo="none",
        name="Graph Edges",
        showlegend=False
    )

    # ---------- Required nodes ----------
    req_trace = None
    if required_nodes:
        req_x = [pos[n][0] for n in required_nodes]
        req_y = [pos[n][1] for n in required_nodes]
        req_trace = go.Scatter(
            x=req_x, y=req_y,
            mode="markers+text",
            text=[str(n) for n in required_nodes],
            textposition="top center",
            marker=dict(size=12, color='blue', line=dict(width=2, color='darkblue')),
            name="Required Nodes",
            showlegend=True
        )

    # ---------- Highlight removed nodes (red) ----------
    removed_trace = None
    if removed_nodes:
        rem_x = [pos[n][0] for n in removed_nodes if n in pos]
        rem_y = [pos[n][1] for n in removed_nodes if n in pos]
        removed_trace = go.Scatter(
            x=rem_x, y=rem_y,
            mode="markers",
            marker=dict(size=14, color='red', symbol='x'),
            name="Excluded Nodes",
            showlegend=True
        )

    # ---------- Highlight added nodes (yellow) ----------
    added_trace = None
    if added_nodes:
        add_x = [pos[n][0] for n in added_nodes if n in pos]
        add_y = [pos[n][1] for n in added_nodes if n in pos]
        added_trace = go.Scatter(
            x=add_x, y=add_y,
            mode="markers",
            marker=dict(size=14, color='yellow', symbol='star'),
            name="Included Nodes",
            showlegend=True
        )

    # ---------- Frames for iterations ----------
    frames = []
    for idx, path in enumerate(iteration_paths):
        x, y = [], []
        for k in range(len(path)-1):
            if path[k] in pos and path[k+1] in pos:
                x0, y0 = pos[path[k]]
                x1, y1 = pos[path[k+1]]
                x += [x0, x1, None]
                y += [y0, y1, None]
        path_trace = go.Scatter(
            x=x, y=y,
            line=dict(width=4, color='orange'),
            mode="lines",
            name=f"Iteration {idx+1}",
            showlegend=True
        )
        data = [base_edge_trace, path_trace]
        if req_trace:
            data.append(req_trace)
        if removed_trace:
            data.append(removed_trace)
        if added_trace:
            data.append(added_trace)
        frames.append(go.Frame(name=str(idx), data=data))

    # ---------- Final best path ----------
    final_frame_data = [base_edge_trace]
    if final_best_path:
        x, y = [], []
        for k in range(len(final_best_path)-1):
            if final_best_path[k] in pos and final_best_path[k+1] in pos:
                x0, y0 = pos[final_best_path[k]]
                x1, y1 = pos[final_best_path[k+1]]
                x += [x0, x1, None]
                y += [y0, y1, None]
        final_path_trace = go.Scatter(
            x=x, y=y,
            line=dict(width=6, color='red'),
            mode="lines",
            name="Best ACO Path"
        )
        final_frame_data.append(final_path_trace)

    # ---------- Optimal path ----------
    if optimal_path:
        x_opt = [pos[n][0] for n in optimal_path]
        y_opt = [pos[n][1] for n in optimal_path]
        optimal_trace = go.Scatter(
            x=x_opt, y=y_opt,
            line=dict(width=4, color='green', dash='dash'),
            mode="lines",
            name="Optimal Path"
        )
        final_frame_data.append(optimal_trace)

    if req_trace:
        final_frame_data.append(req_trace)
    if removed_trace:
        final_frame_data.append(removed_trace)
    if added_trace:
        final_frame_data.append(added_trace)
    frames.append(go.Frame(name="final", data=final_frame_data))

    # ---------- Initial Figure ----------
    initial_data = frames[0].data if frames else [base_edge_trace]

    fig = go.Figure(
        data=initial_data,
        layout=go.Layout(
            title="ACO: Iterative Best Paths vs Optimal",
            showlegend=True,
            hovermode="closest",
            margin=dict(b=50, l=50, r=50, t=80),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white',
            updatemenus=[{
                "buttons": [
                    {"args": [None, {"frame": {"duration": 1000, "redraw": False},
                                     "fromcurrent": True, "transition": {"duration": 200}}],
                     "label": "▶ Play", "method": "animate"},
                    {"args": [[None], {"frame": {"duration": 0, "redraw": False},
                                       "mode": "immediate",
                                       "transition": {"duration": 0}}],
                     "label": "⏸ Pause", "method": "animate"}
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
                    {"args": [[f.name], {"frame": {"duration": 0, "redraw": False},
                                         "mode": "immediate",
                                         "transition": {"duration": 100}}],
                     "label": f"Iter {i+1}",
                     "method": "animate"} for i, f in enumerate(frames)
                ],
                "active": 0,
                "x": 0.1,
                "len": 0.85,
                "y": 0,
                "yanchor": "top",
                "currentvalue": {"prefix": "Iteration: ", "visible": True}
            }] if frames else []
        ),
        frames=frames
    )

    fig.show()
