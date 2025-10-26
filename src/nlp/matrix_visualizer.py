import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

def visualize_intent_matrix(matrix, num_nodes, intent_type, save_path=None):
    """
    Visualize as sorted edge list - most space efficient
    """
    if matrix is None:
        print("No negative intent matrix to visualize")
        return
    
    # Get unique paths
    upper_triangle_mask = np.triu(np.ones_like(matrix, dtype=bool), k=1)
    
    # Extract all path info
    paths = []
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            paths.append((i, j, matrix[i][j]))
    
    # Sort by modifier value
    paths.sort(key=lambda x: x[2])
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: All paths sorted (bar chart)
    ax1 = axes[0]
    
    modifiers = [p[2] for p in paths]
    colors = ['red' if m < 0.05 else 'orange' if m < 0.5 else 'yellow' if m < 0.8 else 'green' 
              for m in modifiers]
    
    x_pos = np.arange(len(paths))
    ax1.bar(x_pos, modifiers, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    
    ax1.axhline(0.05, color='red', linestyle='--', linewidth=2, label='Exclusion (0.05)')
    ax1.axhline(0.5, color='orange', linestyle='--', linewidth=1.5, label='Strong avoidance (0.5)')
    ax1.axhline(1.0, color='green', linestyle='--', linewidth=2, label='Neutral (1.0)')
    
    ax1.set_xlabel('Path Index (sorted by modifier)')
    ax1.set_ylabel('Pheromone Modifier')
    ax1.set_title(f'All Paths Sorted by Intent Strength\n({intent_type})', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim([0, 1.1])
    
    # Plot 2: Top/Bottom paths details
    ax2 = axes[1]
    
    # Show top 20 most avoided and top 10 most neutral
    n_show = min(20, len(paths))
    most_avoided = paths[:n_show]
    most_neutral = paths[-10:] if len(paths) > 10 else []
    
    # Combine for display
    display_paths = most_avoided + [('---', '---', None)] + most_neutral[::-1]
    
    y_pos = np.arange(len(display_paths))
    labels = [f"{p[0]}→{p[1]}" if p[2] is not None else "..." for p in display_paths]
    values = [p[2] if p[2] is not None else 0.5 for p in display_paths]
    colors_detail = ['red' if v < 0.05 else 'orange' if v < 0.5 else 'yellow' if v < 0.8 else 'green' 
                     for v in values]
    
    ax2.barh(y_pos, values, color=colors_detail, alpha=0.7, edgecolor='black')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels, fontsize=8)
    ax2.axvline(0.05, color='red', linestyle='--', linewidth=2, alpha=0.5)
    ax2.axvline(0.5, color='orange', linestyle='--', linewidth=1.5, alpha=0.5)
    ax2.axvline(1.0, color='green', linestyle='--', linewidth=2, alpha=0.5)
    ax2.set_xlabel('Pheromone Modifier')
    ax2.set_title(f'Top {n_show} Most Avoided Paths (top) & Top 10 Neutral Paths (bottom)')
    ax2.invert_yaxis()
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Add statistics box
    modifiers_array = np.array([p[2] for p in paths])
    stats_text = f"""
    Total paths: {len(paths)}
    Mean: {np.mean(modifiers_array):.3f}
    Median: {np.median(modifiers_array):.3f}
    
    Excluded (<0.05): {np.sum(modifiers_array < 0.05)} ({100*np.sum(modifiers_array < 0.05)/len(paths):.1f}%)
    Avoided (<0.5): {np.sum(modifiers_array < 0.5)} ({100*np.sum(modifiers_array < 0.5)/len(paths):.1f}%)
    Neutral (≥0.8): {np.sum(modifiers_array >= 0.8)} ({100*np.sum(modifiers_array >= 0.8)/len(paths):.1f}%)
    """
    fig.text(0.98, 0.5, stats_text, transform=fig.transFigure,
             verticalalignment='center', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
             fontsize=10, family='monospace')
    
    plt.tight_layout(rect=[0, 0, 0.88, 1])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Intent matrix visualization saved to {save_path}")
    
    plt.show()

# THIS ONE IS NETWORK ITS ALSO FIRE
# def visualize_intent_matrix(matrix, num_nodes, intent_type, node_positions=None, save_path=None):
#     """
#     Visualize intent matrix as a network graph with colored edges
#     """
#     if matrix is None:
#         print("No negative intent matrix to visualize")
#         return
    
#     fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
#     # Plot 1: Network graph with edge colors
#     ax1 = axes[0]
    
#     # Create graph
#     G = nx.Graph()
#     G.add_nodes_from(range(num_nodes))
    
#     # Get unique paths (upper triangle only)
#     upper_triangle_mask = np.triu(np.ones_like(matrix, dtype=bool), k=1)
#     modifiers = matrix[upper_triangle_mask].flatten()
    
#     # Add edges with weights
#     edge_colors = []
#     edge_widths = []
#     edges = []
    
#     for i in range(num_nodes):
#         for j in range(i+1, num_nodes):
#             modifier = matrix[i][j]
#             G.add_edge(i, j, weight=modifier)
#             edges.append((i, j))
#             edge_colors.append(modifier)
#             # Width based on how much it's avoided (thicker = more avoided)
#             edge_widths.append(1 + 3 * (1 - modifier))
    
#     # Layout
#     if node_positions is None:
#         pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
#     else:
#         pos = node_positions
    
#     # Draw nodes
#     nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
#                            node_size=300, ax=ax1)
#     nx.draw_networkx_labels(G, pos, font_size=8, ax=ax1)
    
#     # Draw edges with color mapping
#     # Green (neutral) -> Yellow (moderate) -> Red (avoided)
#     edges_collection = nx.draw_networkx_edges(
#         G, pos, 
#         edge_color=edge_colors,
#         edge_cmap=plt.cm.RdYlGn,
#         edge_vmin=0, edge_vmax=1,
#         width=edge_widths,
#         ax=ax1
#     )
    
#     # Add colorbar
#     sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn, 
#                                norm=plt.Normalize(vmin=0, vmax=1))
#     sm.set_array([])
#     cbar = plt.colorbar(sm, ax=ax1)
#     cbar.set_label('Pheromone Modifier', rotation=270, labelpad=20)
    
#     ax1.set_title(f'Path Intent Network\n({intent_type})\nThicker edges = more avoided',
#                   fontsize=12, fontweight='bold')
#     ax1.axis('off')
    
#     # Plot 2: Distribution
#     ax2 = axes[1]
    
#     n_bins = min(50, len(modifiers) // 2) if len(modifiers) > 100 else 30
#     ax2.hist(modifiers, bins=n_bins, color='steelblue', alpha=0.7, edgecolor='black')
#     ax2.axvline(0.05, color='red', linestyle='--', linewidth=2, label='Exclusion (0.05)')
#     ax2.axvline(np.mean(modifiers), color='orange', linestyle='--', linewidth=2,
#                 label=f'Mean: {np.mean(modifiers):.3f}')
#     ax2.axvline(1.0, color='green', linestyle='--', linewidth=2, label='Neutral (1.0)')
    
#     ax2.set_xlabel('Pheromone Modifier Value')
#     ax2.set_ylabel('Frequency')
#     ax2.set_title('Distribution of Path Modifiers')
#     ax2.legend()
#     ax2.grid(True, alpha=0.3)
    
#     # Statistics
#     unique_paths = len(modifiers)
#     stats_text = f"""
#     Unique paths: {unique_paths}
#     Mean: {np.mean(modifiers):.3f}
#     Std: {np.std(modifiers):.3f}
    
#     Excluded (<0.05): {np.sum(modifiers < 0.05)}
#     Avoided (<0.5): {np.sum(modifiers < 0.5)}
#     Neutral (≥0.8): {np.sum(modifiers >= 0.8)}
#     """
#     ax2.text(0.98, 0.97, stats_text, transform=ax2.transAxes,
#              verticalalignment='top', horizontalalignment='right',
#              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
#              fontsize=9, family='monospace')
    
#     plt.tight_layout()
    
#     if save_path:
#         plt.savefig(save_path, dpi=300, bbox_inches='tight')
#         print(f"Intent matrix visualization saved to {save_path}")
    
#     plt.show()