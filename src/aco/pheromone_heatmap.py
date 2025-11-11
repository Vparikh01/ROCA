import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def pheromone_heatmap(pheromone_matrices, save_path=None, interval=150, cmap='coolwarm'):
    """
    Animated heatmap of pheromone matrices over iterations.
    """
    if not pheromone_matrices:
        raise ValueError("No pheromone matrices provided.")

    vmin = min(np.min(m) for m in pheromone_matrices)
    vmax = max(np.max(m) for m in pheromone_matrices)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(pheromone_matrices[0], cmap=cmap, interpolation='nearest', vmin=vmin, vmax=vmax)
    ax.set_title("Pheromone Matrix Evolution (Iteration 1)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    def update(frame):
        ax.clear()
        im = ax.imshow(pheromone_matrices[frame], cmap=cmap, interpolation='nearest', vmin=vmin, vmax=vmax)
        ax.set_title(f"Pheromone Matrix Evolution (Iteration {frame + 1})")
        return [im]

    ani = FuncAnimation(fig, update, frames=len(pheromone_matrices), interval=interval, blit=False, repeat=False)

    if save_path:
        if save_path.endswith(".gif"):
            ani.save(save_path, writer='pillow', fps=1000//interval)
        elif save_path.endswith(".mp4"):
            ani.save(save_path, writer='ffmpeg', fps=1000//interval)
        else:
            raise ValueError("Unsupported file type. Use .gif or .mp4.")
    else:
        plt.show()

    plt.close(fig)

def pheromone_composite(pheromone_matrices, cmap='coolwarm', save_path=None):
    """
    Single static heatmap: average pheromone intensity across all iterations.
    """
    avg_matrix = np.mean(np.array(pheromone_matrices), axis=0)
    plt.figure(figsize=(5, 4))
    plt.imshow(avg_matrix, cmap=cmap, interpolation='nearest')
    plt.title("Cumulative Pheromone Intensity")
    plt.colorbar(label="Average Pheromone Strength")
    plt.xticks([]); plt.yticks([])
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()

def pheromone_blend(pheromone_matrices, iterations=(0, 100, -1), save_path=None):
    """
    Overlay early, mid, and late iterations in a single RGB heatmap (blue->red).
    """
    num_iters = len(pheromone_matrices)
    mats = [pheromone_matrices[i if i >= 0 else num_iters + i] for i in iterations]
    vmin = min(np.min(m) for m in mats)
    vmax = max(np.max(m) for m in mats)
    mats = [(m - vmin) / (vmax - vmin) for m in mats]

    blended = np.zeros((*mats[0].shape, 3))
    blended[..., 2] += mats[0]             # Blue = early
    if len(mats) > 2:
        blended[..., 1] += mats[1] * 0.5   # Optional mid = purple
    blended[..., 0] += mats[-1]            # Red = final

    plt.figure(figsize=(5, 4))
    plt.imshow(blended, interpolation='nearest')
    plt.title("Pheromone Convergence (Blue→Red)")
    plt.xticks([]); plt.yticks([])
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()
    """
    1D line plot of mean pheromone value per iteration.
    """
    mean_values = [np.mean(m) for m in pheromone_matrices]
    plt.figure(figsize=(5, 3))
    plt.plot(mean_values, color='red')
    plt.title("Average Pheromone Intensity Over Time")
    plt.xlabel("Iteration")
    plt.ylabel("Mean Pheromone Value")
    plt.grid(True, alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()
