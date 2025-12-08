import os
import pickle
import tempfile
import shutil
import subprocess
import numpy as np
import networkx as nx
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.aco.engine import MaxMinACO
import warnings
from typing import Tuple, List, Optional
import plotly.io as pio

# ------------------ USER TUNABLES ------------------
NUM_RUNS = 2
I_MAX = 200
T_CHANGE = 80               # iteration where weight modification happens
P_VALUES = [5, 10, 20]      # percent improvements to target (relative to original cost)
FOUND_ALLOWANCE = 1.12      # route ≤ 112% of new optimal
SEEDS = list(range(1, NUM_RUNS + 1))
EXPECTED_FACTOR = 1.75      # expected allowed iterations = round(1.75 * P)
MIN_ITER = 1

# How many top edges by length to consider when searching (None = all)
TOP_K_EDGES = 200
# How many iterations for multiplicative binary search per chosen edge
BINARY_ITERS = 25
# Quick test: set candidate edge to zero first to compute upper bound; cheaper to skip impossible edges
ZERO_WEIGHT_FILTER = True

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GRAPH_PATH = os.path.join(BASE_DIR, "..", "tsplib_graphs", "eil51.pkl")

# Known optimal tour for eil51 (0-indexed) and cost (from your earlier context)
KNOWN_OPTIMAL_TOUR = [
    0, 21, 7, 25, 30, 27, 2, 35, 34, 19, 1, 28, 20, 15, 49, 33, 29, 8, 48, 9,
    38, 32, 44, 14, 43, 41, 39, 18, 40, 12, 24, 13, 23, 42, 6, 22, 47, 5, 26,
    50, 45, 11, 46, 17, 3, 16, 36, 4, 37, 10, 31
]
KNOWN_OPTIMAL_COST = 426

CONCORDE_BINARY = os.environ.get("CONCORDE_BIN", "concorde")
USE_PYCONCORDE = True
VERBOSE = False
pio.renderers.default = "browser"


# ------------------ I/O and Concorde wrappers ------------------


def load_graph(pkl_path: str):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    coords = np.array(data.get("coordinates", data.get("coords", None)))
    if coords is None:
        if isinstance(data, nx.Graph):
            coords_list = []
            nodes = sorted(data.nodes())
            for n in nodes:
                c = data.nodes[n].get("coord") or data.nodes[n].get("coords") or data.nodes[n].get("coordinate")
                if c is None:
                    raise RuntimeError("Could not find node coordinate attributes in the pickle.")
                coords_list.append(tuple(c))
            coords = np.array(coords_list)
        else:
            raise RuntimeError("Cannot extract coordinates from the provided pickle.")
    n = len(coords)

    G = nx.Graph()
    for i in range(n):
        for j in range(i + 1, n):
            w = float(np.round(np.linalg.norm(coords[i] - coords[j])))
            G.add_edge(i, j, weight=w)

    return G, coords


def compute_tour_cost(tour: List[int], cost_matrix: np.ndarray) -> float:
    n = len(tour)
    return sum(cost_matrix[tour[i]][tour[(i + 1) % n]] for i in range(n))


def write_explicit_tsp(cost_matrix: np.ndarray, path: str, name: str = "modified"):
    n = cost_matrix.shape[0]
    with open(path, "w") as f:
        f.write(f"NAME: {name}\nTYPE: TSP\nDIMENSION: {n}\n")
        f.write("EDGE_WEIGHT_TYPE: EXPLICIT\nEDGE_WEIGHT_FORMAT: FULL_MATRIX\nEDGE_WEIGHT_SECTION\n")
        for i in range(n):
            row = " ".join(str(int(round(cost_matrix[i, j]))) for j in range(n))
            f.write(row + "\n")
        f.write("EOF\n")


def run_concorde_on_tspfile(tsp_path: str, work_dir: str) -> Tuple[List[int], Optional[float]]:
    if USE_PYCONCORDE:
        try:
            from concorde.tsp import TSPSolver  # type: ignore
            solver = TSPSolver.from_tspfile(tsp_path)
            sol = solver.solve()
            tour = list(sol.tour)
            cost = float(sol.tour_length) if hasattr(sol, "tour_length") else None
            return tour, cost
        except Exception:
            pass

    try:
        with open(os.devnull, "w") as fnull:
            subprocess.check_call([CONCORDE_BINARY, tsp_path], cwd=work_dir, stdout=fnull, stderr=fnull)
    except FileNotFoundError as e:
        raise RuntimeError(f"Concorde binary not found at '{CONCORDE_BINARY}'.") from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Concorde returned non-zero exit status: {e}")

    base = os.path.splitext(os.path.basename(tsp_path))[0]
    sol_file = next((os.path.join(work_dir, f) for f in os.listdir(work_dir) if f.startswith(base) and f.endswith(".sol")), None)
    if not sol_file:
        raise RuntimeError("Concorde did not produce a .sol file (expected).")

    tour, cost = [], None
    with open(sol_file, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    for ln in lines:
        if ln.upper().startswith("TOUR_SECTION") or ln.upper().startswith("EOF"):
            continue
        try:
            v = int(ln)
            if v == -1:
                continue
            tour.append(v - 1)
        except ValueError:
            continue
    return tour, cost


def compute_optimal_tour_via_concorde_explicit(cost_matrix: np.ndarray) -> Tuple[List[int], float]:
    tempdir = tempfile.mkdtemp(prefix="tmp_concorde_")
    try:
        tsp_path = os.path.join(tempdir, "modified_explicit.tsp")
        write_explicit_tsp(cost_matrix, tsp_path)
        tour, cost = run_concorde_on_tspfile(tsp_path, tempdir)
        if cost is None:
            cost = compute_tour_cost(tour, cost_matrix)
        return tour, cost
    finally:
        shutil.rmtree(tempdir)


# ------------------ helper functions for single-edge evaluation ------------------


def build_base_cost_matrix(coords: np.ndarray) -> np.ndarray:
    n = len(coords)
    cm = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                cm[i, j] = float(np.round(np.linalg.norm(coords[i] - coords[j])))
    return cm


def candidate_edge_list(cost_matrix: np.ndarray, top_k: Optional[int] = TOP_K_EDGES) -> List[Tuple[int, int, float]]:
    n = cost_matrix.shape[0]
    candidates = [(i, j, cost_matrix[i, j]) for i in range(n) for j in range(i + 1, n)]
    candidates.sort(key=lambda x: x[2], reverse=True)
    if top_k is not None and top_k > 0:
        candidates = candidates[:top_k]
    return candidates


def evaluate_edge_zero_bound(cost_matrix: np.ndarray, u: int, v: int) -> Optional[float]:
    """
    Return the optimal tour cost when (u,v) is set to zero, or None if Concorde fails.
    """
    tmp = cost_matrix.copy()
    tmp[u, v] = tmp[v, u] = 0.0
    try:
        _, cost_if_zero = compute_optimal_tour_via_concorde_explicit(tmp)
    except Exception:
        return None
    return cost_if_zero


def binary_search_weight_for_target(cost_matrix: np.ndarray, u: int, v: int,
                                   original_cost: float, target_pct_from_original: float,
                                   original_edge_weight: float,
                                   binary_iters: int = BINARY_ITERS) -> Tuple[float, List[int], float]:
    """
    Multiplicative binary search on keep factor in (0,1] to find a weight for edge (u,v)
    that achieves (or approximates) target_pct_from_original improvement relative to original_cost.
    Returns: (new_weight, tour, new_cost)
    """
    low = 0.0
    high = 1.0
    best = None  # tuple (new_weight, tour, new_cost, actual_pct)

    for _ in range(binary_iters):
        mid = (low + high) / 2.0
        w_try = max(1e-12, original_edge_weight * mid)
        tmp = cost_matrix.copy()
        tmp[u, v] = tmp[v, u] = w_try
        try:
            tour_try, cost_try = compute_optimal_tour_via_concorde_explicit(tmp)
        except Exception:
            # if Concorde fails (rare), favor smaller weight
            high = mid
            continue

        actual_pct = (original_cost - cost_try) / original_cost * 100.0
        if best is None or abs(actual_pct - target_pct_from_original) < abs(best[3] - target_pct_from_original):
            best = (w_try, tour_try, cost_try, actual_pct)

        # if we overachieved (actual_pct > target), increase weight (less improvement)
        if actual_pct > target_pct_from_original:
            low = mid
        else:
            high = mid

        if abs(actual_pct - target_pct_from_original) <= 0.01:
            break

    if best is None:
        raise RuntimeError("binary_search_weight_for_target failed for edge ({},{})".format(u, v))
    return best[0], best[1], best[2]


# ------------------ core: build multi-edge modification sequence ------------------


def build_modification_sequence_for_target(base_cost_matrix: np.ndarray,
                                           original_cost: float,
                                           target_pct: float,
                                           coords: np.ndarray) -> Tuple[List[Tuple[Tuple[int, int], float, float]], np.ndarray, float]:
    """
    Build a sequence of single-edge modifications (applied cumulatively) that reduces the
    tour cost from original_cost down to (1 - target_pct/100) * original_cost, if possible.

    Returns:
      - list of ( (u,v), new_weight, achieved_pct_after_applying ) in the order applied
      - final_cost_matrix (copy of base_cost_matrix with modifications applied)
      - achieved_pct_final (percentage improvement achieved relative to original)
    """
    cm = base_cost_matrix.copy()
    achieved_pct = 0.0
    target_cost = original_cost * (1 - target_pct / 100.0)
    sequence: List[Tuple[Tuple[int, int], float, float]] = []

    # candidate list (by length)
    candidates = candidate_edge_list(base_cost_matrix, top_k=TOP_K_EDGES)

    # loop until we hit target or cannot improve further
    while True:
        # compute current optimal tour cost on current cm
        try:
            _, current_cost = compute_optimal_tour_via_concorde_explicit(cm)
        except Exception:
            # If concorde fails on current matrix (unlikely), break
            if VERBOSE:
                print("Concorde failed on current matrix during building sequence; stopping.")
            break

        achieved_pct = (original_cost - current_cost) / original_cost * 100.0
        remaining_pct = target_pct - achieved_pct
        if VERBOSE:
            print(f"[build_seq] current achieved {achieved_pct:.6f}% ; remaining {remaining_pct:.6f}%")

        if achieved_pct >= target_pct - 1e-9:
            # target reached
            break

        best_candidate = None  # (u,v, max_possible_pct, cost_if_zero, original_w)
        # evaluate candidates' zero-weight upper bound (cheap filter)
        for u, v, orig_w in candidates:
            if (u, v) in [(e[0][0], e[0][1]) for e in sequence]:
                # already modified this edge in sequence; skip it
                continue
            # set (u,v) to zero in the current cm to test upper bound
            tmp = cm.copy()
            tmp[u, v] = tmp[v, u] = 0.0
            try:
                _, cost_if_zero = compute_optimal_tour_via_concorde_explicit(tmp)
            except Exception:
                continue
            max_possible_pct = (original_cost - cost_if_zero) / original_cost * 100.0
            # want candidate that reduces the current_cost as much as possible (equivalently maximize max_possible_pct)
            if best_candidate is None or max_possible_pct > best_candidate[2]:
                best_candidate = (u, v, max_possible_pct, cost_if_zero, orig_w)

        if best_candidate is None:
            # no more improving edge found
            if VERBOSE:
                print("No improving candidate found in this iteration; stopping.")
            break

        u_best, v_best, max_possible_pct, cost_if_zero, orig_w = best_candidate
        if VERBOSE:
            print(f"Chosen candidate edge ({u_best},{v_best}) with max possible {max_possible_pct:.6f}% (zero-weight)")
        # If adding this edge (to zero) still doesn't change achieved_pct (i.e., no improvement), break
        if max_possible_pct <= achieved_pct + 1e-9:
            if VERBOSE:
                print("Best candidate does not increase total improvement; stopping.")
            break

        # If the candidate's max_possible reaches or exceeds target_pct, compute the precise weight to reach target
        if max_possible_pct >= target_pct:
            # we want to find weight on (u_best,v_best) so that final improvement >= target_pct
            # binary search weight on the current cm (since prior modifications are already in cm)
            try:
                new_w, new_tour, new_cost = binary_search_weight_for_target(
                    cm, u_best, v_best, original_cost, target_pct, orig_w, binary_iters=BINARY_ITERS
                )
                achieved_pct_after = (original_cost - new_cost) / original_cost * 100.0
                # apply this exact weight to cm and record
                cm[u_best, v_best] = cm[v_best, u_best] = new_w
                sequence.append(((u_best, v_best), new_w, achieved_pct_after))
                if VERBOSE:
                    print(f"Applied edge ({u_best},{v_best}) weight {new_w:.6e} -> achieved {achieved_pct_after:.6f}% (target reached or approximated)")
            except Exception:
                # fallback: apply zero weight
                cm[u_best, v_best] = cm[v_best, u_best] = 0.0
                achieved_pct_after = (original_cost - cost_if_zero) / original_cost * 100.0
                sequence.append(((u_best, v_best), 0.0, achieved_pct_after))
                if VERBOSE:
                    print(f"Binary search failed; applied zero weight to ({u_best},{v_best}) -> achieved {achieved_pct_after:.6f}%")
            break  # target reached or best attempt applied

        else:
            # Candidate can't alone reach target: apply maximal modification (set to zero) and continue loop
            cm[u_best, v_best] = cm[v_best, u_best] = 0.0
            achieved_pct_after = (original_cost - cost_if_zero) / original_cost * 100.0
            sequence.append(((u_best, v_best), 0.0, achieved_pct_after))
            if VERBOSE:
                print(f"Applied zero to ({u_best},{v_best}); achieved now {achieved_pct_after:.6f}%. continuing...")

            # loop continues until target reached or no more improvements
            # Note: candidates list remains same top_k; future iterations will skip already modified edges

    # final recompute achieved_pct and final cm
    try:
        _, final_cost = compute_optimal_tour_via_concorde_explicit(cm)
        final_achieved_pct = (original_cost - final_cost) / original_cost * 100.0
    except Exception:
        final_achieved_pct = (original_cost - KNOWN_OPTIMAL_COST) / original_cost * 100.0  # fallback; shouldn't happen

    return sequence, cm, final_achieved_pct


# ------------------ run ACO with multiple edge modifications applied at T_CHANGE ------------------


def run_single_with_modifications(G: nx.Graph, coords: np.ndarray, seed: int,
                                  modifications: List[Tuple[Tuple[int, int], float]] ,
                                  new_optimal_cost: float):
    """
    modifications: list of ((u,v), new_weight) to apply at T_CHANGE (all applied together)
    """
    n = len(coords)
    cost_matrix = build_base_cost_matrix(coords)

    aco = MaxMinACO(
        cost_matrix,
        start_node=0,
        reducedGraph=G,
        completeGraph=G,
        shortest_paths={},
        required_nodes=list(range(n)),
        index_map={i: i for i in range(n)},
        seed=seed,
    )

    global_best = np.inf
    global_best_costs = []
    recorded_t_found = None

    for it in range(I_MAX):
        aco.run(iterations=1, n=it)

        # collect best among ants
        iter_best = np.inf
        for ant in getattr(aco, "ants", []):
            if ant is None or len(getattr(ant, "tour", [])) != n:
                continue
            ant_cost = compute_tour_cost(ant.tour, cost_matrix)
            iter_best = min(iter_best, ant_cost)

        if iter_best < global_best:
            global_best = iter_best
        global_best_costs.append(global_best)

        if it == T_CHANGE:
            # apply all modifications to the running cost_matrix and graph
            for (u, v), new_w in modifications:
                cost_matrix[u, v] = cost_matrix[v, u] = new_w
                if G.has_edge(u, v):
                    G[u][v]["weight"] = new_w

                if hasattr(aco, "cost_matrix"):
                    aco.cost_matrix = cost_matrix

                if hasattr(aco, "heuristic"):
                    eps = 1e-8
                    try:
                        aco.heuristic[u][v] = 1.0 / max(new_w, eps)
                        aco.heuristic[v][u] = 1.0 / max(new_w, eps)
                    except Exception:
                        pass

                if hasattr(aco, "on_cost_matrix_update"):
                    try:
                        aco.on_cost_matrix_update(u, v, new_w)
                    except Exception:
                        warnings.warn("aco.on_cost_matrix_update failed")

        if it >= T_CHANGE and recorded_t_found is None:
            if global_best <= FOUND_ALLOWANCE * new_optimal_cost:
                recorded_t_found = it

    if recorded_t_found is None:
        recorded_t_found = np.inf
    iterations_to_adapt = recorded_t_found - T_CHANGE if recorded_t_found != np.inf else np.inf

    edge_info = (modifications, None)  # placeholder
    return global_best_costs, iterations_to_adapt, new_optimal_cost, None, edge_info


# ------------------ visualization (unchanged) ------------------
def visualize(target_P, actual_P, all_curves, adapt_times, prop, allowed, new_optimal):
    num_runs = len(all_curves)
    max_len = max(len(c) for c in all_curves) if all_curves else 0
    fig = make_subplots(rows=2, cols=1, subplot_titles=[
        f"Global Best Cost per Iteration (Target P={target_P}%, Actual P={actual_P:.2f}%)",
        f"Adaptation Time Distribution (Pass Rate={prop*100:.1f}%)"
    ], vertical_spacing=0.13, row_heights=[0.6, 0.4])

    for i, curve in enumerate(all_curves):
        fig.add_trace(go.Scatter(y=curve, x=list(range(len(curve))), mode='lines',
                                 opacity=0.25, line=dict(width=1), showlegend=(i == 0),
                                 name="Individual runs" if i == 0 else None), row=1, col=1)

    arr = np.full((num_runs, max_len), np.nan)
    for i, L in enumerate(all_curves):
        arr[i, :len(L)] = L
    if arr.size:
        avg = np.nanmean(arr, axis=0)
        fig.add_trace(go.Scatter(y=avg, x=list(range(len(avg))), line=dict(width=3), name="Mean best cost"),
                      row=1, col=1)

    fig.add_vline(x=T_CHANGE, line_dash="dash", line_color="red",
                  annotation_text=f"Weight change (t={T_CHANGE})", annotation_position="top", row=1, col=1)
    fig.add_hline(y=new_optimal, line_dash="dot", line_color="green",
                  annotation_text=f"New optimal = {new_optimal:.0f}", annotation_position="right", row=1, col=1)
    fig.add_hline(y=FOUND_ALLOWANCE * new_optimal, line_dash="dot", line_color="orange",
                  annotation_text=f"Threshold = {FOUND_ALLOWANCE:.2f}×", annotation_position="right", row=1, col=1)
    fig.add_hline(y=KNOWN_OPTIMAL_COST, line_dash="dot", line_color="blue",
                  annotation_text=f"Original optimal = {KNOWN_OPTIMAL_COST}", annotation_position="right", row=1, col=1)

    finite = [t for t in adapt_times if t != np.inf]
    if finite:
        fig.add_trace(go.Histogram(x=finite, nbinsx=min(20, max(5, len(finite))), name="Adaptation times"), row=2, col=1)
        fig.add_vline(x=allowed, line_dash="dash", line_color="red",
                      annotation_text=f"Allowed = {allowed}", annotation_position="top", row=2, col=1)

    fig.update_xaxes(title_text="Iteration", row=1, col=1)
    fig.update_yaxes(title_text="Best Cost Found", row=1, col=1)
    fig.update_xaxes(title_text="Iterations to Adapt", row=2, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=1)
    fig.update_layout(height=900, width=1100,
                      title_text=f"Edge Weight Adaptivity Test — Target P={target_P}%, Actual={actual_P:.2f}%, Allowed ≤ {allowed} iterations",
                      template="plotly_white", showlegend=True)
    fig.show()


# ------------------ main benchmark loop ------------------


def run_benchmark():
    print(f"Loading {GRAPH_PATH}...")
    G_master, coords = load_graph(GRAPH_PATH)
    print(f"Loaded graph with {len(coords)} nodes.")
    base_cm = build_base_cost_matrix(coords)

    # original (known) cost: use Concorde computed on base_cm to be safe
    try:
        original_tour, original_cost = compute_optimal_tour_via_concorde_explicit(base_cm)
    except Exception:
        # fallback to known cost
        original_tour = KNOWN_OPTIMAL_TOUR.copy()
        original_cost = KNOWN_OPTIMAL_COST

    print(f"Original tour cost (used as baseline): {original_cost:.6f}")

    all_results = {}

    for target_P in P_VALUES:
        print("\n=======================================")
        print(f" Building modification sequence for target P = {target_P}%")
        print("=======================================")

        # Build sequence of single-edge modifications applied cumulatively until reaching target_P
        seq, final_cm, achieved_pct = build_modification_sequence_for_target(base_cm, original_cost, target_P, coords)

        print(f"Sequence length: {len(seq)} ; achieved_pct = {achieved_pct:.6f}% (target was {target_P}%)")
        for idx, (edge, w, ach) in enumerate(seq):
            (u, v) = edge
            print(f"  [{idx+1}] edge=({u},{v}) weight={w:.6e} -> cumulative achieved {ach:.6f}%")

        # If sequence is empty or achieved_pct < small epsilon, we still pick the best single-edge (fallback)
        if len(seq) == 0:
            print("No sequence produced (no single-edge improvements found). Skipping this target.")
            continue

        # modifications to apply at T_CHANGE for benchmarking (list of ((u,v), new_weight))
        modifications = [((e[0][0], e[0][1]), e[1]) for e in seq]
        # new optimal cost estimate is cost of final_cm's optimal tour
        try:
            _, new_optimal_cost = compute_optimal_tour_via_concorde_explicit(final_cm)
        except Exception:
            new_optimal_cost = None

        # Run NUM_RUNS trials using the same modifications (apply all at once at T_CHANGE)
        adapt_times = []
        all_cost_curves = []
        for seed in SEEDS:
            G_fresh, _ = load_graph(GRAPH_PATH)
            costs, adapt_time, new_opt, _, _ = run_single_with_modifications(G_fresh, coords, seed, modifications, new_optimal_cost)
            adapt_times.append(adapt_time)
            all_cost_curves.append(costs)
            status = f"{adapt_time}" if adapt_time != np.inf else "NOT FOUND"
            print(f" Seed {seed:2d} → adapt time = {status}")

        prop = sum(t <= max(MIN_ITER, int(round(EXPECTED_FACTOR * achieved_pct))) for t in adapt_times if t != np.inf) / NUM_RUNS

        print(f"\nOriginal optimal: {original_cost:.6f}, New optimal (est): {new_optimal_cost}, Achieved P: {achieved_pct:.6f}%")
        print(f"Pass (≥90%): {'PASS' if prop >= 0.9 else 'FAIL'}")

        all_results[target_P] = (all_cost_curves, adapt_times, prop, new_optimal_cost, achieved_pct)

        visualize(target_P, achieved_pct, all_cost_curves, adapt_times, prop, max(MIN_ITER, int(round(EXPECTED_FACTOR * achieved_pct))) , new_optimal_cost if new_optimal_cost is not None else 0.0)

    return all_results


if __name__ == "__main__":
    run_benchmark()