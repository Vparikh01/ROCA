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

NUM_RUNS = 2
I_MAX = 200
T_CHANGE = 80               # iteration where weight modification happens
P_VALUES = [5, 10, 20]      # percent improvements to target
FOUND_ALLOWANCE = 1.12      # route ≤ 112% of new optimal
SEEDS = list(range(1, NUM_RUNS + 1))
EXPECTED_FACTOR = 1.75      # expected allowed iterations = round(1.75 * P)
MIN_ITER = 1

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GRAPH_PATH = os.path.join(BASE_DIR, "..", "tsplib_graphs", "eil51.pkl")

# Known optimal tour for eil51 (from TSPLIB) - 0-indexed
KNOWN_OPTIMAL_TOUR = [
    0, 21, 7, 25, 30, 27, 2, 35, 34, 19, 1, 28, 20, 15, 49, 33, 29, 8, 48, 9,
    38, 32, 44, 14, 43, 41, 39, 18, 40, 12, 24, 13, 23, 42, 6, 22, 47, 5, 26,
    50, 45, 11, 46, 17, 3, 16, 36, 4, 37, 10, 31
]
KNOWN_OPTIMAL_COST = 426

# Concorde run behavior
CONCORDE_BINARY = os.environ.get("CONCORDE_BIN", "concorde")  # fallback
USE_PYCONCORDE = True  # try pyconcorde first
VERBOSE = False        # suppress Concorde logs

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
            from concorde.tsp import TSPSolver
            solver = TSPSolver.from_tspfile(tsp_path)
            sol = solver.solve()
            tour = list(sol.tour)
            cost = float(sol.tour_length) if hasattr(sol, "tour_length") else None
            return tour, cost
        except Exception:
            pass

    try:
        with open(os.devnull, "w") as fnull:
            subprocess.check_call([CONCORDE_BINARY, tsp_path], cwd=work_dir,
                                  stdout=fnull, stderr=fnull)
    except FileNotFoundError as e:
        raise RuntimeError(f"Concorde binary not found at '{CONCORDE_BINARY}'.") from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Concorde returned non-zero exit status: {e}")

    base = os.path.splitext(os.path.basename(tsp_path))[0]
    sol_file = next((os.path.join(work_dir, f) for f in os.listdir(work_dir)
                     if f.startswith(base) and f.endswith(".sol")), None)
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
                if v == -1: continue
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

# ============================================================
# EDGE MODIFICATION
# ============================================================
def create_edge_modification(cost_matrix: np.ndarray, target_P: float, coords: np.ndarray,
                             prev_edge: Optional[Tuple[int,int]] = None,
                             prev_weight: Optional[float] = None,
                             prev_actual_P: Optional[float] = None) -> Tuple[Optional[int], Optional[int], Optional[float], Optional[List[int]], Optional[float], Optional[float]]:
    n = cost_matrix.shape[0]
    if prev_edge is None:
        # First modification: search for best edge
        original_tour = KNOWN_OPTIMAL_TOUR.copy()
        orig_cost = compute_tour_cost(original_tour, cost_matrix)
        target_new_cost = orig_cost * (1 - target_P/100)

        # candidate edges not in original tour
        original_edges = {(min(original_tour[i], original_tour[(i+1)%n]),
                           max(original_tour[i], original_tour[(i+1)%n])) for i in range(n)}
        candidate_edges = [(i,j,cost_matrix[i,j])
                           for i in range(n) for j in range(i+1,n)
                           if (i,j) not in original_edges and cost_matrix[i,j]>1]

        candidate_edges.sort(key=lambda x: x[2], reverse=True)  # prioritize larger weights
        best_result = None
        for u,v,w in candidate_edges[:40]:
            for keep in [0.01,0.02,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
                new_w = max(1e-8, w*keep)
                modified = cost_matrix.copy()
                modified[u,v] = modified[v,u] = new_w
                try:
                    new_tour,new_cost = compute_optimal_tour_via_concorde_explicit(modified)
                except:
                    continue
                actual_P = (orig_cost - new_cost)/orig_cost*100
                if best_result is None or abs(actual_P - target_P)<abs(best_result[5]-target_P):
                    best_result = (u,v,new_w,new_tour,new_cost,actual_P)
                if actual_P >= target_P:
                    break
            if best_result and best_result[5]>=target_P: break
        if best_result is None:
            return (None, None, None, None, None, None)
        return best_result
    else:
        # scale previous edge proportionally
        if not isinstance(prev_edge, tuple) or len(prev_edge)!=2:
            raise RuntimeError(f"prev_edge is not a 2-tuple: {prev_edge}")
        u,v = prev_edge
        # proportional weight adjustment
        new_weight = prev_weight * ((1 - target_P/100) / (1 - prev_actual_P/100))
        modified = cost_matrix.copy()
        modified[u,v] = modified[v,u] = new_weight
        new_tour,new_cost = compute_optimal_tour_via_concorde_explicit(modified)
        actual_P = (KNOWN_OPTIMAL_COST - new_cost)/KNOWN_OPTIMAL_COST*100
        return u,v,new_weight,new_tour,new_cost,actual_P


# ============================================================
# RUN
# ============================================================
def run_single(G: nx.Graph, coords: np.ndarray, seed: int, target_P: float,
               prev_edge: Optional[Tuple[int,int]]=None,
               prev_weight: Optional[float]=None,
               prev_actual_P: Optional[float]=None):
    n = len(coords)
    cost_matrix = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if i!=j:
                cost_matrix[i,j]=float(np.round(np.linalg.norm(coords[i]-coords[j])))

    u,v,new_weight,new_optimal_tour,new_optimal_cost,actual_P = create_edge_modification(
        cost_matrix,target_P,coords,prev_edge,prev_weight,prev_actual_P
    )

    if u is None:
        raise RuntimeError("create_edge_modification failed")

    # Initialize ACO
    aco = MaxMinACO(
        cost_matrix,
        start_node=0,
        reducedGraph=G,
        completeGraph=G,
        shortest_paths={},
        required_nodes=list(range(n)),
        index_map={i:i for i in range(n)},
        seed=seed,
    )

    global_best = np.inf
    global_best_costs = []
    recorded_t_found = None

    for it in range(I_MAX):
        aco.run(iterations=1, n=it)
        iter_best = np.inf
        for ant in getattr(aco,"ants",[]):
            if ant is None or len(getattr(ant,"tour",[]))!=n: continue
            ant_cost = compute_tour_cost(ant.tour,cost_matrix)
            iter_best = min(iter_best,ant_cost)
        if iter_best<global_best:
            global_best=iter_best
        global_best_costs.append(global_best)
        if it==T_CHANGE:
            cost_matrix[u,v]=cost_matrix[v,u]=new_weight
            if G.has_edge(u,v): G[u][v]["weight"]=new_weight
            if hasattr(aco,"cost_matrix"): aco.cost_matrix=cost_matrix
            if hasattr(aco,"heuristic"):
                eps=1e-8
                try: aco.heuristic[u][v]=1/ max(new_weight,eps); aco.heuristic[v][u]=1/max(new_weight,eps)
                except: pass
            if hasattr(aco,"on_cost_matrix_update"):
                try: aco.on_cost_matrix_update(u,v,new_weight)
                except: warnings.warn("aco.on_cost_matrix_update failed")

        if it>=T_CHANGE and recorded_t_found is None:
            if global_best <= FOUND_ALLOWANCE*new_optimal_cost:
                recorded_t_found=it

    if recorded_t_found is None: recorded_t_found=np.inf
    iterations_to_adapt = recorded_t_found - T_CHANGE if recorded_t_found!=np.inf else np.inf

    # edge_info: tuple of (prev_edge, prev_weight, prev_actual_P) for next run
    edge_info = ((u,v), new_weight, actual_P)
    return global_best_costs, iterations_to_adapt, new_optimal_cost, actual_P, edge_info

def run_benchmark():
    print(f"Loading {GRAPH_PATH}...")
    G, coords = load_graph(GRAPH_PATH)
    print(f"Loaded graph with {len(coords)} nodes.")
    print(f"Known optimal tour cost: {KNOWN_OPTIMAL_COST}")

    all_results = {}

    for target_P in P_VALUES:
        # Reset previous edge info for each target P
        prev_edge = prev_weight = prev_actual_P = None

        print("\n===============================")
        print(f"   Testing P = {target_P}% improvement")
        print("===============================")
        adapt_times = []
        all_cost_curves = []
        new_optimal = None
        actual_P = None

        for seed in SEEDS:
            # Reload fresh graph for each run
            G_fresh, _ = load_graph(GRAPH_PATH)

            costs, adapt_time, new_opt, act_P, edge_info = run_single(
                G_fresh, coords, seed, target_P,
                prev_edge, prev_weight, prev_actual_P
            )

            adapt_times.append(adapt_time)
            all_cost_curves.append(costs)

            if new_optimal is None:
                new_optimal = new_opt
                actual_P = act_P

            # update prev_edge info for next seed in the same P
            prev_edge, prev_weight, prev_actual_P = edge_info

            status = f"{adapt_time}" if adapt_time != np.inf else "NOT FOUND"
            print(f"  Seed {seed:2d} → adapt time = {status}")

        allowed = max(MIN_ITER, int(round(EXPECTED_FACTOR * actual_P)))
        passes = sum(t <= allowed for t in adapt_times if t != np.inf)
        prop = passes / NUM_RUNS

        print(f"\nOriginal optimal: {KNOWN_OPTIMAL_COST}, New optimal: {new_optimal}, Actual P: {actual_P:.2f}%")
        print(f"Allowed iterations: {allowed}, Passes: {passes}/{NUM_RUNS} ({prop*100:.1f}%)")
        print(f"Pass (≥90%): {'PASS' if prop >= 0.9 else 'FAIL'}")

        all_results[target_P] = (all_cost_curves, adapt_times, prop, new_optimal, actual_P)
        visualize(target_P, actual_P, all_cost_curves, adapt_times, prop, allowed, new_optimal)

    return all_results

# ============================================================
# VISUALIZATION
# ============================================================
def visualize(target_P, actual_P, all_curves, adapt_times, prop, allowed, new_optimal):
    num_runs=len(all_curves)
    max_len=max(len(c) for c in all_curves)
    fig=make_subplots(rows=2,cols=1,subplot_titles=[
        f"Global Best Cost per Iteration (Target P={target_P}%, Actual P={actual_P:.1f}%)",
        f"Adaptation Time Distribution (Pass Rate={prop*100:.1f}%)"
    ],vertical_spacing=0.13,row_heights=[0.6,0.4])

    for i,curve in enumerate(all_curves):
        fig.add_trace(go.Scatter(y=curve,x=list(range(len(curve))),mode='lines',
                                 opacity=0.25,line=dict(width=1),showlegend=(i==0),
                                 name="Individual runs" if i==0 else None), row=1,col=1)

    arr=np.full((num_runs,max_len),np.nan)
    for i,L in enumerate(all_curves): arr[i,:len(L)]=L
    avg=np.nanmean(arr,axis=0)
    fig.add_trace(go.Scatter(y=avg,x=list(range(len(avg))),line=dict(width=3),name="Mean best cost"),row=1,col=1)

    fig.add_vline(x=T_CHANGE,line_dash="dash",line_color="red",annotation_text=f"Weight change (t={T_CHANGE})",annotation_position="top",row=1,col=1)
    fig.add_hline(y=new_optimal,line_dash="dot",line_color="green",annotation_text=f"New optimal = {new_optimal:.0f}",annotation_position="right",row=1,col=1)
    fig.add_hline(y=FOUND_ALLOWANCE*new_optimal,line_dash="dot",line_color="orange",annotation_text=f"Threshold = {FOUND_ALLOWANCE:.2f}×",annotation_position="right",row=1,col=1)
    fig.add_hline(y=KNOWN_OPTIMAL_COST,line_dash="dot",line_color="blue",annotation_text=f"Original optimal = {KNOWN_OPTIMAL_COST}",annotation_position="right",row=1,col=1)

    finite=[t for t in adapt_times if t!=np.inf]
    if finite:
        fig.add_trace(go.Histogram(x=finite,nbinsx=min(20,max(5,len(finite))),name="Adaptation times"),row=2,col=1)
        fig.add_vline(x=allowed,line_dash="dash",line_color="red",annotation_text=f"Allowed = {allowed}",annotation_position="top",row=2,col=1)

    fig.update_xaxes(title_text="Iteration",row=1,col=1)
    fig.update_yaxes(title_text="Best Cost Found",row=1,col=1)
    fig.update_xaxes(title_text="Iterations to Adapt",row=2,col=1)
    fig.update_yaxes(title_text="Count",row=2,col=1)
    fig.update_layout(height=900,width=1100,title_text=f"Edge Weight Adaptivity Test — Target P={target_P}%, Actual={actual_P:.1f}%, Allowed ≤ {allowed} iterations",template="plotly_white",showlegend=True)
    fig.show()

if __name__=="__main__":
    run_benchmark()