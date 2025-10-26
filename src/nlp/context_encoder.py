import pickle
import numpy as np
from InstructorEmbedding import INSTRUCTOR
from sentence_transformers import util
import os
import matplotlib.pyplot as plt
import math

# ------------------- LOAD GRAPH -------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PALO_ALTO_PKL = os.path.join(BASE_DIR, "..", "datasets", "palo_alto_graph.pkl")

def load_graph(pkl_path):
    with open(pkl_path, "rb") as f:
        G = pickle.load(f)
    if G.is_directed():
        G = G.to_undirected()
    return G

# ------------------- EXTRACT POI NODE TEXT -------------------
def extract_node_info(graph):
    node_info = {}

    # detect whether any node uses an is_poi-like flag (case-insensitive)
    def has_is_poi_key(data):
        return any(k.lower() in ("is_poi", "ispoi") for k in data.keys())

    any_is_poi = any(has_is_poi_key(data) for _, data in graph.nodes(data=True))

    for node_id, node_data in graph.nodes(data=True):
        # if graph uses is_poi, enforce it; otherwise include all nodes
        if any_is_poi:
            # prefer normalized key checks to support variations
            is_poi = False
            for k, v in node_data.items():
                if k.lower() in ("is_poi", "ispoi"):
                    is_poi = bool(v)
                    break
            if not is_poi:
                continue

        # build combined text from all non-empty fields
        text_fields = []
        for k, v in node_data.items():
            if v:
                if isinstance(v, dict):
                    text_fields.append(" ".join(str(x) for x in v.values()))
                else:
                    text_fields.append(str(v))
        combined_text = " ".join(text_fields).strip()

        if combined_text:
            node_info[node_id] = {
                "text": combined_text,
                "metadata": node_data,
                "embedding": None,
                "pheromone_modifier": 1.0,
            }

    print(f"Total usable POI nodes for NLP: {len(node_info)}")
    return node_info


# ------------------- FUNCTION: PROCESS INSTRUCTION -------------------
model = INSTRUCTOR("hkunlp/instructor-large")

intent_templates = {
    "avoid": ["avoid", "skip", "do not go", "go around", "bypass", "circumvent"],
    "prefer": ["include", "prefer", "go to", "favor", "choose", "select"],
}

def process_instruction(instruction, node_info):
    node_ids = list(node_info.keys())
    targets = [info["text"] for info in node_info.values()]

    # Step 1: Encode instruction
    instruction_emb = model.encode(
        [("Extract the target from this instruction:", instruction)]
    )

    # Step 2: Encode intent prototypes
    avoid_embs = model.encode(
        [("Represent the intent:", t) for t in intent_templates["avoid"]]
    )
    prefer_embs = model.encode(
        [("Represent the intent:", t) for t in intent_templates["prefer"]]
    )

    avoid_score = util.cos_sim(instruction_emb, avoid_embs).mean().item()
    prefer_score = util.cos_sim(instruction_emb, prefer_embs).mean().item()

    intent_type = "avoid" if avoid_score > prefer_score else "prefer"
    intent_multiplier = -1 if intent_type == "avoid" else 1

    raw_diff = (
        avoid_score - prefer_score if intent_type == "avoid" else prefer_score - avoid_score
    )
    confidence = 1 / (1 + math.exp(-20 * raw_diff))  # sharp sigmoid

    # Step 3: Encode node metadata
    node_embs = model.encode(
        [
            ("Represent the metadata of a node for matching this target object:", text)
            for text in targets
        ]
    )

    # Step 4: Similarity computation
    node_sims = util.cos_sim(instruction_emb, node_embs)[0].cpu().numpy()

    # Step 5: Dynamic threshold
    threshold = np.percentile(node_sims, 70)
    max_sim_above_threshold = node_sims[node_sims >= threshold].max()

    # Step 6: Compute modifiers & build consistent node_info_list
    node_info_list = []
    for node_id, sim in zip(node_ids, node_sims):
        if sim < threshold:
            modifier = 1.0
        else:
            sim_factor = (sim - threshold) / (max_sim_above_threshold - threshold)
            sim_factor = sim_factor ** 1.8
            modifier = 1 + intent_multiplier * sim_factor * 5
            modifier = max(0.05, min(modifier, 2.0))

        node_info[node_id]["pheromone_modifier"] = modifier
        node_info_list.append(
            {
                "node_id": node_id,
                "text": node_info[node_id]["text"],
                "metadata": node_info[node_id]["metadata"],
                "similarity": sim,
                "pheromone_modifier": modifier,
            }
        )

    return intent_type, confidence, node_info_list


# ------------------- MAIN -------------------
if __name__ == "__main__":
    graph = load_graph(PALO_ALTO_PKL)
    node_info = extract_node_info(graph)

    instruction = "add libraries"
    intent_type, confidence, node_info_list = process_instruction(
        instruction, node_info
    )

    sorted_nodes = sorted(node_info_list, key=lambda x: x["similarity"], reverse=True)[:100]

    print(
        f"\nInstruction: '{instruction}' (inferred intent: {intent_type}, confidence: {confidence:.2f})"
    )
    for node in sorted_nodes:
        print(
            f"Node {node['node_id']}: similarity {node['similarity']:.3f}, modifier {node['pheromone_modifier']:.3f}"
        )
        print("Metadata:", node["metadata"])
        print("---")

    plt.hist([node["pheromone_modifier"] for node in node_info_list], bins=50)
    plt.title("Distribution of Pheromone Modifiers")
    plt.xlabel("Modifier")
    plt.ylabel("Number of nodes")
    plt.show()
