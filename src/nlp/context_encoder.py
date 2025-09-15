import pickle
import numpy as np
from InstructorEmbedding import INSTRUCTOR
from sentence_transformers import util
import os
import matplotlib.pyplot as plt

# ------------------- LOAD GRAPH -------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PALO_ALTO_PKL = os.path.join(BASE_DIR, "..", "datasets", "palo_alto_graph.pkl")

def load_graph(pkl_path):
    with open(pkl_path, "rb") as f:
        G = pickle.load(f)
    if G.is_directed():
        G = G.to_undirected()
    return G

graph = load_graph(PALO_ALTO_PKL)

# ------------------- EXTRACT NODE TEXT -------------------
# ------------------- EXTRACT POI NODE TEXT -------------------
targets = []
target_metadata = {}
for node_id, node_data in graph.nodes(data=True):
    if not node_data.get("is_poi", False):
        continue  # skip non-POI nodes

    text_fields = []
    for k, v in node_data.items():
        if v:
            if isinstance(v, dict):
                text_fields.append(" ".join(str(x) for x in v.values()))
            else:
                text_fields.append(str(v))
    combined_text = " ".join(text_fields).strip()
    if combined_text:
        targets.append(combined_text)
        target_metadata[combined_text] = node_data


print(f"Total nodes with metadata: {len(targets)}")

# ------------------- PRELOAD MODEL -------------------
# Preload once to reuse across instructions
model = INSTRUCTOR('hkunlp/instructor-large')

# ------------------- INTENT PROTOTYPES -------------------
intent_templates = {
    "avoid": [
        "avoid", "skip", "do not go", "go around", "bypass", "circumvent"
    ],
    "prefer": [
        "include", "prefer", "go to", "favor", "choose", "select"
    ]
}

# ------------------- FUNCTION: PROCESS INSTRUCTION -------------------
def process_instruction(instruction, targets, model, intent_templates):
    # Step 1: Extract target object embedding from instruction
    instruction_emb = model.encode([
        ("Extract the target   from this instruction:", instruction)
    ])
    
    # Step 2: Determine intent type by comparing to avoid/prefer prototypes
    avoid_embs = model.encode([("Represent the intent:", t) for t in intent_templates["avoid"]])
    prefer_embs = model.encode([("Represent the intent:", t) for t in intent_templates["prefer"]])
    
    avoid_score = util.cos_sim(instruction_emb, avoid_embs).mean().item()
    prefer_score = util.cos_sim(instruction_emb, prefer_embs).mean().item()
    
    intent_type = "avoid" if avoid_score > prefer_score else "prefer"
    intent_multiplier = -1 if intent_type == "avoid" else 1
    
    # Step 3: Encode nodes relative to extracted target
    target_node_embs = model.encode([
        ("Represent the metadata of a node for matching this target object:", t) 
        for t in targets
    ])
    
    # Step 4: Similarity between instruction target and node metadata
    node_sims = util.cos_sim(instruction_emb, target_node_embs)[0].cpu().numpy()
    
    # Step 5: Dynamic threshold (top 30%)
    threshold = np.percentile(node_sims, 70)
    max_sim_above_threshold = node_sims[node_sims >= threshold].max()
    
    # Step 6: Compute pheromone modifiers (smooth gradient)
    pheromone_modifiers = {}
    for t, sim in zip(targets, node_sims):
        if sim < threshold:
            modifier = 1.0
        else:
            sim_factor = (sim - threshold) / (max_sim_above_threshold - threshold)
            sim_factor = sim_factor ** 1.8  # exaggerate outliers
            modifier = 1 + intent_multiplier * sim_factor * 5
            modifier = max(0.05, min(modifier, 2.0))
        pheromone_modifiers[t] = modifier
    
    return intent_type, node_sims, pheromone_modifiers

# ------------------- EXAMPLE INSTRUCTION -------------------
instruction = "try to add movie theaters"
intent_type, node_sims, pheromone_modifiers = process_instruction(instruction, targets, model, intent_templates)

# ------------------- DISPLAY TOP MATCHES -------------------
top_matches = sorted(zip(targets, node_sims), key=lambda x: x[1], reverse=True)[:100]

print(f"\nInstruction: '{instruction}' (inferred intent: {intent_type})")
for t, sim in top_matches:
    print(f"{t}: similarity {sim:.3f}, modifier {round(pheromone_modifiers[t],3)}")
    print("Metadata:", target_metadata[t])
    print("---")

plt.hist(list(pheromone_modifiers.values()), bins=50)
plt.title("Distribution of Pheromone Modifiers")
plt.xlabel("Modifier")
plt.ylabel("Number of nodes")
plt.show()
