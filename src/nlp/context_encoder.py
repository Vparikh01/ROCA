import pickle
import numpy as np
from sentence_transformers import SentenceTransformer, util
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
targets = []
target_metadata = {}
for node_id, node_data in graph.nodes(data=True):
    text_fields = []
    # Use all available metadata fields
    for k in node_data.keys():
        if node_data.get(k):
            if isinstance(node_data[k], dict):
                text_fields.append(" ".join(str(v) for v in node_data[k].values()))
            else:
                text_fields.append(str(node_data[k]))
    combined_text = " ".join(text_fields).strip()
    if combined_text:
        targets.append(combined_text)
        target_metadata[combined_text] = node_data

print(f"Total nodes with metadata: {len(targets)}")

# ------------------- MODEL -------------------
model = SentenceTransformer('all-MiniLM-L6-v2')

# ------------------- INPUT INSTRUCTION -------------------
instruction = "avoid gas stations"

# ------------------- INTENT PROTOTYPES -------------------
intent_prototypes = {
    "avoid": [
        "avoid", "skip", "do not go", "forbid", "exclude", "stay away from",
        "do not choose", "ignore", "ban", "prevent", "prohibit", "not allowed", 
        "restricted", "off limits", "no entry", "disallow", "refrain", "sidestep", 
        "evade", "shun", "omit", "leave out", "neglect", "refrain from", "pass over", 
        "steer clear of", "keep clear", "block","do not enter", "do not select", 
        "avoidance", "go around", "detour", "bypass", "skirt", "circumvent"
    ],
    "prefer": [
        "prefer", "choose", "go to", "favor", "select", "opt for", "prioritize",
        "seek", "aim for", "target", "like", "pick", "favoring", "recommend",
        "desire", "wish for", "incline toward", "lean toward", "best choice",
        "recommendation", "focus on", "highlight", "favoring", "go towards",
        "preferably", "advocate", "encourage", "take", "selecting", "choose first"
    ]
}

# ------------------- ENCODINGS -------------------
instruction_emb = model.encode(instruction, convert_to_tensor=True)
target_embs = model.encode(targets, convert_to_tensor=True)
avoid_embs = model.encode(intent_prototypes["avoid"], convert_to_tensor=True)
prefer_embs = model.encode(intent_prototypes["prefer"], convert_to_tensor=True)

# ------------------- SIMILARITIES -------------------
target_sims = util.cos_sim(instruction_emb, target_embs)[0].cpu().numpy()

avoid_cos = util.cos_sim(instruction_emb, avoid_embs)
prefer_cos = util.cos_sim(instruction_emb, prefer_embs)

avoid_score = 0.7 * avoid_cos.mean().item() + 0.3 * avoid_cos.max().item()
prefer_score = 0.7 * prefer_cos.mean().item() + 0.3 * prefer_cos.max().item()

intent_score = prefer_score - avoid_score
intent_scaled = np.tanh(5 * intent_score)  # Range: -1 to 1

# ------------------- PHEROMONE MODIFIERS -------------------
pheromone_modifiers = {}
baseline_sim = 0.25  # Only nodes with similarity above this are affected

for t, sim in zip(targets, target_sims):
    if sim < baseline_sim:
        modifier = 1.0  # Neutral, no change
    else:
        # Exaggerate outliers using a non-linear transform
        sim_factor = np.tanh(sim * 3) ** 2
        modifier = 1 + intent_scaled * sim_factor * 5
        modifier = max(0.05, min(modifier, 2.0))  # Clip to [0.05, 2.0]
    pheromone_modifiers[t] = modifier

# ------------------- TOP MATCHES -------------------
if intent_scaled >= 0:
    # Prefer instruction → top matches with highest similarity
    top_matches = sorted(zip(targets, target_sims), key=lambda x: x[1], reverse=True)[:19]
else:
    # Avoid instruction → top nodes to avoid = highest similarity to avoid instruction
    top_matches = sorted(zip(targets, target_sims), key=lambda x: x[1], reverse=True)[:19]

print("\nTop 5 matches:")
for t, sim in top_matches:
    print(f"{t}: similarity {sim:.3f}")
    print("Pheromone modifier:", round(pheromone_modifiers[t], 3))
    print("Metadata:", target_metadata[t])
    print("---")

print(f"\nModifier range: 0.05 (max avoid) → 2.0 (max prefer)")
print(f"Intent score (scaled): {intent_scaled:.3f}")

plt.hist(list(pheromone_modifiers.values()), bins=50)
plt.title("Distribution of Pheromone Modifiers")
plt.xlabel("Modifier")
plt.ylabel("Number of nodes")
plt.show()