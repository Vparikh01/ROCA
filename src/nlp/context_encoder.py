import pickle
import numpy as np
from InstructorEmbedding import INSTRUCTOR
from sentence_transformers import util
import os
import matplotlib.pyplot as plt
import math
import spacy

# ------------------- LOAD GRAPH -------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PALO_ALTO_PKL = os.path.join(BASE_DIR, "..", "datasets", "palo_alto_graph.pkl")
EURO_AIRPORTS_PKL = os.path.join(BASE_DIR, "..", "datasets", "europe_airports_graph.pkl")

def load_graph(pkl_path):
    with open(pkl_path, "rb") as f:
        G = pickle.load(f)
    if G.is_directed():
        G = G.to_undirected()
    return G

# ------------------------------------------------------------ 
# DOMAIN-AGNOSTIC METADATA EXTRACTION
# ------------------------------------------------------------

# Primary category/type fields across different domains
# These are the "what is it?" fields that classify the entity
CATEGORY_KEYS = [
    # OpenStreetMap classification systems
    "amenity",           # POIs: library, restaurant, school, hospital
    "shop",              # Retail: grocery, bakery, electronics
    "tourism",           # Tourist: hotel, museum, viewpoint
    "office",            # Business: company, government, lawyer
    "leisure",           # Recreation: park, sports_centre, playground
    "sport",             # Sports: tennis, swimming, soccer
    "natural",           # Nature: water, peak, beach, forest
    "place",             # Settlements: city, town, village
    "historic",          # Heritage: monument, castle, ruins
    "building",          # Structures: residential, commercial, industrial
    "highway",           # Roads: primary, residential, motorway
    "railway",           # Rail: station, platform, rail
    "aeroway",           # Airport infrastructure: runway, taxiway, terminal
    "public_transport",  # Transit: stop_position, platform, station
    
    # OpenFlights/Airport data (case-insensitive)
    "type",              # OpenFlights: "airport", "station", "port"
    "Type",              # Capitalized version
    "source",            # Data source: "OurAirports", "DAFIF", etc.
    "Source",            # Capitalized version
]

# Secondary descriptive fields
# These are the "what's it called / where is it?" fields
DESCRIPTIVE_KEYS = [
    # Names and identifiers (lowercase)
    "name",              # Primary name
    "brand",             # Brand name (e.g., "Starbucks")
    "operator",          # Operating entity (e.g., "City Library System")
    "description",       # Text description
    "alt_name",          # Alternative name
    "ref",               # Reference code
    
    # OpenFlights specific (lowercase)
    "iata",              # 3-letter airport code (e.g., "SFO")
    "icao",              # 4-letter airport code (e.g., "KSFO")
    "city",              # City name
    "country",           # Country name
    "callsign",          # Airline callsign
    
    # OpenFlights specific (capitalized versions)
    "Name",              # Airport name
    "IATA",              # 3-letter code
    "ICAO",              # 4-letter code
    "City",              # City name
    "Country",           # Country name
]

# Additional contextual fields (lower priority)
CONTEXTUAL_KEYS = [
    "opening_hours", "phone", "website", "cuisine", "addr:street",
    "addr:city", "tags", "label", "category"
]

def extract_category_fields(metadata):
    """Extract only the PRIMARY category/type fields."""
    parts = []
    for key in CATEGORY_KEYS:
        if key in metadata and metadata[key]:
            val = str(metadata[key]).strip()
            if val:
                parts.append(val)
    return " ".join(parts).strip()

def extract_descriptive_fields(metadata):
    """Extract secondary descriptive information."""
    parts = []
    for key in DESCRIPTIVE_KEYS:
        if key in metadata and metadata[key]:
            val = str(metadata[key]).strip()
            if val:
                parts.append(val)
    return " ".join(parts).strip()

def extract_contextual_fields(metadata):
    """Extract additional contextual information."""
    parts = []
    for key in CONTEXTUAL_KEYS:
        if key in metadata and metadata[key]:
            val = str(metadata[key]).strip()
            if val:
                parts.append(val)
    return " ".join(parts).strip()

# ------------------------------------------------------------ 
# EXTRACT POI NODE TEXT WITH STRUCTURED FIELDS
# ------------------------------------------------------------

def extract_node_info(graph):
    node_info = {}
    
    # Detect whether any node uses an is_poi-like flag (case-insensitive)
    def has_is_poi_key(data):
        return any(k.lower() in ("is_poi", "ispoi") for k in data.keys())
    
    any_is_poi = any(has_is_poi_key(data) for _, data in graph.nodes(data=True))
    
    for node_id, node_data in graph.nodes(data=True):
        # If graph uses is_poi flag, enforce it
        if any_is_poi:
            is_poi = False
            for k, v in node_data.items():
                if k.lower() in ("is_poi", "ispoi"):
                    is_poi = bool(v)
                    break
            if not is_poi:
                continue
        
        # Extract structured fields
        category_text = extract_category_fields(node_data)
        descriptive_text = extract_descriptive_fields(node_data)
        contextual_text = extract_contextual_fields(node_data)
        
        # Create full text (for fallback matching)
        full_text = f"{category_text} {descriptive_text} {contextual_text}".strip()
        
        # More lenient: accept nodes with ANY text content
        # This helps with airport graphs that might not have traditional categories
        if full_text:  # Changed from: if category_text or descriptive_text
            node_info[node_id] = {
                "category_text": category_text,
                "descriptive_text": descriptive_text,
                "contextual_text": contextual_text,
                "full_text": full_text,
                "metadata": node_data,
                "pheromone_modifier": 1.0,
            }
    
    print(f"Total usable POI nodes for NLP: {len(node_info)}")
    if len(node_info) == 0:
        print("WARNING: No nodes extracted! Check if graph has appropriate metadata fields.")
        print("Sample node metadata:", dict(list(graph.nodes(data=True))[:1]) if graph.number_of_nodes() > 0 else "No nodes in graph")
    
    return node_info

# ------------------- INSTRUCTOR / SPACY SETUP -------------------
model = INSTRUCTOR("hkunlp/instructor-large")
nlp = spacy.load("en_core_web_sm")

intent_templates = {
    "avoid": ["avoid", "skip", "do not go", "go around", "bypass", "circumvent"],
    "prefer": ["include", "prefer", "go to", "favor", "choose", "select"],
}

# ------------------------------------------------------------ 
# CATEGORY EXTRACTION FROM INSTRUCTION
# ------------------------------------------------------------

def extract_category_from_instruction(instruction):
    """Extract the core category/entity type being requested."""
    doc = nlp(instruction)
    
    candidates = []
    for token in doc:
        if token.pos_ in ("NOUN", "PROPN") and not token.is_stop:
            lemma = token.lemma_.lower()
            candidates.append({
                "term": lemma,
                "original": token.text.lower(),
                "is_plural": token.tag_ in ("NNS", "NNPS")
            })
    
    return candidates if candidates else [{"term": instruction.lower(), "original": instruction.lower(), "is_plural": False}]

# ------------------------------------------------------------ 
# MAIN INSTRUCTION PROCESSING WITH MULTI-STAGE MATCHING
# Uses NEW ranking system but ORIGINAL threshold/modifier logic
# ------------------------------------------------------------

def process_instruction(instruction, node_info):
    node_ids = list(node_info.keys())
    
    # ---------- 1. Extract category candidates from instruction ----------
    category_candidates = extract_category_from_instruction(instruction)
    
    # ========== ORIGINAL INTENT DETECTION (EXACT) ==========
    
    # ---------- 2. Encode instruction (ORIGINAL) ----------
    instruction_emb = model.encode(
        [("Extract the target from this instruction:", instruction)]
    )[0]  # Note: original returns array, we take [0]
    
    # ---------- 3. Encode intent prototypes (ORIGINAL) ----------
    avoid_embs = model.encode(
        [("Represent the intent:", t) for t in intent_templates["avoid"]]
    )
    prefer_embs = model.encode(
        [("Represent the intent:", t) for t in intent_templates["prefer"]]
    )
    
    # ---------- 4. Detect intent (ORIGINAL) ----------
    avoid_score = util.cos_sim(instruction_emb, avoid_embs).mean().item()
    prefer_score = util.cos_sim(instruction_emb, prefer_embs).mean().item()
    
    intent_type = "avoid" if avoid_score > prefer_score else "prefer"
    intent_multiplier = -1 if intent_type == "avoid" else 1
    
    raw_diff = (
        avoid_score - prefer_score if intent_type == "avoid" else prefer_score - avoid_score
    )
    confidence = 1 / (1 + math.exp(-20 * raw_diff))
    
    # ========== NEW RANKING SYSTEM ==========
    
    # ========== NEW RANKING SYSTEM ==========
    
    # ---------- 4. STAGE 1: Category Matching ----------
    category_texts = [info["category_text"] for info in node_info.values()]
    
    category_embs = model.encode([
        ("Represent this entity type or category:", text if text else "unknown") 
        for text in category_texts
    ])
    
    category_sims = util.cos_sim(instruction_emb, category_embs)[0].cpu().numpy()
    
    # ---------- 5. Exact Term Matching Bonus ----------
    exact_match_bonus = np.zeros(len(node_ids))
    
    for i, info in enumerate(node_info.values()):
        cat_lower = info["category_text"].lower()
        desc_lower = info["descriptive_text"].lower()
        
        for candidate in category_candidates:
            # Check for exact or lemma matches in category field
            if candidate["term"] in cat_lower or candidate["original"] in cat_lower:
                exact_match_bonus[i] = 0.4
                break
            # Partial match in descriptive text (weaker signal)
            elif candidate["term"] in desc_lower or candidate["original"] in desc_lower:
                exact_match_bonus[i] = 0.2
                break
    
    # ---------- 6. STAGE 2: Full Context Matching ----------
    full_texts = [info["full_text"] for info in node_info.values()]
    
    full_embs = model.encode([
        ("This is an entity description used for matching search queries:", text)
        for text in full_texts
    ])
    
    full_sims = util.cos_sim(instruction_emb, full_embs)[0].cpu().numpy()
    
    # ---------- 7. Weighted Combination (NEW RANKING) ----------
    final_sims = (
        0.45 * category_sims +      # Category semantic similarity
        0.35 * exact_match_bonus +  # Exact term matches (strongest signal)
        0.20 * full_sims            # Full context similarity
    )
    
    # ========== ORIGINAL THRESHOLD/MODIFIER LOGIC ==========
    
    # ---------- 8. ORIGINAL: 70th percentile threshold ----------
    threshold = np.percentile(final_sims, 70)
    max_sim_above_threshold = final_sims[final_sims >= threshold].max()
    
    # ---------- 9. ORIGINAL: Pheromone modifier calculation ----------
    node_info_list = []
    
    for node_id, sim in zip(node_ids, final_sims):
        if sim < threshold:
            modifier = 1.0
        else:
            denom = max(1e-12, max_sim_above_threshold - threshold)
            sim_factor = (sim - threshold) / denom
            sim_factor = sim_factor ** 1.8  # ORIGINAL exponent
            modifier = 1 + intent_multiplier * sim_factor * 5  # ORIGINAL scale
            modifier = max(0.05, min(modifier, 2.0))
        
        node_info[node_id]["pheromone_modifier"] = modifier
        
        node_info_list.append({
            "node_id": node_id,
            "text": node_info[node_id]["full_text"],
            "metadata": node_info[node_id]["metadata"],
            "similarity": sim,
            "pheromone_modifier": modifier,
        })
    
    return intent_type, confidence, node_info_list

# ------------------- MAIN -------------------
if __name__ == "__main__":
    graph = load_graph(EURO_AIRPORTS_PKL)
    node_info = extract_node_info(graph)

    instruction = "include czech airports"

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