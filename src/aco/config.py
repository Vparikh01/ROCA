import yaml
from pathlib import Path

def load_config(path="configs/default.yaml"):
    # Always resolve relative to the current working directory
    with open(Path(path), "r") as f:
        return yaml.safe_load(f)