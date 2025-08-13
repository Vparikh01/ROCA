import yaml

def main(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    print("Loaded config:", config)
    print("ACO engine placeholder ready!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()
    main(args.config)