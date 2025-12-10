import yaml
import os

def load_config(filename="config.yaml"):
    # Trouve la racine du projet automatiquement
    project_root = os.path.dirname(os.path.dirname(__file__))  # remonte de /src/utils/ à /src/
    project_root = os.path.dirname(project_root)              # remonte encore à la racine du projet

    config_path = os.path.join(project_root, filename)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"❌ Config file not found: {config_path}")

    with open(config_path, "r") as f:
        return yaml.safe_load(f)

CONFIG = load_config()

