import os as os
from pathlib import Path


def get_project_root() -> Path:
    return Path(__file__).parent.parent


ROOT_DIR = get_project_root()
DATA_DIR = os.path.join(ROOT_DIR, "data")
DATASET_DIR = os.path.join(ROOT_DIR, "dataset")
MODELS_DIR = os.path.join(ROOT_DIR, "models")

CODE_METRICS_MODELS = os.path.join(MODELS_DIR, "code_metrics")
CODE_REPRESENTATION_MODELS = os.path.join(MODELS_DIR, "code_representation")
