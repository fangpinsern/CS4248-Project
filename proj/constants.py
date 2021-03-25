import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
DATA_DIR = os.path.join(ROOT_DIR, "..", "data")
JSON_DIR = os.path.join(ROOT_DIR, "data")
JSON_FILES = os.listdir(JSON_DIR)
LOG_DIR = os.path.join(ROOT_DIR, "logs")
WEIGHTS_DIR = os.path.join(ROOT_DIR, "model_weights")
