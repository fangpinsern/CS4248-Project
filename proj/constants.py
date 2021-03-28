import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
DATA_DIR = os.path.join(ROOT_DIR, "..", "data")
JSON_FILE = os.path.join(ROOT_DIR, "News_Category_Dataset_v2.json")
LOG_DIR = os.path.join(ROOT_DIR, "logs")
WEIGHTS_DIR = os.path.join(ROOT_DIR, "model_weights")
