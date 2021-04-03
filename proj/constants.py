import os

# file path constants
# This is your Project Root
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "..", "data")
PREDS_DIR = os.path.join(DATA_DIR, "predictions")
JSON_FILE = os.path.join(DATA_DIR, "News_Category_Dataset_v2.json")
DF_FILE = os.path.join(DATA_DIR, "subsetNews.csv")
TRAIN_TEST_SPLIT_FILE = os.path.join(DATA_DIR, "train_test_split_dataset.csv")
LOG_DIR = os.path.join(ROOT_DIR, "logs")
WEIGHTS_DIR = os.path.join(ROOT_DIR, "model_weights")
EMBEDDINGS_DIR = os.path.join(DATA_DIR, "embeddings")
GLOVE_FILE = os.path.join(EMBEDDINGS_DIR, "glove.6B.50d.txt")
WORD2IDX_FILE = os.path.join(EMBEDDINGS_DIR, "6B.50_idx.pkl")

# SEED
SEED = 42

# Data related constants
X_COL = "headline"
Y_COL = "category"
PRED_COL = "prediction"

CATEGORY_DICT = {
    0: "CRIME",
    1: "RELIGION",
    2: "TECH",
    3: "MONEY",
    4: "FOOD & DRINK",
    5: "SPORTS",
    6: "TRAVEL",
    7: "WOMEN",
    8: "STYLE",
    9: "ENTERTAINMENT",
}

CATEGORY_SUBSET = [
    "CRIME",
    "RELIGION",
    "TECH",
    "MONEY",
    "FOOD & DRINK",
    "SPORTS",
    "TRAVEL",
    "WOMEN",
    "STYLE",
    "ENTERTAINMENT",
]

MAX_INPUT_LENGTH = 16
