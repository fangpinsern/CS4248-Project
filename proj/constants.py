import os

# file path constants
# This is your Project Root
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "..", "data")
PREDS_DIR = os.path.join(DATA_DIR, "predictions")
VOCAB_DIR = os.path.join(ROOT_DIR, "vocab")

JSON_FILE = os.path.join(DATA_DIR, "News_Category_Dataset_v2.json")
DF_FILE = os.path.join(DATA_DIR, "subsetNews.csv")
TRAIN_TEST_SPLIT_FILE = os.path.join(DATA_DIR, "train_test_split_dataset.csv")
LOG_DIR = os.path.join(ROOT_DIR, "logs")
WEIGHTS_DIR = os.path.join(ROOT_DIR, "model_weights")
EMBEDDINGS_DIR = os.path.join(DATA_DIR, "embeddings")
GLOVE_FILE = os.path.join(EMBEDDINGS_DIR, "glove.6B.50d.txt")
WORD2IDX_FILE = os.path.join(EMBEDDINGS_DIR, "6B.50_idx.pkl")

DL_BIGRAM_GLOVE_EMBEDDINGS = os.path.join(
    ".vector_cache", "glove.6B.50d.bigram.pkl")
BIGRAM_VOCAB_EMBEDDINGS = os.path.join(
    VOCAB_DIR, "mitten_bigram_dict_50d_515_10000.pkl")
DISTILBERT_BIGRAM_TOKENIZER = os.path.join(
    ".vector_cache", "distilbert")
DISTILBERT_POS_TOKENIZER = os.path.join(
    ".vector_cache", "distilbertPOS")
DISTILBERT_EMBED_TOKENIZER = os.path.join(".vector_cache", "distilbertEmbed")

BIGRAM_TRIGRAM_VOCAB = os.path.join(VOCAB_DIR, "bigram_trigram_vocab_PMI.csv")


def BIGRAM_VOCAB_EMBEDDINGS_FN(n): return os.path.join(
    VOCAB_DIR, f"mitten_bigram_dict_{n}d_515_10000.pkl")


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

MAX_INPUT_LENGTH = 26

UPENN_TAGSET = ['LS', 'TO', 'VBN', 'WP', 'UH', 'VBG', 'JJ', 'VBZ', 'VBP', 'NN', 'DT', 'PRP', 'WP$', 'NNPS', 'PRP$', 'WDT',
                'RB', 'RBR', 'RBS', 'VBD', 'IN', 'FW', 'RP', 'JJR', 'JJS', 'PDT', 'MD', 'VB', 'WRB', 'NNP', 'EX', 'NNS', 'SYM', 'CC', 'CD', 'POS']

UPENN_TAGSET = ['<' + t.lower() + '>' for t in UPENN_TAGSET]
