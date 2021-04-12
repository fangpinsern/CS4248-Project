import torch
import os
import pandas as pd
from proj.constants import DATA_DIR, DL_BIGRAM_GLOVE_EMBEDDINGS
from proj.data.data import NewsDataset
import pickle


def split_col(df):
    train = df[df['phase'] == 'train']
    val = df[df['phase'] == 'dev']
    test = df[df['phase'] == 'test']
    return train, val, test


def representUnknown():
    gloveSize = glove.vectors.shape[0]
    for text in dfs[0]['headline']:
        tokens = tokenizer.tokenize(text)
        for t in tokens:
            if t not in glove.stoi:
                glove.stoi[t] = gloveSize
                glove.itos.append(t)
                glove.vectors = torch.cat(
                    [glove.vectors, torch.zeros(1, glove.dim)])
                torch.nn.init.normal_(
                    glove.vectors[gloveSize], mean=0, std=0.05)
                gloveSize += 1


def getUnknownTokens(tokenizer, ds):
    # Tokenizer should be the distilBerttokenizer
    # ds should be using the bigram tokenizer
    unknownTokens = []
    for text in dfs[0]['headline']:
        tokens = ds.tokenize(text)
        for t in tokens:
            if tokenizer.convert_tokens_to_ids(t) == 100:
                unknownTokens.append(t)
                tokenizer.add_tokens(t)
    return unknownTokens


def addPOSEmbeds(tokenizer):
    from proj.constants import UPENN_TAGSET
    unknownTokens = []
    for tag in UPENN_TAGSET:
        if tokenizer.convert_tokens_to_ids(tag) == 100:
            unknownTokens.append(tag)
            tokenizer.add_tokens(tag)
    return unknownTokens


if __name__ == "__main__":
    global glove, tokenizer, dfs
    subset_df = pd.read_csv(os.path.join(
        DATA_DIR, "train_test_split_dataset.csv"))
    dfs = split_col(subset_df)
    tokenizer = NewsDataset(subset_df, useBigram=False)
    glove = tokenizer.glove
    representUnknown()
    with open(DL_BIGRAM_GLOVE_EMBEDDINGS, "wb") as outfile:
        pickle.dump(glove, outfile)
