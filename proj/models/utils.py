import torch
import pickle
import torchtext
import pandas as pd
from proj.constants import DL_BIGRAM_GLOVE_EMBEDDINGS, DISTILBERT_BIGRAM_TOKENIZER, TRAIN_TEST_SPLIT_FILE, DISTILBERT_EMBED_TOKENIZER, DISTILBERT_POS_TOKENIZER
from proj.utils.data_util import Bigram_Trigram_Tokenizer
from proj.models import all_tokenizers
from proj.data.data import NewsDataset, split_col


def addBigramEmbedsTokenizer():
    tokenizer = all_tokenizers["distilBert"]()
    btTokenizer = Bigram_Trigram_Tokenizer()
    bigramTokens = btTokenizer.get_bigram_token_list()

    for t in bigramTokens:
        if tokenizer.convert_tokens_to_ids(t) == 100:
            tokenizer.add_tokens(t)
    tokenizer.save_pretrained(DISTILBERT_BIGRAM_TOKENIZER)


def addBigramEmbeddings(embedding=None):
    btTokenizer = Bigram_Trigram_Tokenizer()
    bigramTokens = btTokenizer.get_bigram_token_list()

    glove = torchtext.vocab.GloVe(name="6B", dim=50)

    with open(DL_BIGRAM_GLOVE_EMBEDDINGS, "wb") as outfile:
        for t in bigramTokens:
            gloveSize = glove.vectors.shape[0]
            if t not in glove.stoi:
                glove.stoi[t] = gloveSize
                glove.itos.append(t)
                if embedding is None:
                    glove.vectors = torch.cat(
                        [glove.vectors, torch.zeros(1, glove.dim)])
                    torch.nn.init.normal_(
                        glove.vectors[gloveSize], mean=0, std=0.05)
                else:
                    glove.vectors = torch.cat(
                        [glove.vectors, torch.tensor(embedding[t]).unsqueeze(0)])
        pickle.dump(glove, outfile)


def addUnknownTokens(ds, df):
    # Tokenizer should be the distilBerttokenizer
    # ds should be using the bigram tokenizer
    tokenizer = all_tokenizers["distilBert"]()
    unknownTokens = {}
    for text in df['headline']:
        tokens = ds.tokenize(text)
        for t in tokens:
            if tokenizer.convert_tokens_to_ids(t) == 100:
                unknownTokens[t] = unknownTokens.get(t, 0) + 1
                tokenizer.add_tokens(t)
    tokenizer.save_pretrained(DISTILBERT_EMBED_TOKENIZER)
    # return unknownTokens


def addPOSEmbeds():
    from proj.constants import UPENN_TAGSET
    tokenizer = all_tokenizers["distilBert"]()
    unknownTokens = []
    for tag in UPENN_TAGSET:
        if tokenizer.convert_tokens_to_ids(tag) == 100:
            unknownTokens.append(tag)
            tokenizer.add_tokens(tag)
    tokenizer.save_pretrained(DISTILBERT_POS_TOKENIZER)


def representUnknown():
    glove = torchtext.vocab.GloVe(name="6B", dim=50)
    subset_df = pd.read_csv(TRAIN_TEST_SPLIT_FILE)
    dfs = split_col(subset_df)
    tokenizer = NewsDataset(dfs[0])
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


if __name__ == "__main__":
    subset_df = pd.read_csv(TRAIN_TEST_SPLIT_FILE)
    dfs = split_col(subset_df)
    # For LSTM
    addBigramEmbeddings()
    # For transformers
    addBigramEmbedsTokenizer()
    # Transformers adding unkown tokens
    dataset = NewsDataset(dfs[0], tokenizer=all_tokenizers["distilBert"]())
    addUnknownTokens(dataset, dfs[0])
    # Transformers adding POS tags
    addPOSEmbeds()
