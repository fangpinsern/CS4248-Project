import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from ..constants import (X_COL, Y_COL, CATEGORY_SUBSET,
                         MAX_INPUT_LENGTH, DL_BIGRAM_GLOVE_EMBEDDINGS)
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import torchtext
from proj.utils.data_util import (
    Bigram_Trigram_Tokenizer, tokenize_synonyms, tokenize_hypernyms, augment_synonyms)
import pickle
from torch.nn.utils.rnn import pad_sequence

# nltk.download("wordnet")
lem = WordNetLemmatizer()
bigramTokenizer = Bigram_Trigram_Tokenizer()
STOPWORDS = stopwords.words("english")
STOPWORDS = set(STOPWORDS) | set(ENGLISH_STOP_WORDS)


def class_weights(labels):
    weight = []
    big_i = labels.max().item() + 1
    for i in range(big_i):
        weight.append((labels == i).sum())
    weight = torch.tensor(weight, dtype=torch.float32)
    return len(labels) / weight


def get_weighted_sampler(label_list):
    weights = class_weights(label_list)
    weights_as_idx = weights[label_list]
    weighted_sampler = WeightedRandomSampler(
        weights=weights_as_idx, num_samples=len(weights_as_idx), replacement=True
    )
    return weighted_sampler


def to_dataloader(ds, bs=64, sampler=None, drop_last=True):
    dl = DataLoader(
        ds,
        num_workers=torch.cuda.device_count() * 4,
        sampler=sampler,
        drop_last=drop_last,
        batch_size=bs,
    )
    return dl


class NewsDataset(Dataset):
    def __init__(self, df, tokenizer=None, useBigram=False, synonyms=False, hypernyms=False, stopwords=True, augment=False, tag=False, embed=False):
        self.df = df
        if useBigram:
            with open(DL_BIGRAM_GLOVE_EMBEDDINGS, "rb") as infile:
                self.glove = pickle.load(infile)
        else:
            self.glove = torchtext.vocab.GloVe(name="6B", dim=50)
        self.synonyms = synonyms
        self.hypernyms = hypernyms
        self.stopwords = stopwords
        self.tokenizer = tokenizer
        self.maxLength = MAX_INPUT_LENGTH * 2
        self.useBigram = useBigram
        self.augment = augment
        self.tag = tag
        self.embed = embed

    def getDF(self):
        return self.df.copy()

    def __len__(self):
        return len(self.df)

    def tokenize(self, text):
        if self.synonyms:
            return tokenize_synonyms(text)
        if self.hypernyms:
            return tokenize_hypernyms(text)

        if self.useBigram:
            tokens = bigramTokenizer.tokenize_with_bigrams(text)
        else:
            tokens = nltk.word_tokenize(text)
            tokens = [t.lower() for t in tokens if re.match(r"\w+", t)]
        if not self.stopwords:
            tokens = [t for t in tokens if t not in STOPWORDS]
        if self.augment:
            tokens = augment_synonyms(tokens)
        if self.tag:
            taggedTokens = nltk.pos_tag(tokens)
            # flatten
            tokens = [t if i == 0 else '<' + t.lower() +
                      '>' for tpl in taggedTokens for i, t in enumerate(tpl)]
        # tokens = [t.lower() for t in tokens]
        # lem.lemmatize(t)
        # tokens = [t for t in tokens if re.match(r"\w+", t) and t not in STOPWORDS]
        return tokens

    def labels(self):
        return torch.tensor(
            self.df[Y_COL].apply(lambda d: CATEGORY_SUBSET.index(d)).values
        )

    def __getitem__(self, idx):
        label = torch.tensor(CATEGORY_SUBSET.index(self.df.iloc[idx][Y_COL]))
        text = self.df.iloc[idx][X_COL]

        if self.tokenizer is not None:
            if self.useBigram or self.tag or self.embed:
                text = self.tokenize(text)
            # if self.tag:
            #     tokens = self.tokenizer.tokenize(text)
            #     taggedTokens = nltk.pos_tag(tokens)
            #     # flatten
            #     tokens = [t if i == 0 else '<' + t.lower() +
            #               '>' for tpl in taggedTokens for i, t in enumerate(tpl)]
            tokenDict = self.tokenizer.encode_plus(
                text,
                return_tensors="pt",
                max_length=self.maxLength,
                truncation=True,
                padding="max_length",
                add_special_tokens=True
            )
            tokens = (tokenDict["input_ids"][0],
                      tokenDict["attention_mask"][0])
            return tokens, label

        tokens = self.tokenize(text)
        number_to_pad = MAX_INPUT_LENGTH - len(tokens)

        stoi_len = len(self.glove.stoi)
        # print(tokens)
        wordIdx = torch.tensor(
            [self.glove.stoi[t] if t in self.glove.stoi else stoi_len for t in tokens],
            dtype=torch.long,
        )
        padding = torch.tensor(
            [stoi_len + 1] * number_to_pad, dtype=torch.long)
        wordIdx = torch.cat([wordIdx, padding])
        seqLen = torch.tensor(len(tokens))

        return (wordIdx[:MAX_INPUT_LENGTH], seqLen), label


def split_col(df):
    train = df[df['phase'] == 'train']
    val = df[df['phase'] == 'dev']
    test = df[df['phase'] == 'test']
    return train, val, test


def split(df, val_pct=0.2, test_pct=0.1):
    torch.manual_seed(0)
    rand_indices = torch.randperm(len(df))
    num_val = int(val_pct * len(df))
    num_test = int(test_pct * len(df))
    num_train = len(df) - num_val - num_test
    df["phase"] = ""
    trainDf = df.iloc[rand_indices[0:num_train]]
    valDf = df.iloc[rand_indices[num_train: num_train + num_val]]
    testDf = df.iloc[rand_indices[num_train + num_val:]]
    return trainDf, valDf, testDf


if __name__ == "__main__":
    pass
