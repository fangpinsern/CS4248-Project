import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import emoji
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from ..constants import X_COL, Y_COL, CATEGORY_SUBSET, MAX_INPUT_LENGTH
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import torchtext

nltk.download("wordnet")
lem = WordNetLemmatizer()
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


def to_dataloader(ds, bs=64, sampler=None):
    dl = DataLoader(
        ds,
        num_workers=torch.cuda.device_count() * 4,
        shuffle=sampler is None,
        sampler=sampler,
        drop_last=True,
        batch_size=bs,
    )
    return dl


class TransformerDataset(Dataset):
    """
    converts text to tensors based on tokenizer provided
    truncates tokens to max length provided
    Do note the max lengths required per transformer
    """

    def __init__(self, tokenizer, df, maxLength=512):
        self.tokenizer = tokenizer
        self.maxLength = maxLength
        self.df = df

    def __len__(self):
        return len(self.df)

    def tokenize(self, text):
        text = emoji.demojize(text, use_aliases=True)
        # Handles emoji parsing that has :emoji_one: syntax
        text = text.replace(r"[_:]", " ")
        return self.tokenizer.encode_plus(
            text,
            return_tensors="pt",
            max_length=self.maxLength,
            truncation=True,
            padding="max_length",
        )

    def __getitem__(self, idx):
        # The T5 model requires summarize token since it is a multi purpose transformer
        text = self.tokenize("summarize: " + self.df.loc[idx, "tracks"])
        target = self.tokenize(self.df.loc[idx, "name"])
        return (
            text["input_ids"].flatten(),
            text["attention_mask"].flatten(),
            target["input_ids"].flatten(),
            target["attention_mask"].flatten(),
        )


class NewsDataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.glove = torchtext.vocab.GloVe(name="6B", dim=50)

    def __len__(self):
        return len(self.df)

    def tokenize(self, text):
        tokens = nltk.word_tokenize(text)
        # tokens = [t.lower() for t in tokens]
        # remove punctuations and stop words
        # lem.lemmatize(t)
        # tokens = [t for t in tokens if re.match(r"\w+", t) and t not in STOPWORDS]
        tokens = [t.lower() for t in tokens if re.match(r"\w+", t)]
        return tokens

    def labels(self):
        return torch.tensor(self.df[Y_COL].apply(lambda d: CATEGORY_SUBSET.index(d)).values)

    def __getitem__(self, idx):
        text = self.df.iloc[idx][X_COL]
        label = torch.tensor(CATEGORY_SUBSET.index(self.df.iloc[idx][Y_COL]))
        tokens = self.tokenize(text)
        number_to_pad = MAX_INPUT_LENGTH - len(tokens)

        stoi_len = len(self.glove.stoi)
        # print(tokens)
        wordIdx = torch.tensor(
            [self.glove.stoi[t] if t in self.glove.stoi else stoi_len for t in tokens],
            dtype=torch.long,
        )
        # print(wordIdx)
        padding = torch.tensor([stoi_len + 1] * number_to_pad, dtype=torch.long)
        wordIdx = torch.cat([wordIdx, padding])
        return wordIdx[:MAX_INPUT_LENGTH], label


def split(df, val_pct=0.2, test_pct=0.2):
    torch.manual_seed(0)
    rand_indices = torch.randperm(len(df))
    num_val = int(val_pct * len(df))
    num_test = int(test_pct * len(df))
    num_train = len(df) - num_val - num_test
    df["phase"] = ""
    trainDf = df.iloc[rand_indices[0:num_train]]
    valDf = df.iloc[rand_indices[num_train : num_train + num_val]]
    testDf = df.iloc[rand_indices[num_train + num_val :]]
    return trainDf, valDf, testDf


if __name__ == "__main__":
    pass