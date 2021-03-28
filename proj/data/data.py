import torch
from torch.utils.data import Dataset, DataLoader
import emoji
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from ..constants import X_COL, Y_COL, CATEGORY_SUBSET
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import torchtext

nltk.download("wordnet")
lem = WordNetLemmatizer()
STOPWORDS = stopwords.words("english")
STOPWORDS = set(STOPWORDS) | set(ENGLISH_STOP_WORDS)


def to_dataloader(ds, bs=64):
    dl = DataLoader(
        ds,
        num_workers=torch.cuda.device_count() * 4,
        # shuffle=train,
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
        tokens = [t.lower() for t in tokens]
        # remove punctuations and stop words
        tokens = [
            lem.lemmatize(t)
            for t in tokens
            if re.match(r"\w+", t) and t not in STOPWORDS
        ]
        return tokens

    def __getitem__(self, idx):
        text = self.df.iloc[idx][X_COL]
        label = torch.tensor(CATEGORY_SUBSET.index(self.df.iloc[idx][Y_COL]))
        tokens = self.tokenize(text)
        wordIdx = torch.tensor(
            [self.glove.stoi[t] if t in self.glove.stoi else 40001 for t in tokens]
        )
        return wordIdx, label


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