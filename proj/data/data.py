import os
import json
import pandas as pd
from multiprocessing import Pool
import torch
from torch.utils.data import Dataset, DataLoader
import emoji
from ..constants import JSON_DIR, JSON_FILES


def to_dataloader(ds, bs=64):
    dl = DataLoader(
        ds,
        num_workers=torch.cuda.device_count() * 4,
        # shuffle=train,
        drop_last=False,
        batch_size=bs,
    )
    return dl


def readJson(j):
    df = pd.DataFrame(columns=dfCols)
    with open(os.path.join(JSON_DIR, j), "rb") as infile:
        try:
            jsonIn = json.load(infile)
        except Exception:
            print("file error", j)
            return
        for i, playlist in enumerate(jsonIn["playlists"]):
            allTrackData = playlist["tracks"]
            tracks = map(lambda d: d["track_name"], allTrackData)
            albums = map(lambda d: d["album_name"], allTrackData)
            artists = map(lambda d: d["artist_name"], allTrackData)
            playlist["tracks"] = "\n".join(tracks)
            playlist["albums"] = "\n".join(albums)
            playlist["artists"] = "\n".join(artists)
            values = [j, i] + list(playlist.values())
            df = df.append(dict(zip(dfCols, values)), ignore_index=True)
    print(j, "done")
    return df


class PlaylistDataset(Dataset):
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


if __name__ == "__main__":
    global dfCols

    with open(os.path.join(JSON_DIR, JSON_FILES[1]), "rb") as infile:
        samplePlaylist = json.load(infile)["playlists"]
        samplePlaylist[0]["albums"] = 0
        samplePlaylist[0]["artists"] = 0
        dfCols = ["jsonfile", "index"] + list(samplePlaylist[0].keys())

    res = []
    with Pool(6) as p:
        for j in JSON_FILES:
            jsonData = p.apply_async(readJson, args=[j])
            res.append(jsonData)

        allDf = pd.concat([r.get() for r in res])
        allDf.to_csv("verboseAlldata.csv", index=False)
