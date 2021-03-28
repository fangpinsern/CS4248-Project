import torch
from torch import nn
import torchtext
import torch.nn.functional as F
import numpy as np


def create_emb_layer(non_trainable=False):
    np.random.seed(0)
    glove = torchtext.vocab.GloVe(name="6B", dim=50)
    randEmbed = torch.tensor(np.random.normal(scale=0.6, size=(50,))).unsqueeze(0)
    glove.vectors = torch.cat([glove.vectors, randEmbed])
    emb_layer = nn.Embedding(*glove.vectors.shape)
    emb_layer.load_state_dict({"weight": glove.vectors})
    if non_trainable:
        emb_layer.weight.requires_grad = False

    return emb_layer, glove.vectors.shape[-1]


class newsLSTM(nn.Module):
    def __init__(
        self, batch_size, hidden_dims=256, num_classes=10, num_layers=4, dropout=0.2
    ):
        super().__init__()
        self.embedding, embed_dims = create_emb_layer(True)
        kwargs = {
            "input_size": embed_dims,
            "hidden_size": hidden_dims,
            "num_layers": num_layers,
            "dropout": dropout,
            "batch_first": True,
        }
        self.lstm = torch.nn.LSTM(**kwargs)
        self.drop = nn.Dropout(p=0.3)
        self.fc = nn.Linear(hidden_dims, num_classes)
        hidden = torch.zeros((num_layers, batch_size, hidden_dims))
        self.hidden = (hidden, hidden)

    def forward(self, input):
        embeddings = self.embedding(input)
        print(embeddings.shape)
        _, hidden = self.lstm(embeddings, self.hidden)
        # get LSTM's block 1's hidden state
        label = self.fc(hidden[0][-1, :, :])
        return F.softmax(label, 1)
