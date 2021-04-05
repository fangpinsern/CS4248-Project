import torch
from torch import nn
import torchtext
import torch.nn.functional as F
import numpy as np

from proj.constants import MAX_INPUT_LENGTH

IS_CUDA = torch.cuda.is_available()


def create_emb_layer(non_trainable=False, embedDims=50):
    np.random.seed(0)
    glove = torchtext.vocab.GloVe(name="6B", dim=embedDims)
    # here the unknown token embedding is a randomly initialized vector based on normal distribution
    # supposedly these glove vectors i.e. the weights of this embedding layer would change during training as well
    randEmbed = torch.tensor(np.random.normal(scale=0.6, size=(embedDims,)))
    randEmbed = randEmbed.unsqueeze(0)
    # the pad token embedding is 0, such that it won't get involved in training
    padEmbed = torch.zeros(size=(embedDims,), dtype=torch.long).unsqueeze(0)
    glove.vectors = torch.cat([glove.vectors, randEmbed, padEmbed])
    numEmbeddings = glove.vectors.shape[0]
    emb_layer = nn.Embedding(*glove.vectors.shape,
                             padding_idx=numEmbeddings - 1)
    emb_layer.load_state_dict({"weight": glove.vectors})
    if non_trainable:
        emb_layer.weight.requires_grad = False

    return emb_layer, glove.vectors.shape[-1]


class newsLSTM(nn.Module):
    def __init__(
        self, batch_size, hidden_dims=128, num_classes=10, num_layers=1, dropout=0.2
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
        self.device = torch.device("cuda" if IS_CUDA else "cpu")
        hidden = torch.zeros(
            (num_layers, batch_size, hidden_dims), device=self.device)
        cell = torch.zeros(
            (num_layers, batch_size, hidden_dims), device=self.device)
        self.hidden = (hidden, cell)
        self.batch_size = batch_size

    def forward(self, input):
        if isinstance(input, torch.FloatTensor):
            print(input)
        embeddings = self.embedding(input)
        input_lengths = [MAX_INPUT_LENGTH] * self.batch_size
        # Pack padded batch of sequences for RNN module
        # packed = nn.utils.rnn.pack_padded_sequence(
        #     embeddings, torch.tensor(input_lengths), batch_first=True
        # )
        # Forward pass through LSTM
        outputs, hidden = self.lstm(embeddings, self.hidden)
        # Unpack padding
        # outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        # get LSTM's block 1's hidden state
        label = self.fc(hidden[0][-1, :, :])
        return F.softmax(label, 1)
