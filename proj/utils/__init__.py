import torch

# from nlp import load_metric


all_opt = {"SGD": torch.optim.SGD, "ADAM": torch.optim.Adam}

# all_metric = {"rouge": load_metric("rouge")}


def accuracy(y_hat, y):
    preds = torch.argmax(y_hat, dim=1)
    return (preds == y).sum().item()
