# import argparse
import os
import torch
from tensorboardX import SummaryWriter
from tqdm.auto import tqdm  # auto adjust to notebook and terminal

from proj.models import all_models
from proj.utils import all_loss, all_opt, accuracy
from proj.constants import WEIGHTS_DIR, LOG_DIR, SEED, DATA_DIR
from proj.data.data import NewsDataset, split, to_dataloader
from sklearn.metrics import (
    roc_curve,
    auc,
    roc_auc_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import numpy as np
import random

DEVICE_COUNT = torch.cuda.device_count()
IS_CUDA = torch.cuda.is_available()

np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

DEFAULT_HP = {
    "epochs": 5,  # number of times we're training on entire dataset
    "loss": "cross_entropy",
    "opt": "SGD",
    "wd": 0.001,
    "lr": 3e-3,
}


class Trainer:
    def __init__(self, exp_name, dls, hp, bs, weights=None, sched=True):
        self.model = all_models[hp["model"]](bs)
        self.device = torch.device("cuda" if IS_CUDA else "cpu")
        opt = all_opt[hp["opt"]]
        if hp["opt"] == "ADAM":
            self.opt = opt(
                params=self.model.parameters(), lr=hp["lr"], weight_decay=hp["wd"]
            )
        else:
            self.opt = opt(
                params=self.model.parameters(),
                lr=hp["lr"],
                momentum=0.9,
                weight_decay=hp["wd"],
            )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.opt, step_size=2, gamma=0.1 if sched else 1
        )
        if weights is not None:
            weights.to(self.device)
        self.loss = all_loss[hp["loss"]]
        self.epochs = hp["epochs"]
        self.model.to(self.device)
        self.writer = SummaryWriter(os.path.join(LOG_DIR, exp_name, hp["model"]))
        self.exp_name = exp_name
        self.hp = hp
        self.metrics = {}
        self.dls = dls
        self.steps = [0] * 3
        # self.class_names = dls[0].dataset.class_names
        # self.cms = {0: None, 1: None, 2: None}
        # self.auc = {0: None, 1: None, 2: 0}
        # self.recall = {0: None, 1: None, 2: [0 for i in (self.class_names)]}

    def train(self):
        losses = []
        # it turns dropout
        self.model.train()
        acc_count = 0
        # we use tqdm to provide visual feedback on training
        for xb, yb in tqdm(self.dls[0], total=len(self.dls[0])):
            xb = xb.to(self.device)  # BATCH_SIZE, 3, 224, 224
            yb = yb.to(self.device)  # BATCH_SIZE, 1
            self.opt.zero_grad()
            output = self.model(xb)  # BATCH_SIZE, 3
            acc_count += accuracy(output, yb)
            loss = self.loss(output, yb)
            loss.backward()  # calculates gradient descent
            self.opt.step()  # updates model parameters
            losses.append(loss)
            self.steps[0] += 1
            self._log("train_loss", loss, self.steps[0])
        losses = torch.stack(losses)
        epoch_loss = losses.mean().item()
        epoch_acc = acc_count / len(self.dls[0])
        self.trng_loss.append(epoch_loss)
        self.trng_acc.append(epoch_acc)
        print("\nepoch trng info: loss:{}, acc:{}".format(epoch_loss, epoch_acc))

    def validate(self):
        losses = []
        acc_count = 0
        with torch.no_grad():  # don't accumulate gradients, faster processing
            self.model.eval()  # ignore dropouts and weight decay
            for xb, yb in tqdm(self.dls[1], total=len(self.dls[1])):
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                output = self.model(xb)
                acc_count += accuracy(output, yb)
                loss = self.loss(output, yb)
                losses.append(loss)
                self.steps[1] += 1
                self._log("val_loss", loss, self.steps[1], writer=1)
        losses = torch.stack(losses)
        epoch_loss = losses.mean().item()
        epoch_acc = acc_count / len(self.dls[1])
        self.val_loss.append(epoch_loss)
        self.val_acc.append(epoch_acc)
        print("\nepoch val info: loss:{}, acc:{}".format(epoch_loss, epoch_acc))

    def test(self):
        losses = []
        acc_count = 0
        with torch.no_grad():  # don't accumulate gradients, faster processing
            self.model.eval()  # ignore dropouts and weight decay
            for xb, yb in tqdm(self.dls[-1], total=len(self.dls[-1])):
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                output = self.model(xb)
                acc_count += accuracy(output, yb)
                loss = self.loss(output, yb)
                losses.append(loss)

        if len(losses) > 0:
            losses = torch.stack(losses)
            epoch_loss = losses.mean().item()
            epoch_acc = acc_count / len(self.dls[-1])
            self.test_loss.append(epoch_loss)
            self.test_acc.append(epoch_acc)
            print("\nepoch test info: loss:{}, acc:{}".format(epoch_loss, epoch_acc))

    def one_cycle(self):
        for i in range(self.epochs):
            print("epoch number: {}".format(i))
            self.train()
            self.validate()
            self.scheduler.step()
            self._save_weights()
        self.test()
        self._write_hp()  # for comparing between experiments

    def _log(self, phase, value, i, writer=0):
        if writer == 0:
            self.writer.add_scalar(tag=phase, scalar_value=value, global_step=i)
        else:
            self.writer_val.add_scalar(tag=phase, scalar_value=value, global_step=i)

    def _write_hp(self):
        val_loss, val_acc = min(self.val_loss), max(self.val_acc)
        test_loss, test_acc = min(self.test_loss), max(self.test_acc)

        metric_names = ["val_loss", "val_acc", "test_loss", "test_acc"]
        metric_values = [val_loss, val_acc, test_loss, test_acc]
        metrics = dict(zip(metric_names, metric_values))
        self.writer.add_hparams(self.hp, metrics)

    def load_weights(self, pkl_name, num_classes=None, family=None):
        weights_path = os.path.join(WEIGHTS_DIR, self.exp_name, pkl_name)
        sd = torch.load(weights_path)
        self.model.load_state_dict(sd)
        if num_classes is not None and family is not None:
            if family == "densenet":
                temp = self.model.classifier.in_features
                self.model.classifier = torch.nn.Linear(temp, num_classes)
            else:
                temp = self.model.fc.in_features
                self.model.fc = torch.nn.Linear(temp, num_classes)
        self.model.to(self.device)

    def _save_weights(self):
        best_val = min(self.val_loss)
        if self.val_loss[-1] == best_val:
            weights_path = os.path.join(
                WEIGHTS_DIR, self.exp_name, self.hp["model"] + ".pkl"
            )
            os.makedirs(os.path.join(WEIGHTS_DIR, self.exp_name), exist_ok=True)
            self.model.cpu()
            state = self.model.state_dict()
            torch.save(state, weights_path)  # open(pkl), compress
            self.model.to(self.device)


if __name__ == "__main__":
    import pandas as pd

    subset_df = pd.read_csv(os.path.join(DATA_DIR, "subsetNews.csv"))
    dfs = split(subset_df)
    dls = []
    bs = 1
    for d in dfs:
        ds = NewsDataset(d)
        dl = to_dataloader(ds, bs)
        dls.append(dl)
    model = "lstm"
    hp = {**DEFAULT_HP, "model": model}
    trainer = Trainer("deep learning", dls, hp, bs)
    trainer.one_cycle()
