# import argparse
import os
import torch
from tensorboardX import SummaryWriter
from tqdm.auto import tqdm  # auto adjust to notebook and terminal
import sklearn.metrics as skMetrics
from torch.nn.functional import cross_entropy

import numpy as np
import random

from proj.models import all_models
from proj.utils import all_loss, all_opt, accuracy
from proj.constants import WEIGHTS_DIR, LOG_DIR, SEED, DATA_DIR, CATEGORY_SUBSET, PREDS_DIR, Y_COL, PRED_COL
from proj.data.data import NewsDataset, split, to_dataloader


DEVICE_COUNT = torch.cuda.device_count()
IS_CUDA = torch.cuda.is_available()

np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

DEFAULT_HP = {
    "epochs": 5,  # number of times we're training on entire dataset
    "loss": "cross_entropy",
    "opt": "ADAM",
    "wd": 0.00001,
    "lr": 1e-3,
}

# TODO
"""
- [x] calculate and log f1_score 
- [ ] generate and log confusion matrix
- [ ] only set non bias and norm weights to trainable
- [x] add getPreds method for all rows in dataframe
"""
PHASES = ["train", "val", "test", "train"]


class Trainer:
    def __init__(self, exp_name, model_name, dls, hp, bs, sched=False):
        self.device = torch.device("cuda" if IS_CUDA else "cpu")
        self.model = all_models[hp["model"]](bs).to(self.device)
        self.loss = all_loss[hp["loss"]]
        self.epochs = hp["epochs"]
        self.writer = SummaryWriter(
            os.path.join(LOG_DIR, exp_name, model_name))
        self.exp_name = exp_name
        self.hp = hp
        self.metrics = {}
        for p in ["train", "val", "test"]:
            self.metrics[p] = [[], [], []]
        self.dls = dls
        self.steps = [0] * 3
        self.batch_size = dls[0].batch_size
        self.model_name = model_name
        opt = all_opt[hp["opt"]]
        parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        # no_decay = ['bias', 'LayerNorm.weight']
        # optimizer_grouped_parameters = [
        #     {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        #     {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        # ]
        # optimizer = AdamW(optimizer_grouped_parameters, lr=hp["lr"])
        if hp["opt"] == "ADAM":
            self.opt = opt(params=parameters, lr=hp["lr"])
        else:
            self.opt = opt(
                params=parameters,
                lr=hp["lr"],
                momentum=0.9,
                weight_decay=hp["wd"],
            )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.opt, step_size=2, gamma=0.1 if sched else 1
        )
        self.isTransformer = dls[0].dataset.tokenizer is not None
        # self.class_names = dls[0].dataset.class_names
        # self.cms = {0: None, 1: None, 2: None}
        # self.auc = {0: None, 1: None, 2: 0}
        # self.recall = {0: None, 1: None, 2: [0 for i in (self.class_names)]}

    def anEpoch(self, phaseIndex, toLog=True):
        phaseName = PHASES[phaseIndex]
        losses = []
        acc_count = 0
        allPreds, allLabels = [], []
        # we use tqdm to provide visual feedback on training stage
        for xb, yb in tqdm(self.dls[phaseIndex], total=len(self.dls[phaseIndex])):
            if self.isTransformer:
                inputIds, mask = xb
                yb = yb.to(self.device)
                outputs = self.model(
                    inputIds.to(self.device),
                    attention_mask=mask.to(self.device),
                    labels=yb,
                )
                loss = outputs[0]
                output = outputs[1]
                inputIds.detach().cpu()
                mask.detach().cpu()
                yb.detach().cpu()
            else:
                xb = xb.to(self.device)  # BATCH_SIZE, 3, 224, 224
                yb = yb.to(self.device)  # BATCH_SIZE, 1
                output = self.model(xb)  # BATCH_SIZE, 3
                loss = self.loss(output, yb)
                xb.detach().cpu()
                yb.detach().cpu()
            allPreds.append(torch.argmax(output, dim=1).cpu())
            allLabels.append(yb.cpu())

            acc_count += accuracy(output, yb)
            losses.append(loss)
            self.steps[phaseIndex] += 1
            if toLog:
                self._log("{}_loss".format(phaseName),
                          loss, self.steps[phaseIndex])

            if phaseIndex == 0:
                self.opt.zero_grad()
                loss.backward()  # calculates gradient descent
                self.opt.step()  # updates model parameters
        allPreds = torch.cat(allPreds)
        allLabels = torch.cat(allLabels)
        f1Score = skMetrics.f1_score(
            allLabels.cpu(), allPreds.cpu(), average="macro")
        losses = torch.stack(losses)
        epoch_loss = losses.mean().item()
        epoch_acc = acc_count / len(self.dls[phaseIndex]) / self.batch_size
        self.metrics[phaseName][0].append(epoch_loss)
        self.metrics[phaseName][1].append(epoch_acc)
        self.metrics[phaseName][2].append(f1Score)
        print(
            "\nepoch {} info: loss:{}, acc:{}, f1Score:{}".format(
                phaseName, epoch_loss, epoch_acc, f1Score
            )
        )
        return allPreds, allLabels

    def topKLoss(self, phaseIndex, k):
        lossValues = []
        # we use tqdm to provide visual feedback on training stage
        with torch.no_grad():
            for xb, yb in tqdm(self.dls[phaseIndex], total=len(self.dls[phaseIndex])):
                if self.isTransformer:
                    inputIds, mask = xb
                    yb = yb.to(self.device)
                    outputs = self.model(
                        inputIds.to(self.device),
                        attention_mask=mask.to(self.device),
                        labels=yb,
                    )
                    output = outputs[1]
                    inputIds.detach().cpu()
                    mask.detach().cpu()
                    yb.detach().cpu()
                else:
                    xb = xb.to(self.device)  # BATCH_SIZE, 3, 224, 224
                    yb = yb.to(self.device)  # BATCH_SIZE, 1
                    output = self.model(xb)  # BATCH_SIZE, 3
                    xb.detach().cpu()
                    yb.detach().cpu()
                lossValues.append(cross_entropy(
                    output, yb, reduction='none').cpu())
        lossValues = torch.cat(lossValues)
        return torch.topk(lossValues, k=k)

    def one_cycle(self):
        # self.freeze()
        for i in range(self.epochs):
            print("epoch number: {}".format(i))
            self.model.train()
            self.anEpoch(0)
            with torch.no_grad():
                self.model.eval()
                self.anEpoch(1)
            self.scheduler.step()
            self._save_weights()
        self.load_weights(self.model_name + ".pkl")
        if len(self.dls) > 2 and len(self.dls[2]) > 0:
            with torch.no_grad():
                self.model.eval()
                self.anEpoch(2)
        metrics = {}
        for i in range(3):
            metrics.update(self.getMetrics(i))
        self._write_hp(metrics)  # for comparing between experiments

    def freeze(self, toTrain=False):
        if self.isTransformer:
            for param in self.model.base_model.parameters():
                param.requires_grad = toTrain
            return
        for p in self.model.embedding.parameters():
            p.requires_grad = toTrain
        for p in self.model.lstm.parameters():
            p.requires_grad = toTrain

    def getMetrics(self, type):
        phases = ["train", "val", "test"]
        phase = phases[type]
        phaseMetrics = self.metrics[phases[type]]
        metricValues = [min(phaseMetrics[0]), max(
            phaseMetrics[1]), max(phaseMetrics[2])]
        metrics = {}
        for i, metricName in enumerate(["loss", "acc", "f1score"]):
            metricName = f"{phase}_{metricName}"
            metrics[metricName] = metricValues[i]
        return metrics

    def _log(self, phase, value, i):
        self.writer.add_scalar(tag=phase, scalar_value=value, global_step=i)

    def _write_hp(self, metrics):
        self.writer.add_hparams(self.hp, metrics)

    def setLR(self, lr):
        self.opt.param_groups[0]['lr'] = lr

    def load_weights(self, pkl_name, num_classes=None, family=None):
        weights_path = os.path.join(WEIGHTS_DIR, self.exp_name, pkl_name)
        sd = torch.load(weights_path)
        self.model.load_state_dict(sd, strict=False)
        self.model.to(self.device)

    def _save_weights(self):
        bestF1Score = max(self.metrics["val"][-1])
        if self.metrics["val"][-1][-1] == bestF1Score:
            weights_path = os.path.join(
                WEIGHTS_DIR, self.exp_name, self.model_name + ".pkl"
            )
            os.makedirs(os.path.join(
                WEIGHTS_DIR, self.exp_name), exist_ok=True)
            self.model.cpu()
            state = self.model.state_dict()
            torch.save(state, weights_path)  # open(pkl), compress
            self.model.to(self.device)

    def getPreds(self, phaseIdx, toSave=False):
        with torch.no_grad():
            preds, _ = self.anEpoch(phaseIdx, toLog=False)
        if not toSave:
            return preds
        dfCopy = self.dls[phaseIdx].dataset.getDF()
        if len(preds) < len(dfCopy):
            extra = len(dfCopy) - len(preds)
            preds = torch.cat([preds, torch.tensor([-1] * extra)])
        predCategories = list(map(lambda l: CATEGORY_SUBSET[l], preds.numpy()))
        dfCopy[PRED_COL] = predCategories
        dfCopy["correct"] = dfCopy[PRED_COL] == dfCopy[Y_COL]
        csvPath = os.path.join(
            PREDS_DIR, f"{self.model_name}_{PHASES[phaseIdx]}_preds.csv")
        dfCopy.to_csv(csvPath, index=False)
        return preds


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
