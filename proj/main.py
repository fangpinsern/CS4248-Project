# import argparse
import sys
import os
import torch
from tensorboardX import SummaryWriter
from tqdm.auto import tqdm
import numpy as np
import random

from .data.data import PlaylistDataset, to_dataloader
from .constants import LOG_DIR, WEIGHTS_DIR, DATA_DIR
from .models import all_models, all_tokenizers
from .utils import all_opt

sys.path.append(os.path.abspath(__file__))
DEVICE_COUNT = torch.cuda.device_count()
IS_CUDA = torch.cuda.is_available()
SEED = 0

np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

DEFAULT_HP = {
    "epochs": 1,  # number of times we're training on entire dataset
    "metric": "rouge",
    "opt": "ADAM",
    "wd": 0.001,
    "lr": 3e-3,
}


class Trainer:
    def __init__(self, exp_name, df, hp, weights=None, sched=True):
        self.model = all_models[hp["model"]]()
        self.tokenizer = all_tokenizers[hp["model"]]()
        self.device = torch.device("cuda" if IS_CUDA else "cpu")
        opt = all_opt[hp["opt"]]
        self.model.to(self.device)
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
        # self.metric = all_metric[hp["metric"]]
        self.epochs = hp["epochs"]
        self.writer = SummaryWriter(os.path.join(LOG_DIR, exp_name))
        self.exp_name = exp_name
        self.model_name = hp["model_name"]
        self.hp = hp
        # TODO split df into train and val
        self.train_dl = to_dataloader(PlaylistDataset(self.tokenizer, df), bs=4)
        self.eval_dl = to_dataloader(PlaylistDataset(self.tokenizer, df), bs=4)

    def decodeToText(self, embedding):
        gen_text = self.tokenizer.batch_decode(
            embedding, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return gen_text

    def freezeEncoder(self):
        for param in self.model.base_model.parameters():
            param.requires_grad = False

    def unfreeze(self, n):
        for param in self.model.base_model.parameters():
            param.requires_grad = True

    def generate(self, xb, maskX, maskY):
        output = self.model.generate(
            xb,
            attention_mask=maskX,
            use_cache=True,
            decoder_attention_mask=maskY,
            max_length=150,
            num_beams=2,
            repetition_penalty=2.5,
            length_penalty=1.0,
            early_stopping=True,
        )
        return output

    def train(self):
        self.model.train()
        trLoss = 0
        # we use tqdm to provide visual feedback on training
        for data in tqdm(self.train_dl, total=len(self.train_dl)):
            for i in range(len(data)):
                data[i] = data[i].to(self.device)
            xb, maskX, yb, maskY = data
            lm_labels = yb
            lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100
            lm_labels = lm_labels.cuda()
            print(lm_labels.shape)
            self.opt.zero_grad()
            outputs = self.model(
                input_ids=xb,
                attention_mask=maskX,
                lm_labels=lm_labels,
                decoder_attention_mask=maskY,
            )
            loss = outputs[0]

            loss.backward()  # calculates gradient descent
            # preds = self.generate(xb, maskX, maskY)
            # predText = self.decodeToText(preds)
            # yText = self.decodeToText(yb)
            # self.metric.add_batch(predText, yText)
            self.opt.step()  # updates model parameters
            trLoss += loss
            # self._log('train_loss', loss, self.steps[0])

        # rougeScore = self.metric.compute()
        print("\nepoch trng info: loss:{}".format(trLoss))

    def validate(self):
        self.model.eval()
        valLoss = 0
        # we use tqdm to provide visual feedback on training
        with torch.no_grad():  # don't accumulate gradients, faster processing
            for data in tqdm(self.eval_dl, total=len(self.eval_dl)):
                for i in range(len(data)):
                    data[i] = data[i].to(self.device)
                xb, maskX, yb, maskY = data
                lm_labels = yb
                lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100
                outputs = self.model(
                    input_ids=xb,
                    attention_mask=maskX,
                    decoder_attention_mask=maskY,
                    lm_labels=lm_labels,
                )
                loss = outputs[0]
                preds = self.model.generate(
                    xb,
                    attention_mask=maskX,
                    use_cache=True,
                    decoder_attention_mask=maskY,
                    max_length=150,
                    num_beams=2,
                    repetition_penalty=2.5,
                    length_penalty=1.0,
                    early_stopping=True,
                )

                predText = self.decodeToText(preds)
                yText = self.decodeToText(yb)
                # self.metric.add_batch(predText, yText)
                valLoss += loss

        # rougeScore = self.metric.compute()
        print("\nepoch val info: loss:{}, rouge:{}".format(valLoss, 0))

    def one_cycle(self):
        for i in range(self.epochs):
            print("epoch number: {}".format(i))
            self.train()
            self.validate()
            self.scheduler.step()
        # self._write_hp()  # for comparing between experiments

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
                WEIGHTS_DIR, self.exp_name, self.model_name + ".pkl"
            )
            os.makedirs(os.path.join(WEIGHTS_DIR, self.exp_name), exist_ok=True)
            self.model.cpu()
            state = self.model.state_dict()
            torch.save(state, weights_path)  # open(pkl), compress
            self.model.to(self.device)


if __name__ == "__main__":
    import pandas as pd

    df = pd.read_csv(os.path.join(DATA_DIR, "verbose10kdata.csv"))

    hp = {**DEFAULT_HP, "model": "T5", "model_name": "T5_0"}
    trainer = Trainer("T5_finetuned", df, hp)
    trainer.one_cycle()
