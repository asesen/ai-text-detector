# model.py
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC, BinaryF1Score


class BinaryClassNN(pl.LightningModule):
    def __init__(self, input_dim, cfg):
        super().__init__()
        self.save_hyperparameters(ignore=["cfg"])

        self.net = nn.Sequential(
            nn.Linear(input_dim, cfg.training.hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.training.dropout),
            nn.Linear(cfg.training.hidden_dim, 1),
        )
        self.criterion = nn.BCEWithLogitsLoss()

        self.val_acc = BinaryAccuracy()
        self.val_auc = BinaryAUROC()
        self.val_f1 = BinaryF1Score()

        self.cfg = cfg

        # история
        self.train_loss_history = []
        self.val_metrics_history = {
            "val_acc": [],
            "val_auc": [],
            "val_f1": [],
        }

    def forward(self, x):
        return self.net(x).squeeze(1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        self.train_loss_history.append(loss.item())
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        probs = torch.sigmoid(logits)

        self.val_acc.update(probs, y.int())
        self.val_auc.update(probs, y.int())
        self.val_f1.update(probs, y.int())

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        acc = self.val_acc.compute()
        auc = self.val_auc.compute()
        f1 = self.val_f1.compute()

        self.log("val_acc", acc, prog_bar=True)
        self.log("val_auc", auc, prog_bar=True)
        self.log("val_f1", f1, prog_bar=True)

        self.val_metrics_history["val_acc"].append(acc.item())
        self.val_metrics_history["val_auc"].append(auc.item())
        self.val_metrics_history["val_f1"].append(f1.item())

        self.val_acc.reset()
        self.val_auc.reset()
        self.val_f1.reset()

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.cfg.training.lr,
        )
