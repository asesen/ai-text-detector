# model.py
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC, BinaryF1Score


class TfidfDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = X
        self.y = y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx].toarray(), dtype=torch.float32).squeeze(0)

        if self.y is not None:
            y = torch.tensor(self.y[idx], dtype=torch.float32)
            return x, y

        return x


class TfidfDataModule(pl.LightningDataModule):
    def __init__(self, X_train, y_train, X_val, y_val, cfg):
        super().__init__()
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.cfg = cfg

    def train_dataloader(self):
        return DataLoader(
            TfidfDataset(self.X_train, self.y_train),
            batch_size=self.cfg.training.batch_size,
            shuffle=True,
            num_workers=0,
        )

    def val_dataloader(self):
        return DataLoader(
            TfidfDataset(self.X_val, self.y_val),
            batch_size=self.cfg.training.batch_size,
            shuffle=False,
            num_workers=0,
        )


class TfidfLogReg(pl.LightningModule):
    def __init__(self, input_dim, cfg):
        super().__init__()
        self.save_hyperparameters(ignore=["cfg"])

        self.linear = nn.Linear(input_dim, 1)
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
        return self.linear(x).squeeze(1)

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
