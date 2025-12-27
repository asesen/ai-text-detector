import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset


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
