import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset


class TextDataset(Dataset):
    def __init__(self, df, tokenizer, cfg):
        self.texts = df[cfg.dataset.text_column].values
        self.labels = df[cfg.dataset.label_column].values
        self.tokenizer = tokenizer
        self.cfg = cfg

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=self.cfg.tokenizer.truncation,
            padding=self.cfg.tokenizer.padding,
            max_length=self.cfg.tokenizer.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.float),
        }


class TextDataModule(pl.LightningDataModule):
    def __init__(self, train_df, val_df, tokenizer, cfg):
        super().__init__()
        self.train_df = train_df
        self.val_df = val_df
        self.tokenizer = tokenizer
        self.cfg = cfg

    def train_dataloader(self):
        return DataLoader(
            TextDataset(self.train_df, self.tokenizer, self.cfg.preprocessing),
            batch_size=self.cfg.training.training.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            TextDataset(self.val_df, self.tokenizer, self.cfg.preprocessing),
            batch_size=self.cfg.training.training.batch_size,
            shuffle=False,
            num_workers=4,
        )
