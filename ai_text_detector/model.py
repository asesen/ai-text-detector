import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC, BinaryF1Score
from transformers import AutoModel


class DistilBERTClassifier(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)

        self.encoder = AutoModel.from_pretrained(cfg.preprocessing.tokenizer.name)

        if cfg.training.model.freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        h = self.encoder.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(h, h),
            nn.LayerNorm(h),
            nn.ReLU(),
            nn.Dropout(cfg.training.model.dropout),
            nn.Linear(h, 1),
        )

        self.loss_fn = nn.BCEWithLogitsLoss()

        self.val_acc = BinaryAccuracy()
        self.val_f1 = BinaryF1Score()
        self.val_auc = BinaryAUROC()

        self.train_losses = []
        self.val_aucs = []

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]
        return self.classifier(cls).squeeze(-1)

    def training_step(self, batch, _):
        logits = self(batch["input_ids"], batch["attention_mask"])
        loss = self.loss_fn(logits, batch["labels"])
        self.log("train_loss", loss, prog_bar=True)
        self.train_losses.append(loss.detach().cpu())
        return loss

    def validation_step(self, batch, _):
        logits = self(batch["input_ids"], batch["attention_mask"])
        probs = torch.sigmoid(logits)
        labels = batch["labels"].int()

        self.val_acc.update(probs, labels)
        self.val_f1.update(probs, labels)
        self.val_auc.update(probs, labels)

    def on_validation_epoch_end(self):
        acc = self.val_acc.compute()
        f1 = self.val_f1.compute()
        auc = self.val_auc.compute()

        self.log_dict({"val_acc": acc, "val_f1": f1, "val_auc": auc}, prog_bar=True)
        self.val_aucs.append(auc.cpu())

        self.val_acc.reset()
        self.val_f1.reset()
        self.val_auc.reset()

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.classifier.parameters(), lr=self.hparams.training.training.lr
        )
