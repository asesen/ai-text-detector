import subprocess
from pathlib import Path

import mlflow
import pandas as pd
import pytorch_lightning as pl
import torch
from ai_text_detector.data import TextDataModule
from ai_text_detector.gdrive import download_dataset_folder
from ai_text_detector.model import DistilBERTClassifier
from ai_text_detector.plots import plot_training_curves
from transformers import AutoTokenizer


def train(cfg):
    pl.seed_everything(cfg.seed, workers=True)

    mlflow.set_tracking_uri(cfg.logging.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.logging.mlflow.experiment_name)

    with mlflow.start_run():
        git_commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
        )

        mlflow.log_param("git_commit", git_commit)

        raw_dir = download_dataset_folder(
            cfg.data.gdrive_folder_id, Path(cfg.data.raw_data_dir)
        )
        train_df = pd.read_csv(raw_dir / cfg.data.files.train)
        val_df = pd.read_csv(raw_dir / cfg.data.files.val)

        tokenizer = AutoTokenizer.from_pretrained(cfg.preprocessing.tokenizer.name)

        dm = TextDataModule(train_df, val_df, tokenizer, cfg)
        model = DistilBERTClassifier(cfg)

        trainer = pl.Trainer(
            max_epochs=cfg.training.training.epochs, accelerator="gpu", devices=1
        )

        trainer.fit(model, dm)

        plots_dir = Path("plots")
        plot_training_curves(model, plots_dir)

        mlflow.log_artifacts(str(plots_dir))

        onnx_path = Path("models/onnx/model.onnx")
        onnx_path.parent.mkdir(parents=True, exist_ok=True)

        dummy = {
            "input_ids": torch.ones(1, 32, dtype=torch.long).cuda(),
            "attention_mask": torch.ones(1, 32, dtype=torch.long).cuda(),
        }

        torch.onnx.export(
            model.cuda(),
            (dummy["input_ids"], dummy["attention_mask"]),
            onnx_path,
            opset_version=17,
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={"input_ids": {0: "batch"}, "attention_mask": {0: "batch"}},
        )

        mlflow.log_artifact(str(onnx_path))

        mlflow.pytorch.log_model(model, "model")
