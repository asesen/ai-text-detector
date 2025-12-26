# train.py
import subprocess
from pathlib import Path

import joblib
import pandas as pd
import pytorch_lightning as pl
import torch
from ai_text_detector.gdrive import download_data_from_gdrive
from ai_text_detector.model import TfidfDataModule, TfidfLogReg
from ai_text_detector.plots import plot_training_curves
from pytorch_lightning.loggers import MLFlowLogger
from sklearn.feature_extraction.text import TfidfVectorizer


def train(cfg):
    pl.seed_everything(cfg.seed, workers=True)

    mlf_logger = MLFlowLogger(
        experiment_name=cfg.logging.mlflow.experiment_name,
        tracking_uri=cfg.logging.mlflow.tracking_uri,
    )
    exp = mlf_logger.experiment
    run_id = mlf_logger.run_id
    # ======================
    # GIT COMMIT
    # ======================
    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    exp.log_param(run_id, "git_commit", git_commit)

    # ======================
    # LOAD DATA
    # ======================
    download_data_from_gdrive(cfg)

    raw_dir = Path(cfg.data.data_dir)
    train_df = pd.read_csv(raw_dir / cfg.data.gdrive_files.train.name)
    val_df = pd.read_csv(raw_dir / cfg.data.gdrive_files.val.name)

    # ======================
    # TF-IDF
    # ======================
    vectorizer = TfidfVectorizer(
        max_features=cfg.preprocessing.tfidf.max_features,
        ngram_range=tuple(cfg.preprocessing.tfidf.ngram_range),
        stop_words=cfg.preprocessing.tfidf.stop_words,
        sublinear_tf=True,
    )

    X_train = vectorizer.fit_transform(
        train_df[cfg.preprocessing.dataset.text_column].values
    ).astype("float32")
    X_val = vectorizer.transform(
        val_df[cfg.preprocessing.dataset.text_column].values
    ).astype("float32")

    y_train = train_df[cfg.preprocessing.dataset.label_column].values
    y_val = val_df[cfg.preprocessing.dataset.label_column].values

    # сохраняем tf-idf (tf-idx)
    Path("models").mkdir(exist_ok=True)
    joblib.dump(vectorizer, "models/tfidf.joblib")

    # ======================
    # DATAMODULE + MODEL
    # ======================
    dm = TfidfDataModule(X_train, y_train, X_val, y_val, cfg)

    model = TfidfLogReg(
        input_dim=X_train.shape[1],
        cfg=cfg,
    )

    trainer = pl.Trainer(
        max_epochs=cfg.training.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        log_every_n_steps=50,
        logger=mlf_logger,
    )

    trainer.fit(model, dm)
    # ======================
    # PLOTS
    # ======================
    plots_dir = Path("plots")
    plot_training_curves(model, plots_dir)
    exp.log_artifacts(run_id, str(plots_dir))

    # ======================
    # ONNX
    # ======================
    onnx_path = Path("models/onnx/model.onnx")
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    model.cpu()

    dummy = torch.randn(1, X_train.shape[1], dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        opset_version=17,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}},
    )
