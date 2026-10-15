"""Train the classifier.

    python -m src.train --data-dir path/to/dataset

Writes the best checkpoint and the per-epoch history, so the training curve can
be redrawn later without rerunning anything. The original run lost its history
object when the kernel was interrupted; that is why this writes to disk.
"""

from __future__ import annotations

import argparse
import json

from tensorflow.keras.callbacks import (
    Callback,
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)

from . import config as config_module
from .data import build_generators
from .model import build_model


class HistoryWriter(Callback):
    """Appends each epoch to disk as it finishes, so an interrupted run keeps its history."""

    def __init__(self, path):
        super().__init__()
        self.path = path
        self.rows = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.rows.append(
            {
                "epoch": epoch + 1,
                "accuracy": round(float(logs.get("accuracy", 0)), 4),
                "loss": round(float(logs.get("loss", 0)), 4),
                "val_accuracy": round(float(logs.get("val_accuracy", 0)), 4),
                "val_loss": round(float(logs.get("val_loss", 0)), 4),
                "learning_rate": float(
                    self.model.optimizer.learning_rate.numpy()
                ),
            }
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.rows, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", help="dataset root, one subfolder per class")
    parser.add_argument("--epochs", type=int, help="override config.yaml")
    parser.add_argument("--config", help="path to an alternative config.yaml")
    args = parser.parse_args()

    cfg = config_module.load(args.config)
    if args.data_dir:
        import os

        os.environ["BASIL_DATA_DIR"] = args.data_dir

    train_data, validation_data = build_generators(cfg)
    model = build_model(cfg)
    model.summary()

    params = cfg.training_params
    model_path = cfg.path("model")
    model_path.parent.mkdir(parents=True, exist_ok=True)

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=params["early_stopping_patience"],
            restore_best_weights=True,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=params["reduce_lr_factor"],
            patience=params["reduce_lr_patience"],
        ),
        ModelCheckpoint(model_path, monitor="val_loss", save_best_only=True),
        HistoryWriter(cfg.path("history")),
    ]

    model.fit(
        train_data,
        validation_data=validation_data,
        epochs=args.epochs or params["epochs"],
        callbacks=callbacks,
    )
    print(f"best checkpoint written to {model_path}")


if __name__ == "__main__":
    main()
