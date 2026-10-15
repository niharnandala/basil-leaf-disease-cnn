"""Score the trained model on the validation split.

    python -m src.evaluate

Writes a confusion matrix and a classification report. Both are computed from
the model's predictions, never estimated.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from . import config as config_module
from .data import build_generators


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir")
    parser.add_argument("--config")
    args = parser.parse_args()

    cfg = config_module.load(args.config)
    if args.data_dir:
        import os

        os.environ["BASIL_DATA_DIR"] = args.data_dir

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    import tensorflow as tf
    from sklearn.metrics import classification_report, confusion_matrix

    model_path = cfg.path("model")
    if not model_path.exists():
        raise SystemExit(f"no trained model at {model_path}. Run `python -m src.train` first.")

    model = tf.keras.models.load_model(model_path)
    _, validation_data = build_generators(cfg)

    probabilities = model.predict(validation_data, verbose=1)
    predicted = np.argmax(probabilities, axis=1)
    actual = validation_data.classes[: len(predicted)]

    report = classification_report(
        actual, predicted, target_names=cfg.classes, digits=4
    )
    print(report)

    figures = cfg.path("figures")
    figures.mkdir(parents=True, exist_ok=True)
    (figures / "classification_report.txt").write_text(report, encoding="utf-8")

    matrix = confusion_matrix(actual, predicted)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=cfg.classes,
        yticklabels=cfg.classes,
    )
    plt.xlabel("predicted")
    plt.ylabel("actual")
    plt.title("Confusion matrix, validation split")
    plt.tight_layout()
    plt.savefig(figures / "confusion_matrix.png", dpi=150)
    print(f"wrote {figures / 'confusion_matrix.png'}")

    (figures / "confusion_matrix.json").write_text(
        json.dumps(matrix.tolist()), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
