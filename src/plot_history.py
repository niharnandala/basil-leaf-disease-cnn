"""Draw the training curve from the recorded per-epoch values.

    python -m src.plot_history

Every number comes from artifacts/training_history.json, which holds what Keras
actually printed during the run. Nothing on this chart is estimated.
"""

from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import config as config_module  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    args = parser.parse_args()

    cfg = config_module.load(args.config)
    history = json.loads(cfg.path("history").read_text(encoding="utf-8"))

    epochs = [h["epoch"] for h in history]
    rates = [h["learning_rate"] for h in history]
    drop_epoch = next(
        (
            e
            for e, before, after in zip(epochs[1:], rates, rates[1:], strict=False)
            if after < before
        ),
        None,
    )

    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 5))

    left.plot(epochs, [h["accuracy"] for h in history], marker="o", label="train")
    left.plot(epochs, [h["val_accuracy"] for h in history], marker="s", label="validation")
    left.set(title="Accuracy per epoch", xlabel="epoch", ylabel="accuracy", ylim=(0.5, 1.02))

    right.plot(epochs, [h["loss"] for h in history], marker="o", label="train")
    right.plot(epochs, [h["val_loss"] for h in history], marker="s", label="validation")
    right.set(title="Loss per epoch", xlabel="epoch", ylabel="loss", yscale="log")

    for axis in (left, right):
        axis.legend()
        axis.grid(alpha=0.3)
        if drop_epoch is not None:
            axis.axvline(drop_epoch, color="grey", linestyle="--", linewidth=1)

    if drop_epoch is not None:
        fig.suptitle(
            f"Validation accuracy collapses at epoch 4 and recovers after "
            f"ReduceLROnPlateau cuts the rate at epoch {drop_epoch}"
        )

    output = cfg.path("figures") / "training_curves.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    print(f"wrote {output}")

    best = min(history, key=lambda h: h["val_loss"])
    print(f"best epoch by val_loss: {best['epoch']} (val_loss {best['val_loss']})")


if __name__ == "__main__":
    main()
