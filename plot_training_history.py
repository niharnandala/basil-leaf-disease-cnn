"""
Plots the real training curves for the basil disease CNN.

The numbers in training_history.json are the actual per-epoch values Keras
printed during the run. They were recovered from the saved notebook output
after the in-memory `history` object was lost when the kernel was
interrupted. Nothing here is estimated or simulated.

    python plot_training_history.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
OUT = HERE / "static"
OUT.mkdir(parents=True, exist_ok=True)

history = json.loads((HERE / "training_history.json").read_text())

epochs = [h["epoch"] for h in history]
acc = [h["accuracy"] for h in history]
val_acc = [h["val_accuracy"] for h in history]
loss = [h["loss"] for h in history]
val_loss = [h["val_loss"] for h in history]
lr = [h["learning_rate"] for h in history]

# the epoch where ReduceLROnPlateau first cut the learning rate
drop_epoch = next(
    (e for e, prev, cur in zip(epochs[1:], lr, lr[1:]) if cur < prev), None
)

fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(13, 5))

ax_acc.plot(epochs, acc, marker="o", label="train accuracy")
ax_acc.plot(epochs, val_acc, marker="s", label="validation accuracy")
ax_acc.set_title("Accuracy per epoch")
ax_acc.set_xlabel("epoch")
ax_acc.set_ylabel("accuracy")
ax_acc.set_ylim(0.5, 1.02)
ax_acc.legend()
ax_acc.grid(alpha=0.3)

ax_loss.plot(epochs, loss, marker="o", label="train loss")
ax_loss.plot(epochs, val_loss, marker="s", label="validation loss")
ax_loss.set_title("Loss per epoch")
ax_loss.set_xlabel("epoch")
ax_loss.set_ylabel("loss")
ax_loss.set_yscale("log")
ax_loss.legend()
ax_loss.grid(alpha=0.3)

if drop_epoch is not None:
    for ax in (ax_acc, ax_loss):
        ax.axvline(drop_epoch, color="grey", linestyle="--", linewidth=1)
        ax.annotate(
            f"ReduceLROnPlateau\n1e-3 to 2e-4",
            xy=(drop_epoch, ax.get_ylim()[0]),
            xytext=(drop_epoch + 0.15, ax.get_ylim()[0]),
            fontsize=8,
            color="grey",
            va="bottom",
        )

fig.suptitle(
    "Basil disease CNN, 28,000 images, 4 classes "
    "(validation accuracy dips to 74.1% at epoch 4, recovers after the LR drop)"
)
fig.tight_layout()
fig.savefig(OUT / "training_curves.png", dpi=150)
print(f"wrote {OUT / 'training_curves.png'}")

best = min(history, key=lambda h: h["val_loss"])
print(
    f"best epoch by val_loss: {best['epoch']} "
    f"(val_loss {best['val_loss']}, val_accuracy {best['val_accuracy']})"
)
