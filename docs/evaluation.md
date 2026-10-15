# Evaluation, and why the headline number is not trustworthy

## What the run produced

| | |
|---|---|
| Validation accuracy | 99.87% |
| Validation loss | 0.0042 |
| Precision / recall / F1 | 1.0000 on all four classes |
| Validation images | 5,600 (1,400 per class) |

## Why that should not be taken at face value

Perfect precision and recall on four classes over 5,600 images is not a result,
it is a symptom. Real image classifiers confuse visually similar classes, and
bacterial spotting and fungal spotting on a basil leaf are visually similar.

The most likely cause is that the validation set is not independent of the
training set:

**The dataset appears to have been augmented before the split, not after.**
Four classes of exactly 7,000 images each is too neat to be a natural
collection. If the on-disk dataset already contains rotated and flipped copies
of each source photograph, then a copy of a leaf can sit in training while
another copy of the *same leaf* sits in validation.

**`ImageDataGenerator(validation_split=0.2)` splits by file order.** It takes
the last 20% of the filenames in each class folder. Augmented variants of one
original are usually written next to each other, so the split cuts straight
through groups of near-duplicates rather than between distinct plants.

Together those mean the model can score near 100% by recognising images it has
effectively already seen, without having learned anything that transfers to a
leaf photographed by someone else.

## What this result does and does not support

**Supports:** the architecture trains, the pipeline runs end to end, the
callbacks behave correctly, and the model separates the four classes on this
dataset.

**Does not support:** any claim about accuracy on new basil photographs taken
under different lighting, with a different camera, or at a different growth
stage.

## What fixing it properly would require

1. Group images by source photograph and split on **groups**, not files, so
   every augmented copy of one leaf lands on one side only.
2. Augment after the split, inside the training generator only. The code in
   `src/data.py` already applies augmentation only to the training generator;
   the problem is upstream, in how the dataset on disk was built.
3. Hold back a test set photographed separately, ideally by someone else, and
   report that number instead.

That work has not been done here. Until it is, 99.87% is a number about this
dataset, not about basil leaves.

## Training behaviour, which is the more interesting result

The per-epoch values are in [`artifacts/training_history.json`](../artifacts/training_history.json),
recovered from the training log after the notebook kernel was interrupted and
the in-memory `history` object was lost. `src/train.py` now writes history to
disk after every epoch so that cannot happen again.

Two things in that curve are worth more than the accuracy number:

**Epoch 4 broke.** Validation accuracy fell from 97.1% to 74.1% and validation
loss rose from 0.098 to 0.896, while training accuracy kept climbing from 0.912
to 0.960. The model had moved to a point that fit the training batches and not
the validation data.

**The learning rate rescued it.** `ReduceLROnPlateau` cut the rate from 1e-3 to
2e-4 going into epoch 7. Validation loss dropped from 0.231 to 0.027 in that
single epoch, then to 0.0013 by epoch 9, which is the best epoch and the
checkpoint the app loads.
