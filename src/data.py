"""Builds the training and validation generators.

A caveat that belongs next to this code rather than buried in the README:
`validation_split` splits each class folder by file order. If the dataset on
disk was expanded by augmentation before it got here, near-duplicate copies of
one source photograph can land on both sides of the split, and the validation
score stops meaning anything. The recorded run has that problem. See
`docs/evaluation.md`.
"""

from __future__ import annotations

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from .config import Config


def build_generators(cfg: Config):
    """Returns (train_generator, validation_generator)."""
    if not cfg.data_dir.exists():
        raise FileNotFoundError(
            f"dataset not found at {cfg.data_dir}\n"
            "Point at it with --data-dir or the BASIL_DATA_DIR environment variable. "
            "It needs one subfolder per class: " + ", ".join(cfg.classes)
        )

    train_augmenter = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=cfg.validation_split,
        **cfg.augmentation,
    )
    # validation is only rescaled; augmenting it would measure the augmenter
    validation_augmenter = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=cfg.validation_split,
    )

    common = dict(
        directory=cfg.data_dir,
        target_size=(cfg.image_size, cfg.image_size),
        batch_size=cfg.batch_size,
        class_mode="categorical",
        classes=cfg.classes,
        seed=cfg.seed,
    )

    train = train_augmenter.flow_from_directory(subset="training", **common)
    validation = validation_augmenter.flow_from_directory(
        subset="validation", shuffle=False, **common
    )
    return train, validation
