"""The CNN. Built from scratch rather than fine-tuned from a pretrained backbone."""

from __future__ import annotations

from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    MaxPooling2D,
)
from tensorflow.keras.optimizers import AdamW

from .config import Config


def build_model(cfg: Config) -> Sequential:
    params = cfg.model_params
    size = cfg.image_size

    layers = []
    for index, filters in enumerate(params["conv_filters"]):
        kwargs = {"input_shape": (size, size, 3)} if index == 0 else {}
        layers += [
            Conv2D(filters, (3, 3), activation="relu", padding="same", **kwargs),
            BatchNormalization(),
            MaxPooling2D(2, 2),
        ]

    layers.append(Flatten())
    for units, rate in zip(params["dense_units"], params["dropout"], strict=True):
        layers += [Dense(units, activation="relu"), Dropout(rate)]
    layers.append(Dense(len(cfg.classes), activation="softmax"))

    model = Sequential(layers)
    model.compile(
        optimizer=AdamW(learning_rate=params["learning_rate"]),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
