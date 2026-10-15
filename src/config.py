"""Loads config.yaml into typed objects so nothing is hardcoded in the modules."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config.yaml"


@dataclass(frozen=True)
class Config:
    raw: dict

    # --- data ---
    @property
    def data_dir(self) -> Path:
        # command line wins, then the environment, then config.yaml
        env = os.environ.get("BASIL_DATA_DIR")
        return Path(env) if env else (ROOT / self.raw["data"]["dir"])

    @property
    def image_size(self) -> int:
        return int(self.raw["data"]["image_size"])

    @property
    def batch_size(self) -> int:
        return int(self.raw["data"]["batch_size"])

    @property
    def validation_split(self) -> float:
        return float(self.raw["data"]["validation_split"])

    @property
    def seed(self) -> int:
        return int(self.raw["data"]["seed"])

    @property
    def classes(self) -> list[str]:
        return list(self.raw["classes"])

    # --- model and training ---
    @property
    def augmentation(self) -> dict:
        return dict(self.raw["augmentation"])

    @property
    def model_params(self) -> dict:
        return dict(self.raw["model"])

    @property
    def training_params(self) -> dict:
        return dict(self.raw["training"])

    # --- paths ---
    def path(self, key: str) -> Path:
        return ROOT / self.raw["paths"][key]


def load(path: Path | str | None = None) -> Config:
    with open(path or DEFAULT_CONFIG, encoding="utf-8") as fh:
        return Config(yaml.safe_load(fh))
