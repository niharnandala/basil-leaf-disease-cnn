"""Flask front end for the trained classifier.

    python -m app.main

Paths come from config.yaml, overridable with MODEL_PATH and UPLOAD_DIR.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import numpy as np
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

from src import config as config_module

cfg = config_module.load()

HERE = Path(__file__).resolve().parent
MODEL_PATH = Path(os.environ.get("MODEL_PATH", cfg.path("model")))
UPLOAD_DIR = Path(os.environ.get("UPLOAD_DIR", HERE / "static" / "uploads"))
ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

REMEDIES = {
    "bacterial": "Remove infected leaves and apply a copper-based treatment.",
    "fungal": "Apply neem oil or a baking soda spray and improve airflow around the plant.",
    "pests": "Use a natural insecticide such as neem oil or garlic spray.",
    "healthy": "No action needed.",
}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024

_model = None


def get_model():
    """Loaded once, on the first request, so importing this module stays cheap."""
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"no trained model at {MODEL_PATH}. "
                "Run `python -m src.train`, or set MODEL_PATH to an existing checkpoint."
            )
        import tensorflow as tf

        _model = tf.keras.models.load_model(MODEL_PATH)
    return _model


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/health")
def health():
    return {"status": "ok", "model_present": MODEL_PATH.exists()}


@app.route("/predict", methods=["POST"])
def predict():
    uploaded = request.files.get("file")
    if uploaded is None or uploaded.filename == "":
        return render_template("index.html", error="Choose an image first."), 400

    filename = secure_filename(uploaded.filename)
    if Path(filename).suffix.lower() not in ALLOWED_SUFFIXES:
        return render_template("index.html", error="Upload a JPG, PNG or WEBP image."), 400

    # uuid prefix so two people uploading leaf.jpg do not overwrite each other
    stored = UPLOAD_DIR / f"{uuid.uuid4().hex}_{filename}"
    uploaded.save(stored)

    from tensorflow.keras.preprocessing import image

    size = cfg.image_size
    picture = image.load_img(stored, target_size=(size, size))
    batch = np.expand_dims(image.img_to_array(picture) / 255.0, axis=0)

    scores = get_model().predict(batch, verbose=0)[0]
    index = int(np.argmax(scores))
    label = cfg.classes[index]

    return render_template(
        "result.html",
        image_path=f"uploads/{stored.name}",
        disease=label.title(),
        confidence=f"{scores[index] * 100:.1f}%",
        remedy=REMEDIES[label],
    )


if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1")
