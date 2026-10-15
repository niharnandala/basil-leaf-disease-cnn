"""
Flask app that serves the basil leaf disease CNN.

Paths come from the environment so this runs anywhere, not just on the
machine it was trained on:

    MODEL_PATH    defaults to models/basil_disease_best_model.keras
    UPLOAD_DIR    defaults to static/uploads
"""

import os
import uuid
from pathlib import Path

import numpy as np
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

import tensorflow as tf
from tensorflow.keras.preprocessing import image

HERE = Path(__file__).parent
MODEL_PATH = Path(os.environ.get("MODEL_PATH", HERE / "models" / "basil_disease_best_model.keras"))
UPLOAD_DIR = Path(os.environ.get("UPLOAD_DIR", HERE / "static" / "uploads"))
ALLOWED = {".jpg", ".jpeg", ".png", ".webp"}
IMG_SIZE = 224

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8 MB cap on uploads

if not MODEL_PATH.exists():
    raise SystemExit(
        f"model not found at {MODEL_PATH}\n"
        "The trained .keras file is not in the repo because of its size. "
        "Train it with train_model.ipynb, or set MODEL_PATH to point at your copy."
    )

model = tf.keras.models.load_model(MODEL_PATH)

# order matches flow_from_directory, which sorts class folders alphabetically
CLASS_NAMES = ["bacterial", "fungal", "healthy", "pests"]

REMEDIES = {
    "bacterial": "Remove infected leaves and apply a copper-based treatment.",
    "fungal": "Apply neem oil or a baking soda spray, and improve airflow around the plant.",
    "pests": "Use a natural insecticide such as neem oil or garlic spray.",
    "healthy": "No action needed.",
}


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    uploaded = request.files.get("file")
    if uploaded is None or uploaded.filename == "":
        return render_template("index.html", error="Choose an image first."), 400

    name = secure_filename(uploaded.filename)
    if Path(name).suffix.lower() not in ALLOWED:
        return render_template("index.html", error="Upload a JPG, PNG or WEBP image."), 400

    # prefix with a uuid so two people uploading leaf.jpg don't overwrite each other
    stored = UPLOAD_DIR / f"{uuid.uuid4().hex}_{name}"
    uploaded.save(stored)

    img = image.load_img(stored, target_size=(IMG_SIZE, IMG_SIZE))
    batch = np.expand_dims(image.img_to_array(img) / 255.0, axis=0)

    scores = model.predict(batch, verbose=0)[0]
    index = int(np.argmax(scores))
    label = CLASS_NAMES[index]

    return render_template(
        "result.html",
        image_path=f"uploads/{stored.name}",
        disease=label.title(),
        confidence=f"{scores[index] * 100:.1f}%",
        remedy=REMEDIES[label],
    )


if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1")
