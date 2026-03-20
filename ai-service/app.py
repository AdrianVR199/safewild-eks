"""
ai-service — Image classification microservice using MobileNetV2.
Classifies images and enriches predictions with danger level data
from danger_db.json for wildlife safety identification.
"""
import io
import json
import base64
import logging
import os

import numpy as np
from flask import Flask, request, jsonify
from PIL import Image

import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input,
    decode_predictions,
)
from tensorflow.keras.preprocessing import image as keras_image

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("ai-service")

app = Flask(__name__)

# ── Load model ────────────────────────────────────────────────
log.info("Loading MobileNetV2 weights from ImageNet…")
model = MobileNetV2(weights="imagenet")
log.info("Model ready.")

# ── Load danger database ──────────────────────────────────────
DB_PATH = os.path.join(os.path.dirname(__file__), "danger_db.json")
with open(DB_PATH, "r", encoding="utf-8") as f:
    raw_db = json.load(f)

DANGER_DB = {
    k.lower().replace(" ", "_"): v
    for k, v in raw_db.items()
    if not k.startswith("comment")
}
log.info("Danger DB loaded with %d species.", len(DANGER_DB))


def lookup_danger(label: str):
    key = label.lower().replace(" ", "_").replace("-", "_")
    return DANGER_DB.get(key)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": "MobileNetV2", "species_in_db": len(DANGER_DB)})


@app.route("/classify", methods=["POST"])
def classify():
    data = request.get_json(force=True)
    if not data or "image" not in data:
        return jsonify({"error": "Field 'image' (base64) is required"}), 400

    try:
        img_bytes = base64.b64decode(data["image"])
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB").resize((224, 224))
    except Exception as exc:
        return jsonify({"error": f"Cannot decode image: {exc}"}), 400

    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    preds     = model.predict(img_array, verbose=0)
    results   = decode_predictions(preds, top=5)[0]

    top5 = [
        {"class_id": cid, "label": label, "confidence": round(float(score), 6)}
        for cid, label, score in results
    ]

    top_label   = top5[0]["label"]
    danger_info = lookup_danger(top_label)

    if danger_info:
        enriched = {
            "is_wildlife": True,
            "species":     top_label,
            "common_name": danger_info.get("common_name", top_label),
            "danger":      danger_info["danger"],
            "venomous":    danger_info.get("venomous", False),
            "aggressive":  danger_info.get("aggressive", False),
            "action":      danger_info.get("action", ""),
            "confidence":  round(top5[0]["confidence"] * 100, 1),
            "predictions": top5,
        }
    else:
        enriched = {
            "is_wildlife": False,
            "species":     top_label,
            "common_name": top_label.replace("_", " ").title(),
            "danger":      "NO_WILDLIFE",
            "venomous":    False,
            "aggressive":  False,
            "action":      "No se detectó fauna peligrosa en la imagen.",
            "confidence":  round(top5[0]["confidence"] * 100, 1),
            "predictions": top5,
        }

    log.info("Classified → %s | danger=%s | conf=%.1f%%",
             enriched["species"], enriched["danger"], enriched["confidence"])
    return jsonify(enriched)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)
