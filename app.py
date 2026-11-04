"""
app.py
Flask Web Application for the CADF-ITM Multimodal Image-Text Matching Software.
Serves the interactive demo UI and exposes /predict API endpoint.

Run:  python app.py
Open: http://localhost:5000
"""

import os
import io
import torch
import base64
from PIL import Image
from flask import Flask, request, jsonify, render_template, send_file

# Set workspace cache for Torchvision (no internet needed after first download)
workspace_root = os.path.dirname(os.path.abspath(__file__))
torch_cache = os.path.join(workspace_root, ".torch_cache")
os.environ["TORCH_HOME"] = torch_cache

from model import LateFusionMultimodalModel
from model_advanced import DynamicGatedITMModel
from tokenizer import CaptionTokenizer
from dataset import IMAGE_TRANSFORMS

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32MB max upload

# ─── Model Registry ─────────────────────────────────────────────────────────
DEVICE = torch.device("cpu")
MODEL_REGISTRY = {}

QUICK_TEST_IMAGES = {
    "fig1": "data/Images/1009434119_febe49276a.jpg",
    "fig2": "data/Images/101654506_8eb26cfb60.jpg",
    "fig3": "data/Images/1019077836_6fc9b15408.jpg",
    "fig4": "data/Images/1022975728_75515238d8.jpg",
}


def load_models():
    """Loads both models and vocabulary into global registry at startup."""
    global MODEL_REGISTRY

    vocab_path = "vocab.json"
    if not os.path.exists(vocab_path):
        print(f"[Warning] {vocab_path} not found. Run 'python train_advanced.py' first.")
        return

    tokenizer = CaptionTokenizer.load_vocab(vocab_path)
    MODEL_REGISTRY["tokenizer"] = tokenizer
    print(f"[App] Tokenizer loaded: {len(tokenizer.word2idx)} tokens")

    # Baseline model
    baseline_path = "multimodal_late_fusion_model.pt"
    if os.path.exists(baseline_path):
        try:
            ck = torch.load(baseline_path, map_location=DEVICE, weights_only=False)
            m = LateFusionMultimodalModel(
                vocab_size=ck.get("vocab_size", 5000),
                embedding_dim=ck.get("embedding_dim", 128),
                projection_dim=ck.get("projection_dim", 128)
            ).to(DEVICE)
            m.load_state_dict(ck["model_state_dict"])
            m.eval()
            MODEL_REGISTRY["baseline"] = m
            print("[App] Baseline Late-Fusion model loaded.")
        except Exception as e:
            print(f"[Warning] Could not load baseline model: {e}")

    # Advanced CADF-ITM model
    cadf_path = "cadf_multimodal_model.pt"
    if os.path.exists(cadf_path):
        try:
            ck = torch.load(cadf_path, map_location=DEVICE, weights_only=False)
            m = DynamicGatedITMModel(
                vocab_size=ck.get("vocab_size", 5000),
                embed_dim=ck.get("embed_dim", 128)
            ).to(DEVICE)
            m.load_state_dict(ck["model_state_dict"])
            m.eval()
            MODEL_REGISTRY["cadf"] = m
            print("[App] CADF-ITM advanced model loaded.")
        except Exception as e:
            print(f"[Warning] Could not load CADF-ITM model: {e}")


def preprocess_image(pil_img):
    """Resize and normalize image for ResNet50."""
    pil_img = pil_img.convert("RGB")
    return IMAGE_TRANSFORMS(pil_img).unsqueeze(0).to(DEVICE)


def run_inference(model_key, img_tensor, tokens):
    """Runs inference and returns score + optional gate weights."""
    model = MODEL_REGISTRY.get(model_key)
    if model is None:
        return None, None, None

    token_tensor = torch.tensor([tokens], dtype=torch.long).to(DEVICE)

    with torch.no_grad():
        if model_key == "cadf":
            score, cos_sim, gate_weights = model(img_tensor, token_tensor)
            gate_v = float(gate_weights[0][0].item())
            gate_t = float(gate_weights[0][1].item())
            return float(score.item()), gate_v, gate_t
        else:
            score = model(img_tensor, token_tensor)
            return float(score.item()), None, None


# ─── Routes ─────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the main UI."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    POST /predict
    Form data:
      - image: uploaded image file
      - caption: text string
      - model: 'cadf' or 'baseline'
    Returns JSON with score, decision, and optional gate weights.
    """
    if "tokenizer" not in MODEL_REGISTRY:
        return jsonify({"error": "Models not loaded. Run 'python train_advanced.py' first."}), 503

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded."}), 400

    img_file = request.files["image"]
    caption = request.form.get("caption", "").strip()
    model_key = request.form.get("model", "cadf")

    if not caption:
        return jsonify({"error": "Caption is required."}), 400

    if model_key not in MODEL_REGISTRY:
        fallback = "baseline" if "baseline" in MODEL_REGISTRY else ("cadf" if "cadf" in MODEL_REGISTRY else None)
        if fallback is None:
            return jsonify({"error": "No trained models found. Run train_advanced.py first."}), 503
        model_key = fallback

    # Read and preprocess image
    try:
        pil_img = Image.open(img_file.stream)
        img_tensor = preprocess_image(pil_img)
    except Exception as e:
        return jsonify({"error": f"Invalid image file: {str(e)}"}), 400

    # Tokenize caption
    tokenizer = MODEL_REGISTRY["tokenizer"]
    tokens = tokenizer.encode(caption)

    # Inference
    score, gate_v, gate_t = run_inference(model_key, img_tensor, tokens)
    if score is None:
        return jsonify({"error": f"Model '{model_key}' not available."}), 503

    is_match = score >= 0.5
    confidence = (score if is_match else (1.0 - score)) * 100

    response = {
        "score": round(score, 4),
        "match": is_match,
        "confidence": round(confidence, 1),
        "model": model_key,
        "verdict": "MATCH" if is_match else "MISMATCH",
    }

    if gate_v is not None:
        response["gate_vis"] = round(gate_v, 4)
        response["gate_txt"] = round(gate_t, 4)

    return jsonify(response)


@app.route("/quick_image/<img_key>")
def quick_image(img_key):
    """Serves one of the 4 benchmark images for quick-test buttons."""
    img_path = QUICK_TEST_IMAGES.get(img_key)
    if not img_path or not os.path.exists(img_path):
        return jsonify({"error": "Image not found"}), 404
    return send_file(img_path, mimetype="image/jpeg")


@app.route("/status")
def status():
    """Returns server status and loaded models."""
    loaded = [k for k in ["baseline", "cadf"] if k in MODEL_REGISTRY]
    return jsonify({
        "status": "running",
        "models_loaded": loaded,
        "tokenizer_ready": "tokenizer" in MODEL_REGISTRY,
        "vocab_size": len(MODEL_REGISTRY["tokenizer"].word2idx) if "tokenizer" in MODEL_REGISTRY else 0,
        "device": str(DEVICE),
        "quick_images_available": {k: os.path.exists(v) for k, v in QUICK_TEST_IMAGES.items()}
    })


if __name__ == "__main__":
    print("=" * 65)
    print("CADF-ITM MULTIMODAL FUSION — WEB APPLICATION")
    print("=" * 65)
    load_models()
    print("\n[App] Starting Flask server on http://localhost:5000 ...")
    print("[App] Press Ctrl+C to stop.\n")
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)

