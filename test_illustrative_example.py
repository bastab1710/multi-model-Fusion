"""
test_illustrative_example.py
Comparative Benchmark: Baseline Late Fusion vs. Novel CADF-ITM Model
Evaluates the exact 4 Flickr8k images across the 8 test pairs (4 positive + 4 negative):
  - Section 4 Illustrative Example from Assam Science and Technology University research report.
"""

import os
import sys
import torch
from PIL import Image

# Set workspace cache for Torchvision
workspace_root = os.path.dirname(os.path.abspath(__file__))
torch_cache = os.path.join(workspace_root, ".torch_cache")
os.environ["TORCH_HOME"] = torch_cache

from model import LateFusionMultimodalModel
from model_advanced import DynamicGatedITMModel
from tokenizer import CaptionTokenizer
from dataset import IMAGE_TRANSFORMS

FIGURES = {
    "Figure 1": {
        "file": "data/Images/1009434119_febe49276a.jpg",
        "caption": "A black and white Boston terrier runs across a green lawn in front of a white picket fence."
    },
    "Figure 2": {
        "file": "data/Images/101654506_8eb26cfb60.jpg",
        "caption": "A white and tan greyhound leaps through the snow, kicking up a spray of snow behind it."
    },
    "Figure 3": {
        "file": "data/Images/1019077836_6fc9b15408.jpg",
        "caption": "A brown dog runs across a grassy yard beside a green garden hose and a stream of water."
    },
    "Figure 4": {
        "file": "data/Images/1022975728_75515238d8.jpg",
        "caption": "A black dog with a red collar stands in shallow ocean waves splashing on the beach."
    }
}

BENCHMARK_TABLE = [
    ("Figure 1", "Figure 1", "Positive", 1.0, 0.94, "Ground Truth Match"),
    ("Figure 2", "Figure 2", "Positive", 1.0, 0.91, "Ground Truth Match"),
    ("Figure 3", "Figure 3", "Positive", 1.0, 0.88, "Ground Truth Match"),
    ("Figure 4", "Figure 4", "Positive", 1.0, 0.93, "Ground Truth Match"),
    ("Figure 1", "Figure 2", "Negative", 0.0, 0.15, "Lawn vs Snow"),
    ("Figure 2", "Figure 3", "Negative", 0.0, 0.34, "Hard Negative: Dog outdoors running (snow vs hose)"),
    ("Figure 3", "Figure 4", "Negative", 0.0, 0.12, "Grassy yard vs Ocean waves"),
    ("Figure 4", "Figure 1", "Negative", 0.0, 0.20, "Ocean waves vs Picket fence"),
]


def load_model(model_path, vocab_path="vocab.json", device="cpu"):
    tokenizer = CaptionTokenizer.load_vocab(vocab_path)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    if checkpoint.get("model_type") == "CADF-ITM":
        model = DynamicGatedITMModel(
            vocab_size=checkpoint.get("vocab_size", 5000),
            embed_dim=checkpoint.get("embed_dim", 128)
        )
        is_advanced = True
    else:
        model = LateFusionMultimodalModel(
            vocab_size=checkpoint.get("vocab_size", 5000),
            embedding_dim=checkpoint.get("embedding_dim", 128),
            projection_dim=checkpoint.get("projection_dim", 128)
        )
        is_advanced = False

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, tokenizer, is_advanced


def predict_pair(model, tokenizer, is_advanced, img_path, caption, device="cpu"):
    image = Image.open(img_path).convert("RGB")
    img_t = IMAGE_TRANSFORMS(image).unsqueeze(0).to(device)
    tokens = tokenizer.encode(caption)
    txt_t = torch.tensor([tokens], dtype=torch.long).to(device)

    with torch.no_grad():
        if is_advanced:
            score, _, gates = model(img_t, txt_t)
            return score.item(), gates[0].cpu().numpy()
        else:
            score = model(img_t, txt_t)
            return score.item(), None


def run_benchmark(model_path, model_name, device="cpu"):
    if not os.path.exists(model_path):
        print(f"[Skip] {model_name} ({model_path}) not found.")
        return

    print("\n" + "=" * 95)
    print(f"BENCHMARK RESULTS: {model_name}")
    print(f"File: {model_path}")
    print("=" * 95)

    model, tokenizer, is_advanced = load_model(model_path, "vocab.json", device)

    print(f"{'Image':<10} {'Caption':<12} {'Pair Type':<10} {'Label':<6} {'Expected':<10} {'Model Score':<13} {'Decision':<12} {'Status'}")
    print("-" * 95)

    correct = 0
    total = len(BENCHMARK_TABLE)

    for img_fig, cap_fig, pair_type, label, expected_score, note in BENCHMARK_TABLE:
        img_p = FIGURES[img_fig]["file"]
        cap_t = FIGURES[cap_fig]["caption"]

        score, gates = predict_pair(model, tokenizer, is_advanced, img_p, cap_t, device)
        decision = "Match (1)" if score >= 0.5 else "Mismatch(0)"
        is_correct = (score >= 0.5 and label == 1.0) or (score < 0.5 and label == 0.0)
        if is_correct:
            correct += 1

        status = "[PASS]" if is_correct else "[FAIL]"
        gate_str = f" [Vis: {gates[0]:.2f}, Txt: {gates[1]:.2f}]" if gates is not None else ""
        print(f"{img_fig:<10} {cap_fig:<12} {pair_type:<10} {int(label):<6} {expected_score:<10.2f} {score:<13.4f} {decision:<12} {status}{gate_str}")

    print("-" * 95)
    print(f"Total Accuracy: {correct}/{total} ({correct/total*100:.1f}%)")
    print("=" * 95)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Run Baseline Model
    run_benchmark("multimodal_late_fusion_model.pt", "Baseline Late-Fusion Model (ASTU Research Project)", device)

    # Run Novel CADF-ITM Model
    run_benchmark("cadf_multimodal_model.pt", "Novel Cross-Attention & Dynamic-Gated Model (CADF-ITM)", device)


if __name__ == "__main__":
    main()
