"""
predict.py
Standalone inference and evaluation script for the Multimodal Late-Fusion ITM Model (PyTorch).
Predicts whether a given caption accurately matches an image.
"""

import os
import sys
import argparse
import torch
from PIL import Image

# Set workspace cache for Torchvision
workspace_root = os.path.dirname(os.path.abspath(__file__))
torch_cache = os.path.join(workspace_root, ".torch_cache")
os.environ["TORCH_HOME"] = torch_cache

from model import LateFusionMultimodalModel
from tokenizer import CaptionTokenizer
from dataset import IMAGE_TRANSFORMS


def run_inference(image_path, caption, model_path="multimodal_late_fusion_model.pt", vocab_path="vocab.json", threshold=0.5, mock=False):
    """Performs inference using the trained PyTorch model or mock simulation."""
    print("=" * 65)
    print("MULTIMODAL IMAGE-TEXT MATCHING (ITM) INFERENCE")
    print("=" * 65)
    print(f"Image Path:  {image_path}")
    print(f"Caption:     \"{caption}\"")
    print(f"Model File:  {model_path}")
    print("-" * 65)

    if mock:
        print("[MOCK MODE] Simulating forward pass for architecture demonstration...")
        seed = abs(hash(image_path + caption)) % 1000
        prob = min(0.98, max(0.05, (seed / 1000.0)))
    else:
        if not os.path.exists(model_path):
            print(f"[Error] Model weights file '{model_path}' not found.")
            print("Please train the model first by running: python train.py")
            sys.exit(1)

        if not os.path.exists(vocab_path):
            print(f"[Error] Vocabulary file '{vocab_path}' not found.")
            print("Please run 'python train.py' to generate the vocabulary and weights.")
            sys.exit(1)

        if not os.path.exists(image_path):
            print(f"[Error] Input image '{image_path}' not found.")
            sys.exit(1)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = CaptionTokenizer.load_vocab(vocab_path)
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)

        model = LateFusionMultimodalModel(
            vocab_size=checkpoint.get("vocab_size", 5000),
            embedding_dim=checkpoint.get("embedding_dim", 128),
            projection_dim=checkpoint.get("projection_dim", 128)
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        # Load & transform image
        pil_img = Image.open(image_path).convert("RGB")
        img_tensor = IMAGE_TRANSFORMS(pil_img).unsqueeze(0).to(device)

        # Tokenize caption
        tokens = tokenizer.encode(caption)
        token_tensor = torch.tensor([tokens], dtype=torch.long).to(device)

        with torch.no_grad():
            prob = model(img_tensor, token_tensor).item()

    is_match = prob >= threshold
    verdict = "MATCH (1)" if is_match else "MISMATCH (0)"
    confidence = (prob if is_match else (1.0 - prob)) * 100

    print("-" * 65)
    print(f"Match Probability: {prob:.4f} ({prob * 100:.2f}%)")
    print(f"Decision:          [{'MATCH' if is_match else 'MISMATCH'}] -> {verdict}")
    print(f"Confidence:        {confidence:.1f}%")
    print("=" * 65)

    return prob, is_match


def run_batch_test(captions_file, model_path="multimodal_late_fusion_model.pt", vocab_path="vocab.json", n_samples=5, mock=False):
    """Runs prediction on random pairs sampled directly from the captions file."""
    if not os.path.exists(captions_file):
        print(f"[Error] Captions file '{captions_file}' not found.")
        return

    import random

    valid_lines = []
    with open(captions_file, "r", encoding="utf-8", errors="ignore") as f:
        header = f.readline()
        for line in f:
            line = line.strip()
            if line:
                valid_lines.append(line)

    selected = random.sample(valid_lines, min(n_samples, len(valid_lines)))
    print(f"\nRunning batch test on {len(selected)} samples...")

    for i, line in enumerate(selected, start=1):
        if "," in line:
            parts = line.split(",", 1)
        elif "\t" in line:
            parts = line.split("\t", 1)
        else:
            parts = line.split(maxsplit=1)

        img_name = parts[0].strip()
        caption = parts[1].strip() if len(parts) > 1 else ""
        img_path = os.path.join("data", "Images", img_name)
        print(f"\n[Test Sample {i}]")
        run_inference(img_path, caption, model_path=model_path, vocab_path=vocab_path, mock=mock)


def main():
    parser = argparse.ArgumentParser(description="Multimodal Late Fusion Image-Text Matching Prediction")
    parser.add_argument("--image", type=str, help="Path to input image file (JPEG / PNG)")
    parser.add_argument("--caption", type=str, help="Text caption to evaluate against the image")
    parser.add_argument("--model", type=str, default="multimodal_late_fusion_model.pt",
                        help="Path to trained PyTorch .pt model file")
    parser.add_argument("--vocab", type=str, default="vocab.json",
                        help="Path to saved vocab.json")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Decision threshold for matching (default: 0.5)")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Path to data directory containing captions.txt")
    parser.add_argument("--mock", action="store_true",
                        help="Run in mock/simulation mode without needing full model weights")
    parser.add_argument("--batch_test", type=int, default=0,
                        help="Run batch test on N random samples from captions.txt")

    args = parser.parse_args()

    if args.batch_test > 0:
        captions_path = os.path.join(args.data_dir, "captions.txt")
        run_batch_test(captions_path, args.model, vocab_path=args.vocab, n_samples=args.batch_test, mock=args.mock)
        return

    if not args.image or not args.caption:
        print("[Notice] Missing '--image' or '--caption' argument.")
        print("\nExample usage:")
        print('  python predict.py --image "data/Images/1009434119_febe49276a.jpg" --caption "A black and white dog running in grass"')
        print('  python predict.py --batch_test 3 --mock')
        sys.exit(0)

    run_inference(
        image_path=args.image,
        caption=args.caption,
        model_path=args.model,
        vocab_path=args.vocab,
        threshold=args.threshold,
        mock=args.mock
    )


if __name__ == "__main__":
    main()
