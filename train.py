
"""
train.py
Leakage-free training pipeline for Flickr8k Image-Text Matching.

Architecture:
- Frozen pretrained ResNet50 image encoder
- Trainable text embedding branch
- Late-fusion judge network

Dataset protocol:
- 80% training images
- 10% validation images
- 10% test images
- Split by image filename before generating pairs
"""

import os
import json
import time
import random
import argparse
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import LateFusionMultimodalModel
from tokenizer import CaptionTokenizer
from dataset import ITMDataset, load_dataset_records


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_by_image(records, seed=42):
    """Split unique image filenames before generating ITM pairs."""

    grouped = defaultdict(list)

    for image_path, caption in records:
        grouped[os.path.abspath(image_path)].append(
            (image_path, caption)
        )

    image_paths = list(grouped.keys())
    rng = random.Random(seed)
    rng.shuffle(image_paths)

    total = len(image_paths)

    train_end = int(total * 0.80)
    val_end = train_end + int(total * 0.10)

    train_images = image_paths[:train_end]
    val_images = image_paths[train_end:val_end]
    test_images = image_paths[val_end:]

    train_records = [
        item for path in train_images for item in grouped[path]
    ]

    val_records = [
        item for path in val_images for item in grouped[path]
    ]

    test_records = [
        item for path in test_images for item in grouped[path]
    ]

    # Verify that no image occurs in multiple splits.
    train_set = set(train_images)
    val_set = set(val_images)
    test_set = set(test_images)

    assert train_set.isdisjoint(val_set)
    assert train_set.isdisjoint(test_set)
    assert val_set.isdisjoint(test_set)

    print("\n[IMAGE-LEVEL DATA SPLIT]")
    print(f"Training images:   {len(train_images)}")
    print(f"Validation images: {len(val_images)}")
    print(f"Testing images:    {len(test_images)}")

    return train_records, val_records, test_records, {
        "train": [os.path.basename(p) for p in train_images],
        "validation": [os.path.basename(p) for p in val_images],
        "test": [os.path.basename(p) for p in test_images],
    }


def build_balanced_pairs(records, seed=42):
    """
    Generate balanced positive and negative pairs.

    Positive: image and its actual caption.
    Negative: image and a caption belonging to another image.
    """

    rng = random.Random(seed)

    captions_by_image = defaultdict(list)

    for image_path, caption in records:
        captions_by_image[os.path.abspath(image_path)].append(caption)

    image_paths = list(captions_by_image.keys())

    if len(image_paths) < 2:
        raise ValueError("At least two unique images are required.")

    all_captions = [
        (image_path, caption)
        for image_path, captions in captions_by_image.items()
        for caption in captions
    ]

    pairs = []

    for image_path, captions in captions_by_image.items():
        for caption in captions:
            # Positive pair
            pairs.append((image_path, caption, 1.0))

            # Select a caption belonging to a different image.
            negative_image, negative_caption = rng.choice(all_captions)

            while negative_image == image_path:
                negative_image, negative_caption = rng.choice(all_captions)

            pairs.append(
                (image_path, negative_caption, 0.0)
            )

    rng.shuffle(pairs)

    positives = sum(1 for _, _, label in pairs if label == 1.0)
    negatives = sum(1 for _, _, label in pairs if label == 0.0)

    print(
        f"Generated {len(pairs)} pairs "
        f"(Positive: {positives}, Negative: {negatives})"
    )

    return pairs


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, tokens, labels in loader:
        images = images.to(device)
        tokens = tokens.to(device)
        labels = labels.to(device).unsqueeze(1)

        optimizer.zero_grad()

        predictions = model(images, tokens)
        loss = criterion(predictions, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

        predicted_labels = (predictions >= 0.5).float()

        correct += (predicted_labels == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    total = 0

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    with torch.no_grad():
        for images, tokens, labels in loader:
            images = images.to(device)
            tokens = tokens.to(device)
            labels = labels.to(device).unsqueeze(1)

            predictions = model(images, tokens)

            loss = criterion(predictions, labels)

            total_loss += loss.item() * images.size(0)
            total += labels.size(0)

            predicted_labels = (predictions >= 0.5).float()

            tp += ((predicted_labels == 1) & (labels == 1)).sum().item()
            tn += ((predicted_labels == 0) & (labels == 0)).sum().item()
            fp += ((predicted_labels == 1) & (labels == 0)).sum().item()
            fn += ((predicted_labels == 0) & (labels == 1)).sum().item()

    accuracy = (tp + tn) / max(1, total)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0 else 0.0
    )

    return {
        "loss": total_loss / max(1, total),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train Flickr8k Late-Fusion ITM Model"
    )

    parser.add_argument("--captions_file", default="data/captions.txt")
    parser.add_argument("--images_dir", default="data/Images")

    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument("--vocab_size", type=int, default=5000)
    parser.add_argument("--max_seq_len", type=int, default=30)
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--projection_dim", type=int, default=128)

    parser.add_argument(
        "--save_model",
        default="multimodal_late_fusion_model.pt"
    )

    parser.add_argument("--save_vocab", default="vocab.json")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 65)
    print("FLICKR8K LATE-FUSION IMAGE-TEXT MATCHING")
    print("=" * 65)

    print("Device:", device)

    # 1. Load original image-caption records.
    records = load_dataset_records(
        args.captions_file,
        args.images_dir
    )

    print(f"\nLoaded image-caption records: {len(records)}")

    if not records:
        raise RuntimeError("No valid image-caption records found.")

    # 2. Split unique images BEFORE generating positive/negative pairs.
    train_records, val_records, test_records, split_info = (
        split_by_image(records, args.seed)
    )

    # Save split information for reproducibility.
    os.makedirs("data/splits", exist_ok=True)

    with open(
        "data/splits/flickr8k_split.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(split_info, f, indent=2)

    print("\n[Split information saved]")
    print("data/splits/flickr8k_split.json")

    # 3. Fit tokenizer using TRAINING captions only.
    print("\n[Tokenizer] Building training vocabulary...")

    train_captions = [
        caption for _, caption in train_records
    ]

    tokenizer = CaptionTokenizer(
        vocab_size=args.vocab_size,
        max_seq_len=args.max_seq_len
    )

    tokenizer.fit_on_texts(train_captions)

    os.makedirs(
        os.path.dirname(os.path.abspath(args.save_vocab)),
        exist_ok=True
    )

    tokenizer.save_vocab(args.save_vocab)

    # 4. Generate pairs independently within each split.
    print("\n[Training pairs]")
    train_pairs = build_balanced_pairs(
        train_records, args.seed
    )

    print("\n[Validation pairs]")
    val_pairs = build_balanced_pairs(
        val_records, args.seed + 1
    )

    print("\n[Test pairs]")
    test_pairs = build_balanced_pairs(
        test_records, args.seed + 2
    )

    # 5. Create datasets and data loaders.
    train_dataset = ITMDataset(train_pairs, tokenizer)
    val_dataset = ITMDataset(val_pairs, tokenizer)
    test_dataset = ITMDataset(test_pairs, tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    # 6. Initialize the existing model architecture.
    print("\n[Model] Initializing Late-Fusion Model...")

    model = LateFusionMultimodalModel(
        vocab_size=args.vocab_size,
        embedding_dim=args.embedding_dim,
        projection_dim=args.projection_dim
    ).to(device)

    summary = model.get_parameter_summary()

    print("Frozen parameters:", f"{summary['frozen']:,}")
    print("Trainable parameters:", f"{summary['trainable']:,}")
    print("Total parameters:", f"{summary['total']:,}")

    criterion = nn.BCELoss()

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    optimizer = torch.optim.Adam(
        trainable_parameters,
        lr=args.lr,
        weight_decay=1e-4
    )

    # 7. Train and select the best checkpoint using validation loss.
    best_val_loss = float("inf")
    start_time = time.time()

    print("\n[Training] Starting training...")

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        train_loss, train_acc = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            device
        )

        elapsed = time.time() - epoch_start

        print(
            f"Epoch [{epoch}/{args.epochs}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc * 100:.2f}% | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Acc: {val_metrics['accuracy'] * 100:.2f}% | "
            f"Time: {elapsed:.1f}s"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]

            os.makedirs(
                os.path.dirname(os.path.abspath(args.save_model)),
                exist_ok=True
            )

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "vocab_size": args.vocab_size,
                    "embedding_dim": args.embedding_dim,
                    "projection_dim": args.projection_dim,
                    "max_seq_len": args.max_seq_len,
                    "best_val_loss": best_val_loss,
                    "epoch": epoch,
                },
                args.save_model
            )

            print("  -> Best model checkpoint saved.")

    print(
        f"\nTraining completed in "
        f"{(time.time() - start_time) / 60:.2f} minutes."
    )

    # 8. Load best validation checkpoint.
    print("\n[Testing] Loading best checkpoint...")

    checkpoint = torch.load(
        args.save_model,
        map_location=device
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    # 9. Evaluate the independent test set exactly once.
    test_metrics = evaluate(
        model,
        test_loader,
        criterion,
        device
    )

    print("\n" + "=" * 65)
    print("FINAL INDEPENDENT TEST RESULTS")
    print("=" * 65)

    print(f"Test Loss:      {test_metrics['loss']:.4f}")
    print(f"Accuracy:       {test_metrics['accuracy'] * 100:.2f}%")
    print(f"Precision:      {test_metrics['precision'] * 100:.2f}%")
    print(f"Recall:         {test_metrics['recall'] * 100:.2f}%")
    print(f"F1 Score:       {test_metrics['f1'] * 100:.2f}%")

    print("\nConfusion Matrix:")
    print("[[TN, FP],")
    print(" [FN, TP]]")
    print(
        f"[[{test_metrics['tn']}, {test_metrics['fp']}],"
    )
    print(
        f" [{test_metrics['fn']}, {test_metrics['tp']}]]"
    )

    # 10. Save test metrics.
    os.makedirs("results", exist_ok=True)

    results = {
        "test_loss": test_metrics["loss"],
        "accuracy": test_metrics["accuracy"],
        "precision": test_metrics["precision"],
        "recall": test_metrics["recall"],
        "f1_score": test_metrics["f1"],
        "confusion_matrix": [
            [test_metrics["tn"], test_metrics["fp"]],
            [test_metrics["fn"], test_metrics["tp"]]
        ],
        "best_epoch": checkpoint["epoch"],
        "best_validation_loss": checkpoint["best_val_loss"]
    }

    with open(
        "results/late_fusion_test_results.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(results, f, indent=4)

    print("\nResults saved to:")
    print("results/late_fusion_test_results.json")

    print("\nExperiment 1 baseline training completed.")


if __name__ == "__main__":
    main()
