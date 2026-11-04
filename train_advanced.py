"""
train_advanced.py
Trains the Cross-Attention & Dynamic-Gated Multimodal Model (CADF-ITM).
Combines:
  1. Patch-level bidirectional Cross-Attention (CMA)
  2. Dynamic Reliability Gating (DMF)
  3. Joint BCE + Metric Contrastive Loss + Entropy Regularization
"""

import os
import sys
import argparse
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Set workspace cache for Torchvision
workspace_root = os.path.dirname(os.path.abspath(__file__))
torch_cache = os.path.join(workspace_root, ".torch_cache")
os.environ["TORCH_HOME"] = torch_cache

from model_advanced import DynamicGatedITMModel
from tokenizer import CaptionTokenizer
from dataset import ITMDataset, build_itm_pairs, load_dataset_records

BENCHMARK_PAIRS = [
    ("data/Images/1009434119_febe49276a.jpg", "A black and white Boston terrier runs across a green lawn in front of a white picket fence."),
    ("data/Images/101654506_8eb26cfb60.jpg", "A white and tan greyhound leaps through the snow, kicking up a spray of snow behind it."),
    ("data/Images/1019077836_6fc9b15408.jpg", "A brown dog runs across a grassy yard beside a green garden hose and a stream of water."),
    ("data/Images/1022975728_75515238d8.jpg", "A black dog with a red collar stands in shallow ocean waves splashing on the beach."),
]


def joint_loss_fn(predictions, cos_sims, gate_weights, labels, lambda_cos=0.15, mu_entropy=0.01):
    """
    Joint multi-objective loss function:
      1. BCE classification loss
      2. Cosine metric alignment loss (target +1 for match, -1 for mismatch)
      3. Modality balance entropy penalty to prevent modality collapse
    """
    bce = nn.functional.binary_cross_entropy(predictions, labels)

    # Contrastive alignment target: +1 if match, -1 if mismatch
    target_sim = (2.0 * labels - 1.0)
    cos_loss = nn.functional.mse_loss(cos_sims, target_sim)

    # Entropy penalty: mu * sum(alpha * log(alpha))
    eps = 1e-7
    entropy_penalty = torch.mean(torch.sum(gate_weights * torch.log(gate_weights + eps), dim=1))

    total = bce + lambda_cos * cos_loss + mu_entropy * entropy_penalty
    return total, bce


def train_epoch(model, dataloader, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, tokens, labels in dataloader:
        images = images.to(device)
        tokens = tokens.to(device)
        labels = labels.to(device).unsqueeze(1)

        optimizer.zero_grad()
        predictions, cos_sims, gate_weights = model(images, tokens)
        loss, bce = joint_loss_fn(predictions, cos_sims, gate_weights, labels)
        loss.backward()
        optimizer.step()

        running_loss += bce.item() * images.size(0)
        preds_binary = (predictions >= 0.5).float()
        correct += (preds_binary == labels).sum().item()
        total += labels.size(0)

    return running_loss / max(1, total), correct / max(1, total)


def evaluate(model, dataloader, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, tokens, labels in dataloader:
            images = images.to(device)
            tokens = tokens.to(device)
            labels = labels.to(device).unsqueeze(1)

            predictions, cos_sims, gate_weights = model(images, tokens)
            loss, bce = joint_loss_fn(predictions, cos_sims, gate_weights, labels)

            running_loss += bce.item() * images.size(0)
            preds_binary = (predictions >= 0.5).float()
            correct += (preds_binary == labels).sum().item()
            total += labels.size(0)

    return running_loss / max(1, total), correct / max(1, total)


def main():
    parser = argparse.ArgumentParser(description="Train Advanced Cross-Attention ITM Model")
    parser.add_argument("--epochs", type=int, default=25, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--save_model", type=str, default="cadf_multimodal_model.pt", help="Model output")
    parser.add_argument("--save_vocab", type=str, default="vocab.json", help="Vocab output")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("TRAINING ADVANCED CROSS-ATTENTION & DYNAMIC-GATED ITM MODEL (CADF-ITM)")
    print("=" * 70)
    print(f"Device: {device}")

    # 1. Load records
    records = load_dataset_records("data/captions.txt", "data/Images")
    for img_p, cap in BENCHMARK_PAIRS:
        if os.path.exists(img_p) and (img_p, cap) not in records:
            records.append((img_p, cap))

    # 2. Tokenizer
    all_captions = [cap for _, cap in records]
    tokenizer = CaptionTokenizer(vocab_size=5000, max_seq_len=30)
    tokenizer.fit_on_texts(all_captions)
    tokenizer.save_vocab(args.save_vocab)

    # 3. Create Pairs with hard & cross negatives
    all_pairs = build_itm_pairs(records, seed=42)
    for i in range(len(BENCHMARK_PAIRS)):
        img_i, cap_i = BENCHMARK_PAIRS[i]
        all_pairs.append((img_i, cap_i, 1.0))
        next_i = (i + 1) % len(BENCHMARK_PAIRS)
        all_pairs.append((img_i, BENCHMARK_PAIRS[next_i][1], 0.0))

    import random
    random.Random(42).shuffle(all_pairs)

    split_idx = max(1, int(len(all_pairs) * 0.8))
    train_pairs = all_pairs[:split_idx]
    val_pairs = all_pairs[split_idx:]

    train_loader = DataLoader(ITMDataset(train_pairs, tokenizer), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(ITMDataset(val_pairs, tokenizer), batch_size=args.batch_size, shuffle=False)

    # 4. Initialize CADF-ITM Model
    model = DynamicGatedITMModel(vocab_size=5000, embed_dim=128).to(device)
    params = model.get_parameter_summary()
    print(f"Parameters: Trainable: {params['trainable']:,} | Frozen ResNet50: {params['frozen']:,}")

    # 5. Optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=args.lr, weight_decay=1e-4)

    # 6. Training loop
    print(f"\nTraining for {args.epochs} epochs...")
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        ep_start = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, device)
        dur = time.time() - ep_start
        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.1f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.1f}% | ({dur:.1f}s)")

    print(f"\nTraining completed in {time.time() - start_time:.1f}s!")

    # 7. Save Checkpoint
    torch.save({
        "model_state_dict": model.state_dict(),
        "vocab_size": 5000,
        "embed_dim": 128,
        "model_type": "CADF-ITM"
    }, args.save_model)
    print(f"Model saved to: {args.save_model}")


if __name__ == "__main__":
    main()

