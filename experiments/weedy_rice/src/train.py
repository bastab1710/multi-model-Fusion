import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import WeedyRiceDataset
from models import UNet, LateFusionUNet
from metrics import BCEDiceLoss, calculate_metrics


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def create_model(mode):
    if mode == "rgb":
        return UNet(in_channels=3)

    if mode == "multispectral":
        return UNet(in_channels=4)

    if mode == "early_fusion":
        return UNet(in_channels=7)

    if mode == "late_fusion":
        return LateFusionUNet()

    raise ValueError(f"Unknown mode: {mode}")


def get_inputs(batch, mode, device):
    if mode == "late_fusion":
        rgb = batch["rgb"].to(device)
        ms = batch["multispectral"].to(device)
        return rgb, ms

    return batch["image"].to(device)


def forward_model(model, inputs, mode):
    if mode == "late_fusion":
        return model(inputs[0], inputs[1])

    return model(inputs)


def run_epoch(model, loader, criterion, device, mode, optimizer=None):
    training = optimizer is not None

    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_samples = 0

    metric_totals = {
        "IoU": 0.0,
        "Dice": 0.0,
        "Precision": 0.0,
        "Recall": 0.0,
        "F1": 0.0,
        "Pixel Accuracy": 0.0,
    }

    for batch in loader:

        inputs = get_inputs(batch, mode, device)
        targets = batch["mask"].to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):

            logits = forward_model(model, inputs, mode)

            loss = criterion(logits, targets)

            if training:
                loss.backward()
                optimizer.step()

        batch_size = targets.size(0)

        total_loss += loss.item() * batch_size
        total_samples += batch_size

        metrics = calculate_metrics(logits.detach(), targets)

        for name in metric_totals:
            metric_totals[name] += metrics[name] * batch_size

    average_loss = total_loss / total_samples

    average_metrics = {
        name: value / total_samples
        for name, value in metric_totals.items()
    }

    return average_loss, average_metrics


def main(args):

    set_seed(args.seed)

    device = torch.device("cpu")

    print("\nDevice:", device)
    print("Experiment:", args.mode)

    train_dataset = WeedyRiceDataset(
        split="train",
        mode=args.mode,
    )

    val_dataset = WeedyRiceDataset(
        split="val",
        mode=args.mode,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = create_model(args.mode).to(device)

    criterion = BCEDiceLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
    )

    checkpoint_dir = (
        Path(__file__).resolve().parents[1]
        / "models"
        / args.mode
    )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / "best_model.pth"

    history = []

    best_dice = -1.0

    print("\nStarting training...")
    print("Training samples:", len(train_dataset))
    print("Validation samples:", len(val_dataset))
    print("Batch size:", args.batch_size)
    print("Epochs:", args.epochs)

    for epoch in range(1, args.epochs + 1):

        train_loss, train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            args.mode,
            optimizer,
        )

        val_loss, val_metrics = run_epoch(
            model,
            val_loader,
            criterion,
            device,
            args.mode,
        )

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        }

        history.append(epoch_record)

        print(
            f"\nEpoch [{epoch}/{args.epochs}]"
            f"\nTrain Loss: {train_loss:.4f}"
            f" | Val Loss: {val_loss:.4f}"
            f"\nTrain Dice: {train_metrics['Dice']:.4f}"
            f" | Val Dice: {val_metrics['Dice']:.4f}"
            f"\nVal IoU: {val_metrics['IoU']:.4f}"
            f" | Val F1: {val_metrics['F1']:.4f}"
        )

        if val_metrics["Dice"] > best_dice:

            best_dice = val_metrics["Dice"]

            torch.save(
                {
                    "epoch": epoch,
                    "mode": args.mode,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_dice": best_dice,
                    "history": history,
                },
                checkpoint_path,
            )

            print("Best checkpoint saved:", checkpoint_path)

    results_dir = (
        Path(__file__).resolve().parents[1] / "results"
    )

    results_dir.mkdir(parents=True, exist_ok=True)

    history_path = results_dir / f"{args.mode}_history.pt"

    torch.save(history, history_path)

    print("\nTraining completed.")
    print("Best validation Dice:", best_dice)
    print("Checkpoint:", checkpoint_path)
    print("Training history:", history_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=[
            "rgb",
            "multispectral",
            "early_fusion",
            "late_fusion",
        ],
        required=True,
    )

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)

    main(parser.parse_args())
