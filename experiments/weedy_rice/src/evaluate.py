import csv
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Resolve project paths
SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent

sys.path.insert(0, str(SRC_DIR))

from dataset import WeedyRiceDataset
from metrics import BCEDiceLoss
from train import create_model, get_inputs, forward_model


MODES = [
    "rgb",
    "multispectral",
    "early_fusion",
    "late_fusion",
]

MODEL_NAMES = {
    "rgb": "RGB U-Net",
    "multispectral": "Multispectral U-Net",
    "early_fusion": "Early Fusion U-Net",
    "late_fusion": "Late Fusion U-Net",
}


def calculate_global_metrics(tp, fp, fn, tn):

    epsilon = 1e-7

    iou = tp / (tp + fp + fn + epsilon)

    dice = (2 * tp) / (
        2 * tp + fp + fn + epsilon
    )

    precision = tp / (tp + fp + epsilon)

    recall = tp / (tp + fn + epsilon)

    f1 = (2 * precision * recall) / (
        precision + recall + epsilon
    )

    accuracy = (tp + tn) / (
        tp + tn + fp + fn + epsilon
    )

    return {
        "IoU": iou,
        "Dice": dice,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Pixel Accuracy": accuracy,
    }


def evaluate_model(mode, device, batch_size):

    print("\n" + "=" * 60)
    print("Evaluating:", MODEL_NAMES[mode])
    print("=" * 60)

    test_dataset = WeedyRiceDataset(
        split="test",
        mode=mode,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = create_model(mode).to(device)

    checkpoint_path = (
        PROJECT_DIR
        / "models"
        / mode
        / "best_model.pth"
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint missing: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    if checkpoint["mode"] != mode:
        raise ValueError(
            f"Checkpoint mode mismatch: {checkpoint['mode']}"
        )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    criterion = BCEDiceLoss()

    total_loss = 0.0
    total_samples = 0

    # Global confusion counts for the entire test dataset
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0

    with torch.inference_mode():

        for batch_index, batch in enumerate(test_loader, 1):

            inputs = get_inputs(
                batch,
                mode,
                device,
            )

            targets = batch["mask"].to(device)

            logits = forward_model(
                model,
                inputs,
                mode,
            )

            loss = criterion(logits, targets)

            probabilities = torch.sigmoid(logits)

            predictions = (
                probabilities >= 0.5
            )

            actual = (
                targets >= 0.5
            )

            # Accumulate pixel counts across all batches
            total_tp += (
                predictions & actual
            ).sum().item()

            total_fp += (
                predictions & (~actual)
            ).sum().item()

            total_fn += (
                (~predictions) & actual
            ).sum().item()

            total_tn += (
                (~predictions) & (~actual)
            ).sum().item()

            batch_size_actual = targets.size(0)

            total_loss += (
                loss.item() * batch_size_actual
            )

            total_samples += batch_size_actual

            print(
                f"Batch {batch_index}/{len(test_loader)} "
                f"| Loss: {loss.item():.4f}"
            )

    # Average test loss
    average_loss = total_loss / total_samples

    # Calculate metrics once using global pixel counts
    global_metrics = calculate_global_metrics(
        total_tp,
        total_fp,
        total_fn,
        total_tn,
    )

    results = {
        "Model": MODEL_NAMES[mode],
        "Mode": mode,
        "Test Samples": total_samples,
        "Best Epoch": checkpoint["epoch"],
        "Best Validation Dice": checkpoint["best_val_dice"],
        "Test Loss": average_loss,
        "TP": total_tp,
        "FP": total_fp,
        "FN": total_fn,
        "TN": total_tn,
        **global_metrics,
    }

    print("\nGLOBAL TEST RESULTS:", MODEL_NAMES[mode])

    for name, value in results.items():

        if isinstance(value, float):
            print(f"{name}: {value:.6f}")
        else:
            print(f"{name}: {value}")

    return results


def main():

    device = torch.device("cpu")

    batch_size = 2

    print("\nWEEDY RICE MODEL EVALUATION")
    print("=" * 60)
    print("Device:", device)
    print("Batch size:", batch_size)
    print("Evaluation split: TEST")
    print("Metric aggregation: GLOBAL PIXEL-LEVEL")

    all_results = []

    for mode in MODES:

        result = evaluate_model(
            mode,
            device,
            batch_size,
        )

        all_results.append(result)

    # Save comparison results
    results_dir = PROJECT_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    output_path = results_dir / "test_metrics.csv"

    fieldnames = list(all_results[0].keys())

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(all_results)

    # Identify best model by global test Dice
    best_model = max(
        all_results,
        key=lambda item: item["Dice"],
    )

    print("\n" + "=" * 60)
    print("FINAL MODEL COMPARISON")
    print("=" * 60)

    for result in all_results:

        print(
            f"{result['Model']:<25} "
            f"Dice: {result['Dice']:.6f} "
            f"IoU: {result['IoU']:.6f} "
            f"Precision: {result['Precision']:.6f} "
            f"Recall: {result['Recall']:.6f}"
        )

    print("\nBest Test Dice:", best_model["Model"])
    print("Dice Score:", round(best_model["Dice"], 6))

    print("\nEvaluation completed successfully.")
    print("Results saved to:", output_path)


if __name__ == "__main__":
    main()

