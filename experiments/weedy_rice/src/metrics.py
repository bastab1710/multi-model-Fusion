import torch
import torch.nn as nn


class DiceLoss(nn.Module):

    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):

        probabilities = torch.sigmoid(logits)

        probabilities = probabilities.view(-1)
        targets = targets.view(-1)

        intersection = (probabilities * targets).sum()

        dice = (
            2.0 * intersection + self.smooth
        ) / (
            probabilities.sum() + targets.sum() + self.smooth
        )

        return 1.0 - dice


class BCEDiceLoss(nn.Module):

    def __init__(self):
        super().__init__()

        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, logits, targets):

        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)

        return bce_loss + dice_loss


def calculate_metrics(logits, targets, threshold=0.5):

    probabilities = torch.sigmoid(logits)

    predictions = (probabilities >= threshold).float()
    targets = (targets >= 0.5).float()

    predictions = predictions.view(-1)
    targets = targets.view(-1)

    tp = (predictions * targets).sum().item()

    fp = (
        predictions * (1 - targets)
    ).sum().item()

    fn = (
        (1 - predictions) * targets
    ).sum().item()

    tn = (
        (1 - predictions) * (1 - targets)
    ).sum().item()

    epsilon = 1e-7

    iou = tp / (tp + fp + fn + epsilon)

    dice = (
        2 * tp
    ) / (
        2 * tp + fp + fn + epsilon
    )

    precision = tp / (tp + fp + epsilon)

    recall = tp / (tp + fn + epsilon)

    f1 = (
        2 * precision * recall
    ) / (
        precision + recall + epsilon
    )

    accuracy = (
        tp + tn
    ) / (
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


if __name__ == "__main__":

    print("Testing loss functions and metrics...")

    targets = torch.zeros(2, 1, 192, 256)
    targets[:, :, 50:120, 60:150] = 1.0

    logits = torch.full_like(targets, -5.0)
    logits[:, :, 50:120, 60:150] = 5.0

    loss_function = BCEDiceLoss()

    loss = loss_function(logits, targets)

    metrics = calculate_metrics(logits, targets)

    print("Combined BCE + Dice Loss:", loss.item())

    for name, value in metrics.items():
        print(f"{name}: {value:.6f}")

    assert torch.isfinite(loss)
    assert metrics["IoU"] > 0.99
    assert metrics["Dice"] > 0.99

    print("\nALL METRIC TESTS PASSED")
