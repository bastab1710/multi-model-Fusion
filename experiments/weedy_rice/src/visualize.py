import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent

sys.path.insert(0, str(SRC_DIR))

from dataset import WeedyRiceDataset
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

# Five test samples to visualize
SAMPLE_INDICES = [0, 10, 25, 50, 75]


def prepare_rgb(tensor):
    """Convert normalized RGB tensor to displayable image."""
    image = tensor.permute(1, 2, 0).numpy()
    return np.clip(image, 0, 1)


def prepare_multispectral(tensor):
    """
    Create a false-colour composite:
    Red channel   = NIR
    Green channel = Red band
    Blue channel  = Green band

    Dataset band order: G, R, RE, NIR.
    """
    bands = tensor.numpy()

    display_channels = [
        bands[3],
        bands[1],
        bands[0],
    ]

    normalized = []

    for channel in display_channels:
        low, high = np.percentile(channel, [2, 98])

        if high <= low:
            normalized.append(np.zeros_like(channel))
        else:
            channel = (channel - low) / (high - low)
            normalized.append(np.clip(channel, 0, 1))

    return np.stack(normalized, axis=-1)


def load_models(device):

    models = {}

    for mode in MODES:

        print(f"Loading {MODEL_NAMES[mode]}...")

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

        models[mode] = model

    return models


def predict(model, sample, mode, device):

    batch = {
        key: value.unsqueeze(0)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in sample.items()
    }

    inputs = get_inputs(batch, mode, device)

    with torch.inference_mode():

        logits = forward_model(
            model,
            inputs,
            mode,
        )

        probabilities = torch.sigmoid(logits)

        prediction = (
            probabilities >= 0.5
        ).float()

    return prediction.squeeze().cpu().numpy()


def visualize_sample(
    index,
    datasets,
    models,
    device,
    output_dir,
):

    print(f"\nVisualizing test sample {index}...")

    samples = {
        mode: datasets[mode][index]
        for mode in MODES
    }

    filename = samples["rgb"]["filename"]

    # Confirm all modes refer to the same test image
    for mode in MODES:
        if samples[mode]["filename"] != filename:
            raise ValueError(
                f"Filename mismatch at index {index}: {mode}"
            )

    rgb_image = prepare_rgb(samples["rgb"]["image"])

    ms_image = prepare_multispectral(
        samples["multispectral"]["image"]
    )

    ground_truth = (
        samples["rgb"]["mask"]
        .squeeze()
        .numpy()
    )

    predictions = {}

    for mode in MODES:

        predictions[mode] = predict(
            models[mode],
            samples[mode],
            mode,
            device,
        )

    # Create seven-panel comparison
    fig, axes = plt.subplots(
        1,
        7,
        figsize=(24, 5),
    )

    fig.suptitle(
        f"Weedy Rice Segmentation Comparison: {filename}",
        fontsize=15,
        fontweight="bold",
    )

    axes[0].imshow(rgb_image)
    axes[0].set_title("Original RGB")

    axes[1].imshow(ms_image)
    axes[1].set_title("Multispectral\nFalse Colour")

    axes[2].imshow(
        ground_truth,
        cmap="gray",
        vmin=0,
        vmax=1,
    )
    axes[2].set_title("Ground Truth")

    for position, mode in enumerate(MODES, start=3):

        axes[position].imshow(
            predictions[mode],
            cmap="gray",
            vmin=0,
            vmax=1,
        )

        axes[position].set_title(
            MODEL_NAMES[mode]
        )

    for axis in axes:
        axis.axis("off")

    plt.tight_layout()

    safe_filename = Path(filename).stem

    output_path = (
        output_dir
        / f"comparison_{index + 1:03d}_{safe_filename}.png"
    )

    plt.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    print("Saved:", output_path)


def main():

    device = torch.device("cpu")

    print("\nWEEDY RICE VISUALIZATION")
    print("=" * 60)
    print("Device:", device)

    # Create all four test datasets
    datasets = {}

    for mode in MODES:

        datasets[mode] = WeedyRiceDataset(
            split="test",
            mode=mode,
        )

    # Check dataset consistency
    reference_filenames = datasets["rgb"].filenames

    for mode in MODES:

        if datasets[mode].filenames != reference_filenames:
            raise ValueError(
                f"Test dataset mismatch in {mode} mode"
            )

    print("\nTest samples:", len(reference_filenames))

    # Load trained models
    models = load_models(device)

    # Create output directory
    output_dir = PROJECT_DIR / "results" / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Validate selected sample indices
    for index in SAMPLE_INDICES:

        if index >= len(reference_filenames):
            raise IndexError(
                f"Sample index {index} is outside the test dataset"
            )

    # Generate comparison images
    for index in SAMPLE_INDICES:

        visualize_sample(
            index,
            datasets,
            models,
            device,
            output_dir,
        )

    print("\n" + "=" * 60)
    print("VISUALIZATION COMPLETED")
    print("Images saved in:", output_dir)
    print("=" * 60)


if __name__ == "__main__":
    main()

