from pathlib import Path

import numpy as np
import tifffile
import torch
from PIL import Image
from torch.utils.data import Dataset


# --------------------------------------------------
# PATHS AND CONFIGURATION
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[1]

DATASET_DIR = (
    PROJECT_DIR
    / "data"
    / "raw"
    / "WeedyRice-RGBMS-DB"
)

BANDS = ["_G.TIF", "_R.TIF", "_RE.TIF", "_NIR.TIF"]

VALID_SPLITS = ["train", "val", "test"]

VALID_MODES = [
    "rgb",
    "multispectral",
    "early_fusion",
    "late_fusion",
]


# --------------------------------------------------
# DATASET CLASS
# --------------------------------------------------

class WeedyRiceDataset(Dataset):

    def __init__(
        self,
        split="train",
        mode="rgb",
        image_size=(192, 256),
        dataset_dir=DATASET_DIR,
    ):

        if split not in VALID_SPLITS:
            raise ValueError(
                f"Invalid split: {split}. "
                f"Choose from {VALID_SPLITS}"
            )

        if mode not in VALID_MODES:
            raise ValueError(
                f"Invalid mode: {mode}. "
                f"Choose from {VALID_MODES}"
            )

        self.dataset_dir = Path(dataset_dir)
        self.split = split
        self.mode = mode

        # Format: (height, width)
        self.image_size = image_size

        split_file = self.dataset_dir / f"{split}_list.txt"

        if not split_file.exists():
            raise FileNotFoundError(
                f"Split file not found: {split_file}"
            )

        self.filenames = [
            line.strip()
            for line in split_file.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]

        if not self.filenames:
            raise ValueError(
                f"No samples found in {split_file}"
            )

        print(
            f"Loaded {len(self.filenames)} samples "
            f"for {split} split ({mode} mode)"
        )

    def __len__(self):
        return len(self.filenames)

    # --------------------------------------------------
    # LOAD RGB IMAGE
    # --------------------------------------------------

    def load_rgb(self, filename):

        path = self.dataset_dir / "RGB" / filename

        with Image.open(path) as image:
            image = image.convert("RGB")

            image = image.resize(
                (self.image_size[1], self.image_size[0]),
                Image.Resampling.BILINEAR,
            )

            array = np.asarray(
                image,
                dtype=np.float32
            ).copy()

        array = array / 255.0

        # HWC -> CHW
        array = np.transpose(array, (2, 0, 1))

        return torch.from_numpy(array)

    # --------------------------------------------------
    # LOAD FOUR MULTISPECTRAL BANDS
    # --------------------------------------------------

    def load_multispectral(self, stem):

        band_arrays = []

        for band in BANDS:

            path = (
                self.dataset_dir
                / "Multispectral"
                / f"{stem}{band}"
            )

            data = tifffile.imread(path)

            # Each TIFF contains three identical channels.
            # Extract only the first channel.
            if data.ndim == 3:
                data = data[:, :, 0]

            if data.ndim != 2:
                raise ValueError(
                    f"Unexpected TIFF shape {data.shape}: {path}"
                )

            image = Image.fromarray(data)

            image = image.resize(
                (self.image_size[1], self.image_size[0]),
                Image.Resampling.BILINEAR,
            )

            array = np.asarray(
                image,
                dtype=np.float32
            ).copy()

            band_arrays.append(array / 255.0)

        # Four bands -> (4, H, W)
        multispectral = np.stack(
            band_arrays,
            axis=0
        )

        return torch.from_numpy(multispectral)

    # --------------------------------------------------
    # LOAD SEGMENTATION MASK
    # --------------------------------------------------

    def load_mask(self, stem):

        path = self.dataset_dir / "Masks" / f"{stem}.png"

        with Image.open(path) as image:
            image = image.convert("L")

            image = image.resize(
                (self.image_size[1], self.image_size[0]),
                Image.Resampling.NEAREST,
            )

            mask = np.asarray(image).copy()

        # Convert 0/255 into 0/1.
        mask = (mask > 0).astype(np.float32)

        # Add channel dimension: (1, H, W)
        mask = np.expand_dims(mask, axis=0)

        return torch.from_numpy(mask)

    # --------------------------------------------------
    # GET ONE SAMPLE
    # --------------------------------------------------

    def __getitem__(self, index):

        filename = self.filenames[index]
        stem = Path(filename).stem

        mask = self.load_mask(stem)

        sample = {
            "mask": mask,
            "filename": filename,
        }

        if self.mode in ["rgb", "early_fusion", "late_fusion"]:
            rgb = self.load_rgb(filename)

        if self.mode in [
            "multispectral",
            "early_fusion",
            "late_fusion",
        ]:
            multispectral = self.load_multispectral(stem)

        if self.mode == "rgb":
            sample["image"] = rgb

        elif self.mode == "multispectral":
            sample["image"] = multispectral

        elif self.mode == "early_fusion":
            sample["image"] = torch.cat(
                [rgb, multispectral],
                dim=0
            )

        elif self.mode == "late_fusion":
            sample["rgb"] = rgb
            sample["multispectral"] = multispectral

        return sample


# --------------------------------------------------
# QUICK MANUAL TEST
# --------------------------------------------------

if __name__ == "__main__":

    for mode in VALID_MODES:

        print("\n" + "=" * 50)
        print("TESTING MODE:", mode)

        dataset = WeedyRiceDataset(
            split="train",
            mode=mode,
        )

        sample = dataset[0]

        if mode == "late_fusion":
            print("RGB shape:", sample["rgb"].shape)
            print(
                "Multispectral shape:",
                sample["multispectral"].shape
            )
        else:
            print("Image shape:", sample["image"].shape)

        print("Mask shape:", sample["mask"].shape)
        print("Mask values:", torch.unique(sample["mask"]))
        print("Filename:", sample["filename"])

    print("\nALL DATASET MODES TESTED")
