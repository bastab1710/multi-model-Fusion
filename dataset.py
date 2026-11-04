"""
dataset.py
PyTorch Dataset and Pair Generator for Multimodal Image-Text Matching (ITM).
Creates balanced positive pairs (label 1.0) and negative pairs (label 0.0).
"""

import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


# Standard ImageNet normalization for ResNet50
IMAGE_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


class ITMDataset(Dataset):
    """
    PyTorch Dataset for Image-Text Matching.
    Yields:
      image_tensor: shape (3, 224, 224)
      text_tokens: shape (max_seq_len,)
      label: float scalar (1.0 = match, 0.0 = mismatch)
    """
    def __init__(self, pairs, tokenizer, transform=IMAGE_TRANSFORMS):
        """
        pairs: list of tuples (image_path, caption_text, label_float)
        tokenizer: instance of CaptionTokenizer
        transform: torchvision image transforms
        """
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, caption, label = self.pairs[idx]

        # Load and transform image
        try:
            image = Image.open(img_path).convert("RGB")
            img_tensor = self.transform(image)
        except Exception as e:
            # Fallback zero tensor if image read fails
            img_tensor = torch.zeros(3, 224, 224)

        # Tokenize caption
        tokens = self.tokenizer.encode(caption)
        token_tensor = torch.tensor(tokens, dtype=torch.long)

        label_tensor = torch.tensor(label, dtype=torch.float32)

        return img_tensor, token_tensor, label_tensor


def build_itm_pairs(raw_records, hard_negative_ratio=0.5, seed=42):
    """
    Given raw (image_path, caption) pairs:
      1. Positive pairs: image + matching caption (label 1.0)
      2. Negative pairs: image + mismatched caption (label 0.0)
    Generates balanced positive and negative pairs.
    """
    rng = random.Random(seed)
    positives = []
    for img_path, caption in raw_records:
        positives.append((img_path, caption, 1.0))

    # All unique captions
    all_captions = [cap for _, cap in raw_records]
    negatives = []

    for img_path, orig_caption in raw_records:
        # Sample mismatched caption
        sampled_caption = rng.choice(all_captions)
        while sampled_caption == orig_caption and len(all_captions) > 1:
            sampled_caption = rng.choice(all_captions)

        negatives.append((img_path, sampled_caption, 0.0))

    combined = positives + negatives
    rng.shuffle(combined)
    return combined


def load_dataset_records(captions_file="data/captions.txt", images_dir="data/Images"):
    """
    Loads all caption records from file and pairs them with existing image files.
    """
    if not os.path.exists(captions_file):
        raise FileNotFoundError(f"Captions file '{captions_file}' not found.")

    records = []
    with open(captions_file, "r", encoding="utf-8", errors="ignore") as f:
        header = f.readline()
        for line in f:
            line = line.strip()
            if not line:
                continue

            if "," in line:
                parts = line.split(",", 1)
            elif "\t" in line:
                parts = line.split("\t", 1)
            else:
                parts = line.split(maxsplit=1)

            if len(parts) >= 2:
                img_name = parts[0].strip()
                caption = parts[1].strip()
                if not img_name.lower().endswith(".jpg"):
                    img_name += ".jpg"

                img_path = os.path.join(images_dir, img_name)
                if os.path.exists(img_path):
                    records.append((img_path, caption))

    return records

