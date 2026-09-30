"""
multimodal_fusion.py
Late-Fusion Multimodal Model for Image-Text Matching (ITM).
Predicts whether a given caption accurately describes the image (True/1 or False/0).
Uses pre-trained ResNet50 (frozen) for the image branch and Embedding + GlobalAveragePooling for the text branch.
"""

import os
import sys
import glob
import re
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

# Set random seeds for reproducibility
# np.random.seed(42)
tf.random.set_seed(42)


# ==========================================
# 1. Dataset Discovery and Preprocessing
# ==========================================

def find_dataset_files(data_dir="data"):
    """
    Locates the captions file and images directory within data_dir.
    Handles various Flickr8k directory layouts (e.g., adityajn105/flickr8k).
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory '{data_dir}' not found.")

    # Find captions file
    captions_file = None
    potential_caption_files = [
        os.path.join(data_dir, "captions.txt"),
        os.path.join(data_dir, "Flickr8k.token.txt"),
        os.path.join(data_dir, "Flickr8k_text", "Flickr8k.token.txt"),
        os.path.join(data_dir, "captions.csv"),
    ]
    for p in potential_caption_files:
        if os.path.exists(p):
            captions_file = p
            break

    if captions_file is None:
        # Search recursively for any captions/token text file
        found_txts = glob.glob(os.path.join(data_dir, "**", "*caption*.txt"), recursive=True) + \
                     glob.glob(os.path.join(data_dir, "**", "*token*.txt"), recursive=True) + \
                     glob.glob(os.path.join(data_dir, "**", "*captions*.csv"), recursive=True)
        if found_txts:
            captions_file = found_txts[0]

    # Find images directory
    images_dir = None
    potential_image_dirs = [
        os.path.join(data_dir, "Images"),
        os.path.join(data_dir, "images"),
        os.path.join(data_dir, "Flicker8k_Dataset"),
        os.path.join(data_dir, "Flickr8k_Dataset"),
    ]
    for d in potential_image_dirs:
        if os.path.isdir(d):
            images_dir = d
            break

    if images_dir is None:
        # Look for directory containing .jpg files
        jpg_files = glob.glob(os.path.join(data_dir, "**", "*.jpg"), recursive=True)
        if jpg_files:
            images_dir = os.path.dirname(jpg_files[0])

    if not captions_file or not images_dir:
        raise FileNotFoundError(
            f"Could not automatically locate captions and images in '{data_dir}'. "
            f"Captions: {captions_file}, Images: {images_dir}"
        )

    print(f"[Dataset] Captions file: {captions_file}")
    print(f"[Dataset] Images folder: {images_dir}")
    return captions_file, images_dir


def load_captions(captions_file, images_dir):
    """
    Parses captions file and pairs each valid image path with its caption.
    Supports comma-separated (image,caption) and tab/space-separated (image#0 caption) formats.
    """
    records = []
    with open(captions_file, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    header_skipped = False
    for line in lines:
        line = line.strip()
        if not line:
            continue

        img_name = None
        caption_text = None

        if "," in line and ("image" in line.lower() and "caption" in line.lower()) and not header_skipped:
            header_skipped = True
            continue

        if "," in line:
            parts = line.split(",", 1)
            img_name = parts[0].strip()
            caption_text = parts[1].strip()
        elif "\t" in line:
            parts = line.split("\t", 1)
            img_name = parts[0].split("#")[0].strip()
            caption_text = parts[1].strip()
        else:
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                img_name = parts[0].split("#")[0].strip()
                caption_text = parts[1].strip()

        if img_name and caption_text:
            if not img_name.lower().endswith(".jpg"):
                img_name = img_name + ".jpg"
            img_path = os.path.join(images_dir, img_name)
            if os.path.exists(img_path):
                records.append((img_path, caption_text))

    df = pd.DataFrame(records, columns=["image_path", "caption"])
    print(f"[Dataset] Total valid image-caption pairs loaded: {len(df)}")
    return df


# ==========================================
# 2. Image-Text Matching Pair Generation
# ==========================================

def generate_itm_pairs(df):
    """
    Generates positive and negative image-text pairs for the ITM task.
    Positive pairs (Label 1): Original image + correct caption.
    Negative pairs (Label 0): Original image + randomly sampled mismatched caption.
    """
    print("[ITM] Generating positive and negative image-text pairs...")
    
    # 1. Create positive pairs
    pos_df = df.copy()
    pos_df["label"] = 1.0
    
    # 2. Create negative pairs by shuffling captions
    neg_df = df.copy()
    shuffled_captions = neg_df["caption"].sample(frac=1, random_state=42).values
    
    # Ensure they are actually mismatched
    original_captions = neg_df["caption"].values
    for i in range(len(shuffled_captions)):
        if shuffled_captions[i] == original_captions[i]:
            shuffled_captions[i] = original_captions[(i + 1) % len(original_captions)]
            
    neg_df["caption"] = shuffled_captions
    neg_df["label"] = 0.0
    
    # 3. Combine and shuffle the dataset
    itm_df = pd.concat([pos_df, neg_df], ignore_index=True)
    itm_df = itm_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    pos_ratio = (itm_df["label"] == 1.0).mean() * 100
    print(f"[ITM] Created {len(itm_df)} total pairs -> Match (1): {pos_ratio:.1f}%, Mismatch (0): {100-pos_ratio:.1f}%")
    return itm_df


# ==========================================
# 3. TensorFlow Data Pipeline
# ==========================================

def load_and_preprocess_image(path):
    """Loads image, decodes JPEG, resizes to (224, 224), and applies ResNet50 preprocessing."""
    img_raw = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_raw, channels=3)
    img = tf.image.resize(img, [224, 224])
    img = preprocess_input(img)
    return img


def build_tf_dataset(image_paths, tokenized_texts, labels, batch_size=32, shuffle=True):
    """
    Creates a tf.data.Dataset yielding a tuple of inputs:
    ({'image_input': image_tensor, 'text_input': token_tensor}, label)
    """
    image_paths = tf.constant(image_paths, dtype=tf.string)
    tokenized_texts = tf.constant(tokenized_texts, dtype=tf.int32)
    labels = tf.constant(labels, dtype=tf.float32)

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, tokenized_texts, labels))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=min(len(image_paths), 2000), seed=42)

    def _map_fn(img_p, tok_t, lbl):
        img_tensor = load_and_preprocess_image(img_p)
        return {"image_input": img_tensor, "text_input": tok_t}, lbl

    dataset = dataset.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.cache()
    if shuffle:
        dataset = dataset.shuffle(buffer_size=min(len(image_paths), 2000), seed=42)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


# ==========================================
# 4. Multimodal Late-Fusion Architecture
# ==========================================

def build_multimodal_late_fusion_model(vocab_size=5000, max_seq_len=30, embedding_dim=128):
    """
    Constructs a Late-Fusion Multimodal Model:
      - Image branch: Pre-trained ResNet50 (weights frozen) + GlobalAveragePooling2D + Dense projection
      - Text branch: Embedding layer + GlobalAveragePooling1D + Dense projection
      - Late-fusion: Concatenation layer followed by Dense classification head
    """
    print("\n[Architecture] Building Late-Fusion Multimodal Model...")

    # --- Image Branch ---
    image_input = layers.Input(shape=(224, 224, 3), name="image_input")
    base_resnet = ResNet50(
        weights="imagenet",
        include_top=False,
        pooling="avg"
    )
    # Freeze ResNet50 weights
    base_resnet.trainable = False

    image_features = base_resnet(image_input)
    image_dense = layers.Dense(128, activation="relu", name="image_projection")(image_features)
    image_dense = layers.Dropout(0.2, name="image_dropout")(image_dense)

    # --- Text Branch ---
    text_input = layers.Input(shape=(max_seq_len,), dtype="int32", name="text_input")
    text_embedding = layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        mask_zero=True,
        name="text_embedding"
    )(text_input)
    text_gap = layers.GlobalAveragePooling1D(name="text_gap")(text_embedding)
    text_dense = layers.Dense(128, activation="relu", name="text_projection")(text_gap)
    text_dense = layers.Dropout(0.2, name="text_dropout")(text_dense)

    # --- Late-Fusion (Decision/Feature Combination) ---
    fused_features = layers.Concatenate(name="late_fusion_concat")([image_dense, text_dense])
    fusion_dense = layers.Dense(128, activation="relu", name="fusion_dense_1")(fused_features)
    fusion_dense = layers.Dropout(0.3, name="fusion_dropout")(fusion_dense)
    fusion_dense = layers.Dense(64, activation="relu", name="fusion_dense_2")(fusion_dense)

    # --- Binary Classification Head ---
    itm_output = layers.Dense(1, activation="sigmoid", name="itm_output")(fusion_dense)

    model = Model(
        inputs=[image_input, text_input],
        outputs=itm_output,
        name="LateFusion_ResNet50_TextGAP_ITM"
    )

    return model


# ==========================================
# 5. Training and Evaluation Pipeline
# ==========================================

def run_pipeline(args):
    print("=" * 70)
    print("MULTIMODAL LATE-FUSION IMAGE-TEXT MATCHING (IMAGE + TEXT)")
    print("=" * 70)

    # 1. Locate files
    captions_file, images_dir = find_dataset_files(args.data_dir)

    # 2. Load captions and sample
    df = load_captions(captions_file, images_dir)

    if args.max_samples and 0 < args.max_samples < len(df):
        print(f"[Sampling] Selecting {args.max_samples} samples for fast efficient training...")
        df = df.sample(n=args.max_samples, random_state=42).reset_index(drop=True)

    # 3. Generate ITM Pairs
    df = generate_itm_pairs(df)

    # 4. Text Vectorization
    print("\n[NLP] Adapting TextVectorization on captions...")
    vectorizer = layers.TextVectorization(
        max_tokens=args.vocab_size,
        output_sequence_length=args.max_seq_len,
        standardize="lower_and_strip_punctuation",
        split="whitespace",
        output_mode="int"
    )
    vectorizer.adapt(df["caption"].values)

    tokenized_captions = vectorizer(df["caption"].values).numpy()
    actual_vocab_size = len(vectorizer.get_vocabulary())
    print(f"[NLP] Vocabulary adapted: {actual_vocab_size} tokens (max {args.vocab_size})")

    # 5. Train/Validation Split (80% / 20%)
    indices = np.arange(len(df))
    np.random.shuffle(indices)
    split_idx = int(len(indices) * 0.8)
    train_idx, val_idx = indices[:split_idx], indices[split_idx:]

    train_paths = df["image_path"].values[train_idx]
    train_tokens = tokenized_captions[train_idx]
    train_labels = df["label"].values[train_idx]

    val_paths = df["image_path"].values[val_idx]
    val_tokens = tokenized_captions[val_idx]
    val_labels = df["label"].values[val_idx]

    print(f"[Split] Train samples: {len(train_paths)}, Validation samples: {len(val_paths)}")

    # 6. Build tf.data Datasets
    print("\n[Pipeline] Constructing tf.data.Dataset pipelines...")
    train_ds = build_tf_dataset(train_paths, train_tokens, train_labels, batch_size=args.batch_size, shuffle=True)
    val_ds = build_tf_dataset(val_paths, val_tokens, val_labels, batch_size=args.batch_size, shuffle=False)

    # 7. Model Instantiation & Compilation
    model = build_multimodal_late_fusion_model(
        vocab_size=args.vocab_size,
        max_seq_len=args.max_seq_len,
        embedding_dim=args.embedding_dim
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.lr),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc")
        ]
    )

    model.summary(print_fn=lambda x: print(f"  {x}"))

    # Verify frozen status
    resnet_layer = model.get_layer("resnet50")
    print(f"\n[Verification] ResNet50 trainable: {resnet_layer.trainable}")
    trainable_count = sum(np.prod(v.shape) for v in model.trainable_weights)
    non_trainable_count = sum(np.prod(v.shape) for v in model.non_trainable_weights)
    print(f"[Parameters] Trainable: {trainable_count:,} | Non-trainable (frozen): {non_trainable_count:,}")

    # 8. Train the model
    print(f"\n[Training] Starting training for {args.epochs} epochs (batch size: {args.batch_size})...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        verbose=1
    )

    # 9. Final Evaluation
    print("\n" + "=" * 70)
    print("FINAL EVALUATION ON VALIDATION SET")
    print("=" * 70)
    eval_results = model.evaluate(val_ds, return_dict=True, verbose=0)
    for name, val in eval_results.items():
        print(f"  - Validation {name.upper()}: {val:.4f}")

    # 10. Sample Predictions Demonstration
    print("\n" + "=" * 70)
    print("SAMPLE MULTIMODAL PREDICTIONS")
    print("=" * 70)
    for sample_batch in val_ds.take(1):
        sample_inputs, sample_true_labels = sample_batch
        preds = model.predict(sample_inputs, verbose=0).flatten()

        # Display first 5 samples
        for i in range(min(5, len(preds))):
            pred_score = preds[i]
            pred_class = "Match (1)" if pred_score >= 0.5 else "Mismatch (0)"
            true_class = "Match (1)" if sample_true_labels[i].numpy() == 1.0 else "Mismatch (0)"
            print(f"Sample {i+1}:")
            print(f"  Predicted Probability: {pred_score:.4f} -> {pred_class}")
            print(f"  True Label:           {true_class}")
            print("-" * 40)

    # 11. Save model
    model_save_path = "multimodal_late_fusion_itm_model.keras"
    model.save(model_save_path)
    print(f"\n[Saved] Model weights and architecture saved to: {model_save_path}")
    print("[Complete] Multimodal late-fusion training pipeline finished successfully!")


# ==========================================
# 6. Main Entry Point
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multimodal Late-Fusion Model Training")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory where Flickr8k is extracted")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--max_samples", type=int, default=1500,
                        help="Max dataset samples for efficient CPU training (0 for all)")
    parser.add_argument("--vocab_size", type=int, default=5000, help="Vocabulary size for TextVectorization")
    parser.add_argument("--max_seq_len", type=int, default=30, help="Maximum caption sequence length")
    parser.add_argument("--embedding_dim", type=int, default=128, help="Embedding dimension for text branch")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")

    args = parser.parse_args()
    run_pipeline(args)
