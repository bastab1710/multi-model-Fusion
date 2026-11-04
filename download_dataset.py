"""
download_dataset.py
Automates downloading and preparing the Flickr8k dataset for the Multimodal Fusion model.
Uses Kaggle API (dataset: adityajn105/flickr8k) or validates existing local zip archives.
"""

import os
import sys
import glob
import shutil
import zipfile
import argparse

KAGGLE_DATASET = "adityajn105/flickr8k"
DEFAULT_ZIP_NAME = "flickr8k.zip"
DEFAULT_DATA_DIR = "data"


def check_existing_images(data_dir=DEFAULT_DATA_DIR):
    """Checks whether images are already extracted and present."""
    if not os.path.exists(data_dir):
        return 0, None

    jpg_files = glob.glob(os.path.join(data_dir, "**", "*.jpg"), recursive=True)
    if jpg_files:
        img_dir = os.path.dirname(jpg_files[0])
        return len(jpg_files), img_dir
    return 0, None


def extract_zip(zip_path, extract_to=DEFAULT_DATA_DIR):
    """Extracts zip archive with progress reporting."""
    print(f"\n[Extraction] Opening '{zip_path}' ({os.path.getsize(zip_path) / (1024*1024):.1f} MB)...")
    os.makedirs(extract_to, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.infolist()
        total = len(members)
        print(f"[Extraction] Extracting {total} items into '{extract_to}'...")

        for idx, member in enumerate(members):
            zf.extract(member, extract_to)
            if (idx + 1) % 1500 == 0 or (idx + 1) == total:
                print(f"  Extracted {idx + 1}/{total} files ({(idx + 1) / total * 100:.1f}%)...")

    print("[Extraction] Done!")


def download_with_kaggle(dest_dir="."):
    """Attempts download using the kaggle package/API."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print("\n[Error] The 'kaggle' package is not installed.")
        print("Install it with: pip install kaggle")
        return False

    # Check for kaggle credentials (~/.kaggle/kaggle.json)
    kaggle_cfg = os.path.expanduser("~/.kaggle/kaggle.json")
    if not os.path.exists(kaggle_cfg):
        print("\n" + "=" * 65)
        print("[Kaggle Setup Required]")
        print("Could not find Kaggle API credentials at:")
        print(f"  {kaggle_cfg}")
        print("\nTo set up your Kaggle API key:")
        print("  1. Go to https://www.kaggle.com/settings")
        print("  2. Scroll down to 'API' section and click 'Create New Token'")
        print("  3. A file named 'kaggle.json' will be downloaded.")
        print(f"  4. Move 'kaggle.json' to: {os.path.dirname(kaggle_cfg)}")
        print("=" * 65)
        return False

    print(f"\n[Kaggle API] Authenticating and downloading dataset '{KAGGLE_DATASET}'...")
    try:
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(KAGGLE_DATASET, path=dest_dir, unzip=False, quiet=False)
        print(f"[Kaggle API] Dataset successfully downloaded to '{dest_dir}'.")
        return True
    except Exception as e:
        print(f"[Kaggle API] Download failed: {e}")
        return False


def verify_dataset(data_dir=DEFAULT_DATA_DIR):
    """Verifies that captions and images are properly positioned."""
    print("\n" + "=" * 60)
    print("DATASET VERIFICATION")
    print("=" * 60)

    # Check captions file
    captions_file = None
    for candidate in [
        os.path.join(data_dir, "captions.txt"),
        os.path.join(data_dir, "captions.csv"),
        os.path.join(data_dir, "Flickr8k.token.txt"),
    ]:
        if os.path.exists(candidate):
            captions_file = candidate
            break

    if captions_file:
        with open(captions_file, "r", encoding="utf-8", errors="ignore") as f:
            total_lines = sum(1 for line in f if line.strip())
        print(f"  [OK] Captions file found: {captions_file} ({total_lines:,} caption entries)")
    else:
        print(f"  [MISSING] Captions file not found in '{data_dir}'.")

    # Check images
    img_count, img_dir = check_existing_images(data_dir)
    if img_count > 0:
        print(f"  [OK] Images found: {img_count:,} JPEG images in '{img_dir}'")
    else:
        print(f"  [MISSING] No .jpg images found inside '{data_dir}'.")

    if captions_file and img_count > 0:
        print("\n--> Dataset is complete and ready for training!")
    else:
        print("\n--> Dataset is incomplete. Please extract or download missing components.")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Download & Setup Flickr8k Dataset")
    parser.add_argument("--zip_path", type=str, default=DEFAULT_ZIP_NAME, help="Path to local flickr8k.zip if already downloaded")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR, help="Destination directory for dataset")
    parser.add_argument("--skip_download", action="store_true", help="Only extract/verify existing local files")
    args = parser.parse_args()

    # 1. Check if already extracted
    img_count, img_dir = check_existing_images(args.data_dir)
    if img_count >= 8000:
        print(f"[Info] Found {img_count} images already extracted in '{img_dir}'.")
        verify_dataset(args.data_dir)
        return

    # 2. Check if local zip exists
    found_zip = None
    for candidate in [args.zip_path, os.path.join(args.data_dir, args.zip_path), "flickr8k.zip"]:
        if os.path.exists(candidate):
            found_zip = candidate
            break

    if found_zip:
        print(f"[Info] Found existing archive at '{found_zip}'.")
        extract_zip(found_zip, extract_to=args.data_dir)
        verify_dataset(args.data_dir)
        return

    # 3. Attempt Kaggle download if requested
    if not args.skip_download:
        print(f"[Info] No local zip found at '{args.zip_path}'. Attempting Kaggle download...")
        success = download_with_kaggle(dest_dir=".")
        if success:
            downloaded_zip = os.path.join(".", "flickr8k.zip")
            if os.path.exists(downloaded_zip):
                extract_zip(downloaded_zip, extract_to=args.data_dir)
                verify_dataset(args.data_dir)
                return

    # 4. Fallback manual instructions
    print("\n" + "=" * 65)
    print("MANUAL DOWNLOAD INSTRUCTIONS:")
    print("=" * 65)
    print("1. Download the Flickr8k dataset from Kaggle:")
    print("   https://www.kaggle.com/datasets/adityajn105/flickr8k")
    print(f"2. Save 'flickr8k.zip' directly to the project root: {os.path.abspath('.')}")
    print("3. Re-run this script:")
    print("   python download_dataset.py")
    print("=" * 65)


if __name__ == "__main__":
    main()

