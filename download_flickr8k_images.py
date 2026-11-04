
import os
import urllib.request
import zipfile
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
IMAGE_DIR = DATA_DIR / "Images"

ZIP_PATH = DATA_DIR / "Flickr8k_Dataset.zip"

URL = (
    "https://github.com/jbrownlee/Datasets/releases/download/"
    "Flickr8k/Flickr8k_Dataset.zip"
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def download_dataset():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    if not ZIP_PATH.exists():
        print("Starting Flickr8k image download...")
        print("Download size is approximately 1 GB.")

        def report_progress(block_num, block_size, total_size):
            if total_size > 0:
                downloaded = block_num * block_size
                percent = min(100, downloaded * 100 / total_size)
                mb = downloaded / (1024 * 1024)
                total_mb = total_size / (1024 * 1024)

                print(
                    f"\rDownloaded: {mb:.1f}/{total_mb:.1f} MB "
                    f"({percent:.1f}%)",
                    end=""
                )

        urllib.request.urlretrieve(
            URL,
            ZIP_PATH,
            reporthook=report_progress
        )

        print("\nDownload completed.")
    else:
        print("ZIP archive already exists. Skipping download.")

    if not zipfile.is_zipfile(ZIP_PATH):
        raise RuntimeError(
            "Downloaded file is not a valid ZIP archive. "
            "Delete the ZIP and run the script again."
        )

    print("\nExtracting images...")

    extracted = 0

    with zipfile.ZipFile(ZIP_PATH, "r") as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue

            filename = Path(info.filename).name

            if Path(filename).suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            destination = IMAGE_DIR / filename

            if destination.exists():
                continue

            with archive.open(info) as source:
                with open(destination, "wb") as target:
                    while True:
                        chunk = source.read(1024 * 1024)
                        if not chunk:
                            break
                        target.write(chunk)

            extracted += 1

            if extracted % 500 == 0:
                print(f"Extracted {extracted} new images...")

    print("\nExtraction completed.")

    images = [
        f for f in IMAGE_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    ]

    print("------------------------------------")
    print("TOTAL IMAGES:", len(images))
    print("IMAGE DIRECTORY:", IMAGE_DIR)
    print("------------------------------------")

    if len(images) < 8000:
        print("WARNING: Image count is still below 8000.")
    else:
        print("SUCCESS: Flickr8k image collection restored.")


if __name__ == "__main__":
    download_dataset()
