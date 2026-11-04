
import os
from collections import Counter

# ==========================================
# FLICKR8K DATASET VERIFICATION
# ==========================================

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

POSSIBLE_IMAGE_DIRS = [
    os.path.join(PROJECT_DIR, "data", "Images"),
    os.path.join(PROJECT_DIR, "data", "Flickr8k_Dataset"),
    os.path.join(PROJECT_DIR, "Flickr8k_Dataset"),
    os.path.join(PROJECT_DIR, "Images"),
]

POSSIBLE_CAPTION_FILES = [
    os.path.join(PROJECT_DIR, "data", "captions.txt"),
    os.path.join(PROJECT_DIR, "captions.txt"),
    os.path.join(PROJECT_DIR, "data", "Flickr8k.token.txt"),
    os.path.join(PROJECT_DIR, "Flickr8k.token.txt"),
]

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def find_image_directory():
    for folder in POSSIBLE_IMAGE_DIRS:
        if os.path.isdir(folder):
            return folder
    return None


def find_caption_file():
    for file in POSSIBLE_CAPTION_FILES:
        if os.path.isfile(file):
            return file
    return None


def main():
    print("\n" + "=" * 55)
    print("       FLICKR8K DATASET VERIFICATION")
    print("=" * 55)

    image_dir = find_image_directory()
    caption_file = find_caption_file()

    # Check images
    if image_dir:
        images = [
            f for f in os.listdir(image_dir)
            if f.lower().endswith(IMAGE_EXTENSIONS)
        ]
        image_names = set(images)

        print("\n[IMAGE CHECK]")
        print("Image folder:", image_dir)
        print("Total images:", len(images))
    else:
        images = []
        image_names = set()
        print("\n[IMAGE CHECK]")
        print("ERROR: Image folder not found.")

    # Check captions
    caption_counts = Counter()
    caption_rows = 0
    malformed_rows = 0
    caption_names = set()

    if caption_file:
        print("\n[CAPTION CHECK]")
        print("Caption file:", caption_file)

        with open(caption_file, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()

                if not line:
                    continue

                # Supports Flickr8k.token.txt format:
                # image.jpg#0<TAB>caption text
                if "\t" in line:
                    image_id, caption = line.split("\t", 1)
                    image_name = image_id.split("#")[0]

                # Supports captions.txt format:
                # image.jpg,caption text
                elif "," in line:
                    image_name, caption = line.split(",", 1)
                    image_name = image_name.strip().strip('"')
                    caption = caption.strip()

                    if image_name.lower() in (
                        "image", "filename", "image_name"
                    ):
                        continue
                else:
                    malformed_rows += 1
                    continue

                if not image_name or not caption:
                    malformed_rows += 1
                    continue

                caption_names.add(image_name)
                caption_counts[image_name] += 1
                caption_rows += 1

        print("Total caption entries:", caption_rows)
        print("Unique images in captions:", len(caption_counts))
        print("Malformed rows:", malformed_rows)
    else:
        print("\n[CAPTION CHECK]")
        print("ERROR: Caption file not found.")

    # Compare image and caption files
    if image_dir and caption_file:
        matched = image_names.intersection(caption_names)
        missing_images = caption_names - image_names
        images_without_captions = image_names - caption_names

        print("\n[IMAGE-CAPTION MATCHING]")
        print("Matched images:", len(matched))
        print("Caption entries without images:", len(missing_images))
        print("Images without captions:", len(images_without_captions))

        five_captions = sum(
            1 for count in caption_counts.values() if count == 5
        )

        print("\n[CAPTION DISTRIBUTION]")
        print("Images with exactly 5 captions:", five_captions)
        print("Images with fewer than 5 captions:",
              sum(1 for count in caption_counts.values() if count < 5))
        print("Images with more than 5 captions:",
              sum(1 for count in caption_counts.values() if count > 5))

        print("\n[SAMPLE IMAGE FILES]")
        for name in sorted(images)[:5]:
            print(" ", name)

        print("\n[SAMPLE CAPTIONS]")
        for name, count in list(caption_counts.items())[:5]:
            print(f"  {name}: {count} captions")

    print("\n" + "=" * 55)
    print("                 VERIFICATION SUMMARY")
    print("=" * 55)

    if not image_dir or not caption_file:
        print("RESULT: INCOMPLETE - Required files are missing.")
    elif len(images) >= 8000 and caption_rows >= 40000:
        print("RESULT: Dataset appears complete.")
        print("Check matching and caption distribution before training.")
    else:
        print("RESULT: Dataset may be incomplete.")
        print("We need to inspect the missing files.")

    print("=" * 55)


if __name__ == "__main__":
    main()
