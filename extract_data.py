"""
extract_data.py
Extracts flickr8k.zip into data/ directory with progress reporting.
"""

import os
import sys
import zipfile
import time

def extract_archive(zip_path="flickr8k.zip", extract_to="data"):
    if not os.path.exists(zip_path):
        print(f"Error: {zip_path} not found.")
        sys.exit(1)

    os.makedirs(extract_to, exist_ok=True)
    print(f"Opening archive: {zip_path} ({os.path.getsize(zip_path) / (1024*1024):.1f} MB)...")
    start_time = time.time()

    with zipfile.ZipFile(zip_path, 'r') as zf:
        members = zf.infolist()
        total = len(members)
        print(f"Extracting {total} files to '{extract_to}'...")

        for i, member in enumerate(members):
            zf.extract(member, extract_to)
            if (i + 1) % 1000 == 0 or (i + 1) == total:
                print(f"  Extracted {i + 1}/{total} files ({(i + 1)/total*100:.1f}%)...")

    elapsed = time.time() - start_time
    print(f"Extraction complete in {elapsed:.1f} seconds!")

if __name__ == "__main__":
    zip_p = sys.argv[1] if len(sys.argv) > 1 else "flickr8k.zip"
    dest_d = sys.argv[2] if len(sys.argv) > 2 else "data"
    extract_archive(zip_p, dest_d)
