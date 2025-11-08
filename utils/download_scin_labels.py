#!/usr/bin/env python3
"""
Download the SCIN labels CSV from Google Cloud Storage.
"""

from google.cloud import storage
from pathlib import Path

def download_labels():
    """Download SCIN labels file."""
    bucket_name = "dx-scin-public-data"
    labels_blob_name = "dataset/scin_labels.csv"
    output_path = Path("/home/sankar/data/scin/scin_labels.csv")

    print("Downloading SCIN labels...")
    print("="*60)

    try:
        # Connect anonymously
        client = storage.Client.create_anonymous_client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(labels_blob_name)

        # Download
        print(f"Downloading: {labels_blob_name}")
        print(f"To: {output_path}")

        blob.download_to_filename(str(output_path))

        print(f"\n✓ Labels downloaded successfully!")
        print(f"  File: {output_path}")
        print(f"  Size: {output_path.stat().st_size:,} bytes")

        # Show first few lines
        print("\nFirst 10 lines of labels file:")
        print("-"*60)
        with open(output_path, 'r') as f:
            for i, line in enumerate(f):
                if i >= 10:
                    break
                print(line.rstrip())

    except Exception as e:
        print(f"Error downloading labels: {e}")
        return False

    return True

if __name__ == "__main__":
    download_labels()