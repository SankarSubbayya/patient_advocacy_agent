#!/usr/bin/env python3
"""
Download SCIN dataset from Google Cloud Storage (Working Version).

SCIN (Skin Condition Image Network) dataset on GCS:
- Repository: https://github.com/google-research-datasets/scin
- Bucket: dx-scin-public-data
- Images: PNG format with numeric IDs
- Structure: dataset/images/{image_id}.png
"""

import os
import sys
from pathlib import Path
import csv

def download_from_gcs(
    bucket_name: str = "dx-scin-public-data",
    project_id: str = "dx-scin-public",
    output_dir: str = "data/scin",
    limit: int = None  # Set to None for all images, or integer for testing
) -> bool:
    """
    Download SCIN dataset from Google Cloud Storage.

    Args:
        bucket_name: GCS bucket name
        project_id: GCP project ID
        output_dir: Local output directory
        limit: Maximum number of images to download (None = all)

    Returns:
        True if successful, False otherwise
    """
    try:
        from google.cloud import storage
    except ImportError:
        print("✗ google-cloud-storage not installed")
        print("\nInstall with:")
        print("  pip install google-cloud-storage")
        print("  # or")
        print("  uv pip install google-cloud-storage")
        return False

    print("="*80)
    print("SCIN Dataset Downloader from Google Cloud Storage")
    print("="*80)
    print()

    # Configuration
    print(f"GCP Project: {project_id}")
    print(f"Bucket: {bucket_name}")
    print(f"Output Directory: {output_dir}")
    if limit:
        print(f"Limit: {limit} images (for testing)")
    print()

    # Create output directories
    output_path = Path(output_dir)
    images_dir = output_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    print("Step 1: Connecting to Google Cloud Storage")
    print("-" * 80)

    try:
        storage_client = storage.Client(project=project_id)
        bucket = storage_client.bucket(bucket_name)
        print(f"✓ Connected to bucket: {bucket_name}")
    except Exception as e:
        print(f"✗ Failed to connect: {e}")
        return False

    print("\nStep 2: Listing dataset images")
    print("-" * 80)

    try:
        # List all image blobs
        blobs = list(bucket.list_blobs(prefix="dataset/images/"))
        image_blobs = [b for b in blobs if b.name.endswith(('.png', '.jpg', '.jpeg'))]

        print(f"✓ Found {len(image_blobs)} images in bucket")

        if limit:
            image_blobs = image_blobs[:limit]
            print(f"  (limiting to {limit} for testing)")

    except Exception as e:
        print(f"✗ Failed to list images: {e}")
        return False

    print("\nStep 3: Downloading images")
    print("-" * 80)

    downloaded = 0
    failed = 0
    metadata = []

    try:
        total = len(image_blobs)

        for idx, blob in enumerate(image_blobs, 1):
            try:
                # Extract image ID from path: dataset/images/{id}.png -> {id}.png
                image_filename = Path(blob.name).name
                image_id = Path(image_filename).stem

                # Save to local directory
                local_image_path = images_dir / image_filename

                # Download
                blob.download_to_filename(str(local_image_path))

                downloaded += 1

                # Add to metadata
                # Note: image_path should be relative to the images_dir (data/scin/images/)
                metadata.append({
                    'image_id': image_id,
                    'image_path': image_filename,  # Just the filename, no 'images/' prefix
                    'condition': 'unlabeled',  # SCIN provides images but labels are in separate CSV
                    'split': 'train'
                })

                # Progress
                if idx % 50 == 0 or idx == total:
                    progress = (idx / total) * 100
                    print(f"  Progress: {progress:.1f}% ({idx}/{total} images)", end='\r')

            except Exception as e:
                failed += 1
                if failed <= 3:
                    print(f"\n  Warning: Failed to download {blob.name}: {str(e)[:60]}")

        print(f"\n✓ Downloaded {downloaded} images ({failed} failed)")

    except KeyboardInterrupt:
        print(f"\n\n✗ Download interrupted by user")
        print(f"  Downloaded: {downloaded} images before interruption")
        return False

    except Exception as e:
        print(f"✗ Download error: {e}")
        return False

    print("\nStep 4: Creating metadata CSV")
    print("-" * 80)

    try:
        metadata_csv = output_path / "metadata.csv"

        with open(metadata_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['image_id', 'image_path', 'condition', 'split'])
            writer.writeheader()
            writer.writerows(metadata)

        print(f"✓ Created metadata CSV: {metadata_csv}")
        print(f"  Total entries: {len(metadata)}")

    except Exception as e:
        print(f"✗ Failed to create metadata: {e}")
        return False

    print("\n" + "="*80)
    print("✓ DOWNLOAD COMPLETE!")
    print("="*80)
    print(f"\nDataset Location: {output_dir}")
    print(f"Total Images: {len(metadata)}")
    print(f"Metadata File: {metadata_csv}")
    print()

    if downloaded < 100:
        print("⚠ Note: Only downloaded limited images for testing")
        print("  To download all ~10,000 images, remove the 'limit' parameter")
    print()

    print("Next steps:")
    print("  1. Download labels from SCIN repository:")
    print("     https://github.com/google-research-datasets/scin")
    print("  2. Fine-tune embedder:")
    print("     uv run python train_embedder.py")
    print("  3. Build index:")
    print("     uv run python build_index.py")
    print()

    return True

def main():
    """Main download function."""
    try:
        # For full dataset, set limit=None
        # For testing, set limit=100 to download just 100 images
        success = download_from_gcs(limit=None)
        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        print("\n\nDownload interrupted by user")
        sys.exit(0)

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
