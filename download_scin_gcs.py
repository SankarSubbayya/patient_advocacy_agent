#!/usr/bin/env python3
"""
Download SCIN dataset from Google Cloud Storage.

SCIN (Skin Condition Image Network) dataset is hosted on Google Cloud Storage.
Repository: https://github.com/google-research-datasets/scin
Bucket: dx-scin-public-data
"""

import os
import sys
import io
from pathlib import Path
import json
import csv
from typing import Optional, Tuple

def check_gcs_dependencies():
    """Check if required Google Cloud dependencies are installed."""
    try:
        from google.cloud import storage
        return True, None
    except ImportError as e:
        return False, str(e)

def install_gcs_dependencies():
    """Install required Google Cloud Storage dependencies."""
    print("\nInstalling Google Cloud Storage dependencies...")
    print("Run this command to install:")
    print("  pip install google-cloud-storage")
    print("\nOr with uv:")
    print("  uv pip install google-cloud-storage")
    print("\nThen run this script again.")
    return False

def download_from_gcs(
    bucket_name: str = "dx-scin-public-data",
    project_id: str = "dx-scin-public",
    output_dir: str = "data/scin"
) -> bool:
    """
    Download SCIN dataset from Google Cloud Storage.

    Args:
        bucket_name: GCS bucket name
        project_id: GCP project ID
        output_dir: Local output directory

    Returns:
        True if successful, False otherwise
    """
    try:
        from google.cloud import storage
        import pandas as pd
        from PIL import Image
    except ImportError as e:
        print(f"\n✗ Missing dependency: {e}")
        print("\nPlease install Google Cloud Storage:")
        print("  pip install google-cloud-storage pandas pillow")
        return False

    print("="*80)
    print("SCIN Dataset Downloader from Google Cloud Storage")
    print("="*80)
    print()

    # Configuration
    print(f"GCP Project: {project_id}")
    print(f"Bucket: {bucket_name}")
    print(f"Output Directory: {output_dir}")
    print()

    # Create output directories
    output_path = Path(output_dir)
    images_dir = output_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    print("Step 1: Connecting to Google Cloud Storage")
    print("-" * 80)

    try:
        # Initialize storage client
        storage_client = storage.Client(project=project_id)
        bucket = storage_client.bucket(bucket_name)

        print(f"✓ Connected to bucket: {bucket_name}")
    except Exception as e:
        print(f"✗ Failed to connect to GCS: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure you have google-cloud-storage installed")
        print("2. For public bucket access, you may not need authentication")
        print("3. Check internet connection")
        print("\nTo fix, run:")
        print("  pip install google-cloud-storage")
        return False

    print("\nStep 2: Downloading Metadata CSV")
    print("-" * 80)

    try:
        # Download metadata CSV
        csv_path = "dataset/scin_cases.csv"
        print(f"Downloading: {csv_path}")

        csv_blob = bucket.blob(csv_path)
        csv_data = csv_blob.download_as_string()
        metadata_df = pd.read_csv(io.BytesIO(csv_data))

        print(f"✓ Downloaded metadata with {len(metadata_df)} records")
        print(f"  Columns: {', '.join(metadata_df.columns[:5])}...")

    except Exception as e:
        print(f"✗ Failed to download metadata: {e}")
        print("\nNote: Public bucket access may be limited")
        return False

    print("\nStep 3: Downloading Images")
    print("-" * 80)

    try:
        total_images = len(metadata_df)
        downloaded = 0
        failed = 0

        for idx, row in metadata_df.iterrows():
            try:
                # Get image path from metadata
                image_id = row.get('case_id', row.get('id', f'image_{idx}'))
                image_filename = row.get('image_path', f'{image_id}.jpg')
                condition = row.get('condition', 'unknown')

                # GCS image path
                gcs_image_path = f"dataset/images/{image_filename}"

                # Local image path
                local_condition_dir = images_dir / condition
                local_condition_dir.mkdir(parents=True, exist_ok=True)
                local_image_path = local_condition_dir / Path(image_filename).name

                # Download image
                image_blob = bucket.blob(gcs_image_path)
                image_data = image_blob.download_as_string()

                # Save image
                with open(local_image_path, 'wb') as f:
                    f.write(image_data)

                downloaded += 1

                # Progress update
                if (idx + 1) % 100 == 0:
                    progress = ((idx + 1) / total_images) * 100
                    print(f"  Progress: {progress:.1f}% ({downloaded} images downloaded)", end='\r')

            except Exception as e:
                failed += 1
                if failed <= 5:  # Only print first 5 errors
                    print(f"\n  Warning: Failed to download {image_filename}: {e}")

        print(f"\n✓ Downloaded {downloaded} images ({failed} failed)")

    except KeyboardInterrupt:
        print(f"\n\n✗ Download interrupted by user")
        print(f"  Downloaded: {downloaded} images")
        return False

    except Exception as e:
        print(f"✗ Failed to download images: {e}")
        return False

    print("\nStep 4: Creating Metadata CSV")
    print("-" * 80)

    try:
        # Create local metadata CSV
        output_metadata = []

        for root, dirs, files in os.walk(images_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    rel_path = os.path.relpath(os.path.join(root, file), images_dir)
                    condition = os.path.basename(root) if root != str(images_dir) else 'unknown'

                    output_metadata.append({
                        'image_path': rel_path,
                        'condition': condition,
                        'split': 'train'
                    })

        # Save metadata CSV
        metadata_csv = output_path / "metadata.csv"
        with open(metadata_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['image_path', 'condition', 'split'])
            writer.writeheader()
            writer.writerows(output_metadata)

        print(f"✓ Created metadata CSV: {metadata_csv}")
        print(f"  Total images: {len(output_metadata)}")

    except Exception as e:
        print(f"✗ Failed to create metadata CSV: {e}")
        return False

    print("\n" + "="*80)
    print("✓ DOWNLOAD COMPLETE!")
    print("="*80)
    print(f"\nDataset Location: {output_dir}")
    print(f"Total Images: {len(output_metadata)}")
    print(f"Metadata File: {metadata_csv}")
    print("\nNext steps:")
    print("  1. Train embedder:")
    print("     uv run python train_embedder.py")
    print("  2. Build index:")
    print("     uv run python build_index.py")
    print()

    return True

def main():
    """Main download function."""
    # Check dependencies
    has_gcs, error = check_gcs_dependencies()

    if not has_gcs:
        print("\n✗ Google Cloud Storage library not found")
        print(f"Error: {error}")
        print("\n" + "="*80)
        print("INSTALLATION REQUIRED")
        print("="*80)

        response = input("\nWould you like to install google-cloud-storage? (y/n): ").lower().strip()

        if response == 'y':
            install_gcs_dependencies()
            sys.exit(0)
        else:
            print("\nAlternative options:")
            print("1. Use HAM10000 dataset instead:")
            print("   - Download from: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000")
            print("   - Then run: download_scin_dataset.py")
            print("\n2. Install google-cloud-storage manually:")
            print("   pip install google-cloud-storage")
            print("   Then run this script again")
            sys.exit(1)

    try:
        success = download_from_gcs()
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
