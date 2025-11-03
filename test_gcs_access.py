#!/usr/bin/env python3
"""
Test Google Cloud Storage access to SCIN dataset.
This script checks what files are actually available in the GCS bucket.
"""

import sys

def test_gcs_access():
    """Test access to GCS bucket and list contents."""
    try:
        from google.cloud import storage
    except ImportError:
        print("✗ google-cloud-storage not installed")
        print("  Run: uv pip install google-cloud-storage")
        return False

    print("="*80)
    print("Testing Google Cloud Storage Access")
    print("="*80)
    print()

    try:
        # Try to access the bucket
        print("Step 1: Attempting to access SCIN bucket...")
        print("-" * 80)

        storage_client = storage.Client(project="dx-scin-public")
        bucket = storage_client.bucket("dx-scin-public-data")

        print(f"✓ Connected to bucket: dx-scin-public-data")
        print()

        # List first few blobs
        print("Step 2: Listing bucket contents...")
        print("-" * 80)

        blobs = list(bucket.list_blobs(max_results=50))

        print(f"✓ Found {len(blobs)} items in bucket (showing first 50):")
        print()

        for i, blob in enumerate(blobs, 1):
            size_mb = blob.size / (1024 * 1024) if blob.size else 0
            print(f"{i:3d}. {blob.name:<70s} ({size_mb:>8.2f} MB)")

        # Look for data structure
        print()
        print("Step 3: Analyzing bucket structure...")
        print("-" * 80)

        csv_files = [b.name for b in blobs if b.name.endswith('.csv')]
        image_files = [b.name for b in blobs if b.name.endswith(('.jpg', '.png'))]

        print(f"CSV files: {len(csv_files)}")
        for csv_file in csv_files[:5]:
            print(f"  - {csv_file}")

        print()
        print(f"Image files: {len(image_files)}")
        for img_file in image_files[:5]:
            print(f"  - {img_file}")

        return True

    except Exception as e:
        print(f"✗ Error accessing bucket: {e}")
        print()
        print("This could mean:")
        print("1. The bucket requires authentication")
        print("2. You don't have permission to access it")
        print("3. The bucket name is incorrect")
        print()
        print("Alternative: Use HAM10000 dataset instead")
        print("  pip install kaggle")
        print("  kaggle datasets download -d kmader/skin-cancer-mnist-ham10000")
        return False

if __name__ == "__main__":
    success = test_gcs_access()
    sys.exit(0 if success else 1)
