#!/usr/bin/env python3
"""
Download ONLY the SCIN images that have labels.

This script:
1. Reads the scin_labels.csv to get case IDs with labels
2. Downloads only those specific images from GCS
3. Creates a properly labeled metadata.csv
"""

import pandas as pd
from pathlib import Path
from google.cloud import storage
import ast

def parse_condition_labels(label_str):
    """Parse the condition label string into a list."""
    if pd.isna(label_str) or label_str == '[]':
        return []
    try:
        return ast.literal_eval(label_str)
    except:
        return []

def parse_weighted_labels(weighted_str):
    """Parse the weighted label dictionary string."""
    if pd.isna(weighted_str) or weighted_str == '{}':
        return {}
    try:
        return ast.literal_eval(weighted_str)
    except:
        return {}

def download_labeled_images(limit=None):
    """Download images that have labels and create labeled metadata."""

    print("="*80)
    print("SCIN Labeled Images Downloader")
    print("="*80)
    print()

    # Paths
    data_dir = Path("/home/sankar/data/scin")
    labels_path = data_dir / "scin_labels.csv"
    images_dir = data_dir / "labeled_images"
    output_metadata = data_dir / "labeled_metadata.csv"

    # Create directory for labeled images
    images_dir.mkdir(parents=True, exist_ok=True)

    # Load labels
    print("Step 1: Loading labels...")
    labels_df = pd.read_csv(labels_path)
    print(f"  Loaded {len(labels_df)} labeled cases")

    # Filter to only cases with actual labels
    labels_df['condition_list'] = labels_df['dermatologist_skin_condition_on_label_name'].apply(parse_condition_labels)
    labels_df['weighted_labels'] = labels_df['weighted_skin_condition_label'].apply(parse_weighted_labels)

    # Keep only cases with actual conditions
    labeled_cases = labels_df[labels_df['condition_list'].apply(lambda x: len(x) > 0)].copy()
    print(f"  Cases with actual conditions: {len(labeled_cases)}")

    if limit:
        labeled_cases = labeled_cases.head(limit)
        print(f"  Limiting to {limit} cases for testing")

    # Extract primary condition for each case
    def get_primary_condition(row):
        """Get the primary condition from weighted labels or condition list."""
        if row['weighted_labels'] and isinstance(row['weighted_labels'], dict):
            if row['weighted_labels']:
                return max(row['weighted_labels'].items(), key=lambda x: x[1])[0]
        elif row['condition_list'] and isinstance(row['condition_list'], list):
            return row['condition_list'][0]
        return None

    labeled_cases['condition'] = labeled_cases.apply(get_primary_condition, axis=1)

    # Remove cases without a primary condition
    labeled_cases = labeled_cases[labeled_cases['condition'].notna()]

    print(f"\nConditions found:")
    condition_counts = labeled_cases['condition'].value_counts()
    for condition, count in condition_counts.head(10).items():
        print(f"  {condition}: {count} cases")
    if len(condition_counts) > 10:
        print(f"  ... and {len(condition_counts) - 10} more conditions")

    # Connect to GCS
    print("\nStep 2: Connecting to Google Cloud Storage...")
    try:
        client = storage.Client.create_anonymous_client()
        bucket = client.bucket("dx-scin-public-data")
        print("  Connected to GCS bucket")
    except Exception as e:
        print(f"  Error connecting: {e}")
        return False

    # Download images
    print("\nStep 3: Downloading labeled images...")
    print(f"  Target: {len(labeled_cases)} images")

    metadata = []
    downloaded = 0
    failed = 0

    for idx, (_, row) in enumerate(labeled_cases.iterrows()):
        case_id = row['case_id']
        condition = row['condition']

        # Try different possible paths
        possible_paths = [
            f"dataset/images/{case_id}.png",
            f"dataset/{case_id}.png",
            f"images/{case_id}.png",
            f"{case_id}.png"
        ]

        success = False
        for blob_path in possible_paths:
            try:
                blob = bucket.blob(blob_path)
                if blob.exists():
                    # Download the image
                    local_path = images_dir / f"{case_id}.png"
                    blob.download_to_filename(str(local_path))

                    # Add to metadata
                    metadata.append({
                        'image_id': case_id,
                        'image_path': f"{case_id}.png",
                        'condition': condition,
                        'condition_list': row['condition_list'],
                        'weighted_labels': row['weighted_labels']
                    })

                    downloaded += 1
                    success = True
                    break
            except Exception:
                continue

        if not success:
            failed += 1
            if failed <= 3:
                print(f"    Warning: Could not find image for case {case_id}")

        # Progress update
        if (idx + 1) % 50 == 0 or (idx + 1) == len(labeled_cases):
            print(f"  Progress: {idx + 1}/{len(labeled_cases)} - Downloaded: {downloaded}, Failed: {failed}")

    print(f"\n✓ Downloaded {downloaded} images ({failed} not found)")

    # Create metadata with proper labels
    print("\nStep 4: Creating labeled metadata...")

    if len(metadata) > 0:
        metadata_df = pd.DataFrame(metadata)

        # Create numeric labels
        unique_conditions = sorted(metadata_df['condition'].unique())
        condition_to_label = {cond: idx for idx, cond in enumerate(unique_conditions)}
        metadata_df['condition_label'] = metadata_df['condition'].map(condition_to_label)

        # Create train/val/test split
        import numpy as np
        np.random.seed(42)
        n = len(metadata_df)
        indices = np.random.permutation(n)

        train_end = int(0.7 * n)
        val_end = int(0.85 * n)

        metadata_df['split'] = 'test'  # default
        metadata_df.loc[indices[:train_end], 'split'] = 'train'
        metadata_df.loc[indices[train_end:val_end], 'split'] = 'val'
        metadata_df.loc[indices[val_end:], 'split'] = 'test'

        # Save final metadata
        final_columns = ['image_id', 'image_path', 'condition', 'condition_label', 'split']
        final_df = metadata_df[final_columns]
        final_df.to_csv(output_metadata, index=False)

        print(f"✓ Saved labeled metadata to: {output_metadata}")

        # Save condition mapping
        import json
        mapping_path = data_dir / "condition_mapping.json"
        with open(mapping_path, 'w') as f:
            json.dump(condition_to_label, f, indent=2)
        print(f"✓ Saved condition mapping to: {mapping_path}")

        # Print statistics
        print("\n" + "="*80)
        print("Dataset Statistics")
        print("="*80)
        print(f"\nTotal images: {len(final_df)}")
        print(f"Unique conditions: {len(unique_conditions)}")

        print("\nCondition distribution:")
        for condition, count in final_df['condition'].value_counts().head(10).items():
            print(f"  {condition}: {count} images")

        print("\nSplit distribution:")
        for split, count in final_df['split'].value_counts().items():
            print(f"  {split}: {count} images")

        print("\n" + "="*80)
        print("✓ Success!")
        print("="*80)
        print("\nYour labeled dataset is ready for training!")
        print("Update your config to use 'labeled_metadata.csv' and 'labeled_images' directory")

        return True
    else:
        print("\n✗ No images could be downloaded")
        return False

if __name__ == "__main__":
    # Set limit=100 for testing, or None for all images
    download_labeled_images(limit=500)  # Start with 500 images for testing