#!/usr/bin/env python
"""
Quick test to verify labeled data can be loaded.
"""

import pandas as pd
from pathlib import Path

def test_labeled_data():
    """Test loading labeled data."""

    # Load synthetic labeled metadata
    metadata_path = Path('/home/sankar/data/scin/synthetic_labeled_metadata.csv')

    print("Testing Labeled Data Loading")
    print("="*60)

    if not metadata_path.exists():
        print(f"✗ File not found: {metadata_path}")
        return False

    # Load data
    df = pd.read_csv(metadata_path)

    print(f"✓ Loaded {len(df)} samples from {metadata_path.name}")

    # Check columns
    print(f"\nColumns: {list(df.columns)}")

    # Check conditions
    conditions = df['condition'].value_counts()
    print(f"\nUnique conditions: {len(conditions)}")
    print("\nTop 5 conditions:")
    for condition, count in conditions.head(5).items():
        print(f"  {condition}: {count}")

    # Check splits
    if 'split' in df.columns:
        splits = df['split'].value_counts()
        print("\nData splits:")
        for split, count in splits.items():
            print(f"  {split}: {count}")

    # Check if images exist
    images_dir = Path('/home/sankar/data/scin/images')
    sample_image = df.iloc[0]['image_path']
    sample_path = images_dir / sample_image

    print(f"\nChecking if images exist...")
    print(f"  Sample: {sample_image}")
    print(f"  Path: {sample_path}")
    print(f"  Exists: {sample_path.exists()}")

    # Check descriptions
    if 'description' in df.columns:
        print(f"\nSample descriptions:")
        for desc in df['description'].head(3).values:
            print(f"  - {desc}")

    print("\n✓ Data looks good for training!")
    return True

if __name__ == "__main__":
    test_labeled_data()