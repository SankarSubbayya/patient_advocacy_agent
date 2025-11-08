#!/usr/bin/env python3
"""
Merge SCIN image metadata with actual labels to create a properly labeled dataset.

This script:
1. Loads the dummy metadata.csv (with image paths)
2. Loads the real scin_labels.csv (with skin condition labels)
3. Merges them based on image_id/case_id
4. Creates a new labeled_metadata.csv with actual skin conditions
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import ast

def parse_condition_labels(label_str):
    """Parse the condition label string into a list."""
    if pd.isna(label_str) or label_str == '[]':
        return []
    try:
        # Use ast.literal_eval for safe parsing
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

def merge_labels():
    """Merge image metadata with actual labels."""

    print("="*80)
    print("SCIN Label Merger")
    print("="*80)
    print()

    # Paths
    data_dir = Path("/home/sankar/data/scin")
    metadata_path = data_dir / "metadata.csv"
    labels_path = data_dir / "scin_labels.csv"
    output_path = data_dir / "labeled_metadata.csv"

    # Load the current metadata (with image paths)
    print("Step 1: Loading current metadata...")
    metadata_df = pd.read_csv(metadata_path)
    print(f"  Loaded {len(metadata_df)} image entries")
    print(f"  Columns: {list(metadata_df.columns)}")

    # Load the real labels
    print("\nStep 2: Loading real labels...")
    labels_df = pd.read_csv(labels_path)
    print(f"  Loaded {len(labels_df)} label entries")
    print(f"  Columns: {list(labels_df.columns)[:5]}...")  # Show first 5 columns

    # Convert image_id to case_id format for merging
    # The image_id in metadata is the filename without extension
    # The case_id in labels should match this
    print("\nStep 3: Preparing for merge...")

    # Extract image_id from image_path (remove .png extension)
    metadata_df['case_id'] = metadata_df['image_path'].str.replace('.png', '')

    # Convert case_id to string in both dataframes for consistent merging
    metadata_df['case_id'] = metadata_df['case_id'].astype(str)
    labels_df['case_id'] = labels_df['case_id'].astype(str)

    print(f"  Sample image IDs from metadata: {metadata_df['case_id'].head(3).tolist()}")
    print(f"  Sample case IDs from labels: {labels_df['case_id'].head(3).tolist()}")

    # Merge the dataframes
    print("\nStep 4: Merging datasets...")
    merged_df = metadata_df.merge(
        labels_df[['case_id', 'dermatologist_skin_condition_on_label_name',
                   'weighted_skin_condition_label', 'dermatologist_skin_condition_confidence']],
        on='case_id',
        how='left'
    )

    print(f"  Merged {len(merged_df)} entries")
    print(f"  Entries with labels: {merged_df['dermatologist_skin_condition_on_label_name'].notna().sum()}")
    print(f"  Entries without labels: {merged_df['dermatologist_skin_condition_on_label_name'].isna().sum()}")

    # Process the labels
    print("\nStep 5: Processing labels...")

    # Parse condition labels
    merged_df['condition_list'] = merged_df['dermatologist_skin_condition_on_label_name'].apply(parse_condition_labels)
    merged_df['weighted_labels'] = merged_df['weighted_skin_condition_label'].apply(parse_weighted_labels)

    # Extract primary condition (most confident or first listed)
    def get_primary_condition(row):
        """Get the primary condition from weighted labels or condition list."""
        if row['weighted_labels'] and isinstance(row['weighted_labels'], dict):
            # Get condition with highest weight
            if row['weighted_labels']:
                return max(row['weighted_labels'].items(), key=lambda x: x[1])[0]
        elif row['condition_list'] and isinstance(row['condition_list'], list):
            # Use first condition in list
            return row['condition_list'][0]
        return 'unknown'

    merged_df['condition'] = merged_df.apply(get_primary_condition, axis=1)

    # Create numeric labels for conditions
    print("\nStep 6: Creating condition labels...")

    # Get unique conditions (excluding 'unknown')
    valid_conditions = merged_df[merged_df['condition'] != 'unknown']['condition'].unique()
    valid_conditions = sorted([c for c in valid_conditions if c])  # Remove None/empty

    # Create condition to label mapping
    condition_to_label = {condition: idx for idx, condition in enumerate(valid_conditions)}
    condition_to_label['unknown'] = len(valid_conditions)  # Unknown gets the last label

    # Apply numeric labels
    merged_df['condition_label'] = merged_df['condition'].map(condition_to_label)
    merged_df['condition_label'] = merged_df['condition_label'].fillna(len(valid_conditions)).astype(int)

    # Add train/val/test split
    print("\nStep 7: Creating train/val/test splits...")

    # Only split labeled data
    labeled_mask = merged_df['condition'] != 'unknown'
    labeled_indices = merged_df[labeled_mask].index

    # Shuffle indices
    np.random.seed(42)
    shuffled_indices = np.random.permutation(labeled_indices)

    # Split: 70% train, 15% val, 15% test
    n_labeled = len(shuffled_indices)
    train_end = int(0.7 * n_labeled)
    val_end = int(0.85 * n_labeled)

    train_indices = shuffled_indices[:train_end]
    val_indices = shuffled_indices[train_end:val_end]
    test_indices = shuffled_indices[val_end:]

    # Assign splits
    merged_df['split'] = 'unlabeled'  # Default for unlabeled data
    merged_df.loc[train_indices, 'split'] = 'train'
    merged_df.loc[val_indices, 'split'] = 'val'
    merged_df.loc[test_indices, 'split'] = 'test'

    # Create final dataframe with clean columns
    print("\nStep 8: Creating final dataset...")

    final_df = merged_df[['image_id', 'image_path', 'condition', 'condition_label', 'split']].copy()

    # Save the labeled metadata
    final_df.to_csv(output_path, index=False)
    print(f"\n✓ Saved labeled metadata to: {output_path}")

    # Print statistics
    print("\n" + "="*80)
    print("Dataset Statistics")
    print("="*80)
    print(f"\nTotal images: {len(final_df)}")
    print(f"Labeled images: {(final_df['condition'] != 'unknown').sum()}")
    print(f"Unlabeled images: {(final_df['condition'] == 'unknown').sum()}")

    print("\nCondition distribution:")
    condition_counts = final_df['condition'].value_counts()
    for condition, count in condition_counts.head(20).items():
        print(f"  {condition}: {count} images")

    if len(condition_counts) > 20:
        print(f"  ... and {len(condition_counts) - 20} more conditions")

    print(f"\nTotal unique conditions: {len(valid_conditions)}")

    print("\nSplit distribution (labeled data only):")
    split_counts = final_df[final_df['condition'] != 'unknown']['split'].value_counts()
    for split, count in split_counts.items():
        print(f"  {split}: {count} images")

    # Save condition mapping
    mapping_path = data_dir / "condition_mapping.json"
    with open(mapping_path, 'w') as f:
        json.dump(condition_to_label, f, indent=2)
    print(f"\n✓ Saved condition mapping to: {mapping_path}")

    print("\n" + "="*80)
    print("✓ Label merge complete!")
    print("="*80)
    print("\nNext steps:")
    print("1. Update your config to use 'labeled_metadata.csv' instead of 'metadata.csv'")
    print("2. Re-train the embedder with: uv run python train_embedder.py")
    print("3. The training should now show actual learning with decreasing loss!")

    return True

if __name__ == "__main__":
    merge_labels()