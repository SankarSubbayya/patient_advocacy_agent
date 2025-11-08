#!/usr/bin/env python3
"""
Download ONLY the images that have actual labels from scin_labels.csv.
This ensures we have real image-label pairs, not random assignments.
"""

import pandas as pd
import ast
from pathlib import Path
from google.cloud import storage
import numpy as np
import json

def parse_condition_labels(label_str):
    """Parse condition labels from string."""
    if pd.isna(label_str) or label_str == '[]':
        return []
    try:
        return ast.literal_eval(label_str)
    except:
        return []

def parse_weighted_labels(weighted_str):
    """Parse weighted labels from string."""
    if pd.isna(weighted_str) or weighted_str == '{}':
        return {}
    try:
        return ast.literal_eval(weighted_str)
    except:
        return {}

print("="*80)
print("DOWNLOADING CORRECTLY LABELED SCIN IMAGES")
print("="*80)
print("\nThis will download ONLY images that have real labels,")
print("ensuring proper training data for contrastive learning.\n")

# Paths
data_dir = Path('/home/sankar/data/scin')
labels_path = data_dir / 'scin_labels.csv'
output_dir = data_dir / 'labeled_images_correct'
output_metadata = data_dir / 'correct_labeled_metadata.csv'

output_dir.mkdir(parents=True, exist_ok=True)

# Load and process labels
print("Step 1: Loading labels file...")
labels_df = pd.read_csv(labels_path)
print(f"  Total cases in labels file: {len(labels_df)}")

# Parse conditions
labels_df['condition_list'] = labels_df['dermatologist_skin_condition_on_label_name'].apply(parse_condition_labels)
labels_df['weighted_labels'] = labels_df['weighted_skin_condition_label'].apply(parse_weighted_labels)

# Filter to only cases with actual conditions
labeled_cases = labels_df[labels_df['condition_list'].apply(lambda x: len(x) > 0)].copy()
print(f"  Cases with actual conditions: {len(labeled_cases)}")

# Extract primary condition
def get_primary_condition(row):
    """Get the primary condition from weighted labels or condition list."""
    if row['weighted_labels'] and isinstance(row['weighted_labels'], dict):
        if row['weighted_labels']:
            return max(row['weighted_labels'].items(), key=lambda x: x[1])[0]
    elif row['condition_list'] and isinstance(row['condition_list'], list):
        return row['condition_list'][0]
    return None

labeled_cases['primary_condition'] = labeled_cases.apply(get_primary_condition, axis=1)
labeled_cases = labeled_cases[labeled_cases['primary_condition'].notna()]
print(f"  Cases with valid primary condition: {len(labeled_cases)}")

# Download ALL labeled images for better training
# LIMIT = 1000  # Uncomment to limit for testing
# print(f"\n  LIMITING TO {LIMIT} IMAGES FOR TESTING")
# labeled_cases = labeled_cases.head(LIMIT)
print(f"\n  Downloading ALL {len(labeled_cases)} labeled images for full training")

# Show condition distribution
print("\nCondition distribution (top 10):")
conditions = labeled_cases['primary_condition'].value_counts()
for condition, count in conditions.head(10).items():
    print(f"  {condition}: {count}")

# Connect to GCS
print("\nStep 2: Connecting to Google Cloud Storage...")
try:
    client = storage.Client.create_anonymous_client()
    bucket = client.bucket("dx-scin-public-data")
    print("  ✓ Connected to GCS bucket")
except Exception as e:
    print(f"  ✗ Error: {e}")
    exit(1)

# Download images
print(f"\nStep 3: Downloading {len(labeled_cases)} labeled images...")
print("  (This may take several minutes...)\n")

metadata_records = []
downloaded = 0
failed = 0
not_found = []

for idx, (_, row) in enumerate(labeled_cases.iterrows()):
    case_id = str(row['case_id'])
    condition = row['primary_condition']

    # Progress indicator
    if (idx + 1) % 50 == 0:
        print(f"  Progress: {idx + 1}/{len(labeled_cases)} - Downloaded: {downloaded}, Failed: {failed}")

    # Try to find and download the image
    found = False
    # Most images are in dataset/images/ so try that first
    blob_path = f"dataset/images/{case_id}.png"
    try:
        blob = bucket.blob(blob_path)

        # Try to download directly (faster than checking existence first)
        local_path = output_dir / f"{case_id}.png"
        blob.download_to_filename(str(local_path))

        # Add to metadata
        metadata_records.append({
            'image_id': case_id,
            'image_path': f"{case_id}.png",
            'condition': condition,
            'confidence': row.get('dermatologist_skin_condition_confidence', ''),
            'all_conditions': str(row['condition_list']),
            'weighted_labels': str(row['weighted_labels'])
        })

        downloaded += 1
        found = True

    except Exception:
        # Image not found in expected location
        pass

    if not found:
        failed += 1
        not_found.append(case_id)
        if failed <= 5:  # Show first 5 failures
            print(f"    Warning: Could not find image for case {case_id}")

print(f"\n  Final: Downloaded {downloaded}/{len(labeled_cases)} images")
print(f"  Failed: {failed} images not found in bucket")

if downloaded == 0:
    print("\n✗ ERROR: No images could be downloaded!")
    print("  The image IDs in the labels file don't match the images in the bucket.")
    exit(1)

# Create metadata CSV
print("\nStep 4: Creating metadata CSV...")

metadata_df = pd.DataFrame(metadata_records)

# Create numeric labels
unique_conditions = sorted(metadata_df['condition'].unique())
condition_to_label_map = {cond: idx for idx, cond in enumerate(unique_conditions)}
metadata_df['condition_label'] = metadata_df['condition'].map(condition_to_label_map)

# Create train/val/test splits
np.random.seed(42)
n = len(metadata_df)
indices = np.random.permutation(n)

train_end = int(0.7 * n)
val_end = int(0.85 * n)

metadata_df['split'] = 'test'
metadata_df.loc[indices[:train_end], 'split'] = 'train'
metadata_df.loc[indices[train_end:val_end], 'split'] = 'val'
metadata_df.loc[indices[val_end:], 'split'] = 'test'

# Add descriptions
def create_description(condition):
    """Create text description for training."""
    templates = [
        f"A dermatological image showing {condition}",
        f"Clinical presentation of {condition}",
        f"Skin condition diagnosed as {condition}",
        f"Patient presenting with {condition}",
        f"Medical image of {condition}",
        f"Dermatoscopy showing {condition}",
        f"Skin lesion consistent with {condition}"
    ]
    return np.random.choice(templates)

metadata_df['description'] = metadata_df['condition'].apply(create_description)

# Save metadata
metadata_df.to_csv(output_metadata, index=False)
print(f"  ✓ Saved metadata to {output_metadata}")

# Save condition mapping
mapping = {
    'condition_to_label': condition_to_label_map,
    'label_to_condition': {v: k for k, v in condition_to_label_map.items()},
    'total_images': len(metadata_df),
    'num_conditions': len(unique_conditions)
}

mapping_path = data_dir / 'correct_condition_mapping.json'
with open(mapping_path, 'w') as f:
    json.dump(mapping, f, indent=2)
print(f"  ✓ Saved condition mapping to {mapping_path}")

# Print summary
print("\n" + "="*80)
print("DOWNLOAD COMPLETE - REAL LABELED DATA READY!")
print("="*80)
print(f"\nDataset Statistics:")
print(f"  Total images: {len(metadata_df)}")
print(f"  Unique conditions: {len(unique_conditions)}")
print(f"  Images directory: {output_dir}")
print(f"  Metadata file: {output_metadata}")

print(f"\nSplit distribution:")
for split, count in metadata_df['split'].value_counts().items():
    print(f"  {split}: {count}")

print(f"\nTop conditions:")
for condition, count in metadata_df['condition'].value_counts().head(5).items():
    print(f"  {condition}: {count}")

print("\n" + "="*80)
print("NEXT STEPS:")
print("="*80)
print("1. Train with REAL labeled data:")
print("   - Use 'correct_labeled_metadata.csv'")
print("   - Images in 'labeled_images_correct/' directory")
print("\n2. This should show ACTUAL learning because:")
print("   - Images match their medical labels")
print("   - Contrastive loss has real signal to learn from")
print("\n3. Expected: Loss should decrease significantly!")
print("\nTo train: Update your training script to use the new paths.")