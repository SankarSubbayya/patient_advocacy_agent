#!/usr/bin/env python3
"""
Create CORRECTLY labeled metadata using the proper case-to-image mapping.

Now that we understand:
- case_id = medical case (patient visit)
- image paths = actual image files (1-3 per case)

We can properly link images to their real medical labels!
"""

import pandas as pd
import numpy as np
import json
import ast
from pathlib import Path
import os

print("="*80)
print("CREATING CORRECTLY LABELED METADATA")
print("="*80)

# Paths
data_dir = Path('/home/sankar/data/scin')
cases_path = '/tmp/scin_cases.csv'  # Already downloaded
labels_path = data_dir / 'scin_labels.csv'
images_dir = data_dir / 'images'
output_metadata = data_dir / 'real_labeled_metadata.csv'

# Load data
print("\n1. Loading scin_cases.csv...")
cases_df = pd.read_csv(cases_path, dtype={'case_id': str})
print(f"   Loaded {len(cases_df)} cases")

print("\n2. Loading scin_labels.csv...")
labels_df = pd.read_csv(labels_path, dtype={'case_id': str})
print(f"   Loaded {len(labels_df)} labeled cases")

# Merge cases with labels
print("\n3. Merging cases with labels...")
merged_df = cases_df.merge(labels_df, on='case_id', how='inner')
print(f"   Merged {len(merged_df)} cases with labels")

# Parse condition labels
def parse_conditions(label_str):
    """Parse condition list from string."""
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

merged_df['condition_list'] = merged_df['dermatologist_skin_condition_on_label_name'].apply(parse_conditions)
merged_df['weighted_labels'] = merged_df['weighted_skin_condition_label'].apply(parse_weighted_labels)

# Filter to cases with actual conditions
merged_df = merged_df[merged_df['condition_list'].apply(lambda x: len(x) > 0)]
print(f"   Cases with valid conditions: {len(merged_df)}")

# Get primary condition
def get_primary_condition(row):
    """Get primary condition from weighted labels."""
    if row['weighted_labels'] and isinstance(row['weighted_labels'], dict):
        if row['weighted_labels']:
            return max(row['weighted_labels'].items(), key=lambda x: x[1])[0]
    elif row['condition_list'] and isinstance(row['condition_list'], list):
        return row['condition_list'][0]
    return None

merged_df['primary_condition'] = merged_df.apply(get_primary_condition, axis=1)
merged_df = merged_df[merged_df['primary_condition'].notna()]
print(f"   Cases with primary condition: {len(merged_df)}")

# Now expand: each case can have 1-3 images
print("\n4. Expanding cases to individual images...")

image_records = []

for _, case_row in merged_df.iterrows():
    case_id = case_row['case_id']
    condition = case_row['primary_condition']

    # Check each of the 3 possible image paths
    for img_col in ['image_1_path', 'image_2_path', 'image_3_path']:
        if pd.notna(case_row[img_col]):
            # Extract image ID from path
            image_path = case_row[img_col]
            image_filename = os.path.basename(image_path)
            image_id = image_filename.replace('.png', '')

            # Check if we have this image
            if (images_dir / image_filename).exists():
                image_records.append({
                    'image_id': image_id,
                    'image_path': image_filename,
                    'case_id': case_id,
                    'condition': condition,
                    'all_conditions': str(case_row['condition_list']),
                    'weighted_labels': str(case_row['weighted_labels']),
                    'age_group': case_row.get('age_group', ''),
                    'sex_at_birth': case_row.get('sex_at_birth', ''),
                    'fitzpatrick_skin_type': case_row.get('fitzpatrick_skin_type', '')
                })

print(f"   Extracted {len(image_records)} labeled images")

# Create metadata DataFrame
print("\n5. Creating metadata DataFrame...")
metadata_df = pd.DataFrame(image_records)

# Create numeric labels
unique_conditions = sorted(metadata_df['condition'].unique())
condition_to_label = {cond: idx for idx, cond in enumerate(unique_conditions)}
metadata_df['condition_label'] = metadata_df['condition'].map(condition_to_label)

print(f"   Total images: {len(metadata_df)}")
print(f"   Unique conditions: {len(unique_conditions)}")

# Show condition distribution
print("\n   Top 10 conditions:")
for condition, count in metadata_df['condition'].value_counts().head(10).items():
    print(f"     {condition}: {count} images")

# Create train/val/test splits
print("\n6. Creating train/val/test splits...")

# Split by case_id to avoid data leakage (same case shouldn't be in train and test)
unique_cases = metadata_df['case_id'].unique()
np.random.seed(42)
shuffled_cases = np.random.permutation(unique_cases)

n_cases = len(unique_cases)
train_end = int(0.7 * n_cases)
val_end = int(0.85 * n_cases)

train_cases = set(shuffled_cases[:train_end])
val_cases = set(shuffled_cases[train_end:val_end])
test_cases = set(shuffled_cases[val_end:])

# Assign splits
metadata_df['split'] = metadata_df['case_id'].apply(
    lambda x: 'train' if x in train_cases else ('val' if x in val_cases else 'test')
)

print(f"   Train: {len(metadata_df[metadata_df['split']=='train'])} images")
print(f"   Val: {len(metadata_df[metadata_df['split']=='val'])} images")
print(f"   Test: {len(metadata_df[metadata_df['split']=='test'])} images")

# Add text descriptions for training
print("\n7. Adding text descriptions...")

def create_description(condition):
    """Create text description for contrastive learning."""
    templates = [
        f"A dermatological image showing {condition}",
        f"Clinical presentation of {condition}",
        f"Skin condition diagnosed as {condition}",
        f"Patient presenting with {condition}",
        f"Medical photograph of {condition}",
        f"Dermatoscopy image showing {condition}",
        f"Skin lesion consistent with {condition}"
    ]
    return np.random.choice(templates)

metadata_df['description'] = metadata_df['condition'].apply(create_description)

# Save metadata
print("\n8. Saving real labeled metadata...")
metadata_df.to_csv(output_metadata, index=False)
print(f"   ✓ Saved to {output_metadata}")

# Save condition mapping
mapping = {
    'condition_to_label': condition_to_label,
    'label_to_condition': {v: k for k, v in condition_to_label.items()},
    'total_images': len(metadata_df),
    'num_conditions': len(unique_conditions)
}

mapping_path = data_dir / 'real_condition_mapping.json'
with open(mapping_path, 'w') as f:
    json.dump(mapping, f, indent=2)
print(f"   ✓ Saved condition mapping to {mapping_path}")

print("\n" + "="*80)
print("SUCCESS! REAL LABELED DATA READY")
print("="*80)
print(f"\nYour images now have REAL medical labels from dermatologists!")
print(f"- Total labeled images: {len(metadata_df)}")
print(f"- Unique conditions: {len(unique_conditions)}")
print(f"- Metadata file: {output_metadata}")
print(f"- Images directory: {images_dir}")

print("\n" + "="*80)
print("NEXT: TRAIN WITH REAL LABELS")
print("="*80)
print("Now you can train with confidence that:")
print("  • Images actually show what they're labeled as")
print("  • Labels come from medical experts")
print("  • The model should learn real visual-text associations")
print("\nExpected: Loss should DECREASE significantly!")
print("\nTo train, update your script to use: real_labeled_metadata.csv")