#!/usr/bin/env python3
"""
Create metadata file with coarse-grained categories (16 groups instead of 211).
This should make contrastive learning more effective.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

# Load the coarse category metadata (created by analyze_conditions.py)
df = pd.read_csv('/home/sankar/data/scin/coarse_labeled_metadata.csv')

print("="*80)
print("CREATING COARSE-GRAINED LABELED METADATA")
print("="*80)
print(f"\nTotal images: {len(df)}")
print(f"Unique coarse categories: {df['coarse_category'].nunique()}")

# Filter out "Other" category for training
labeled_df = df[df['coarse_category'] != 'Other'].copy()
print(f"Images after removing 'Other': {len(labeled_df)}")

# Create numeric labels for coarse categories
unique_categories = sorted(labeled_df['coarse_category'].unique())
category_to_label = {cat: idx for idx, cat in enumerate(unique_categories)}

print(f"\nCoarse categories ({len(unique_categories)}):")
for cat, label in category_to_label.items():
    count = (labeled_df['coarse_category'] == cat).sum()
    pct = count / len(labeled_df) * 100
    print(f"  {label:2d}. {cat:30s} - {count:4d} images ({pct:5.1f}%)")

# Apply numeric labels
labeled_df['coarse_condition_label'] = labeled_df['coarse_category'].map(category_to_label)

# Re-create train/val/test splits on labeled data only
print("\n" + "="*80)
print("Creating train/val/test splits...")
print("="*80)

np.random.seed(42)
indices = np.random.permutation(len(labeled_df))

# 70% train, 15% val, 15% test
n_total = len(indices)
train_end = int(0.7 * n_total)
val_end = int(0.85 * n_total)

train_indices = indices[:train_end]
val_indices = indices[train_end:val_end]
test_indices = indices[val_end:]

# Assign splits
labeled_df['split'] = 'train'
labeled_df.iloc[val_indices, labeled_df.columns.get_loc('split')] = 'val'
labeled_df.iloc[test_indices, labeled_df.columns.get_loc('split')] = 'test'

print(f"\nTrain: {len(train_indices)} images ({len(train_indices)/n_total*100:.1f}%)")
print(f"Val:   {len(val_indices)} images ({len(val_indices)/n_total*100:.1f}%)")
print(f"Test:  {len(test_indices)} images ({len(test_indices)/n_total*100:.1f}%)")

# Verify each category has samples in each split
print("\nCategory distribution across splits:")
for cat in unique_categories:
    cat_df = labeled_df[labeled_df['coarse_category'] == cat]
    train_count = (cat_df['split'] == 'train').sum()
    val_count = (cat_df['split'] == 'val').sum()
    test_count = (cat_df['split'] == 'test').sum()
    print(f"  {cat:30s} - Train: {train_count:3d}, Val: {val_count:3d}, Test: {test_count:3d}")

# Save the coarse labeled metadata
output_path = '/home/sankar/data/scin/coarse_labeled_metadata_with_labels.csv'
labeled_df.to_csv(output_path, index=False)
print(f"\n✓ Saved coarse labeled metadata to: {output_path}")

# Save category mapping
mapping_path = '/home/sankar/data/scin/coarse_category_mapping.json'
with open(mapping_path, 'w') as f:
    json.dump(category_to_label, f, indent=2)
print(f"✓ Saved category mapping to: {mapping_path}")

# Print summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"\nReduced from 211 fine-grained conditions to {len(unique_categories)} coarse categories")
print(f"Total labeled images: {len(labeled_df)}")
print(f"Average images per category: {len(labeled_df) / len(unique_categories):.1f}")
print(f"\nThis should significantly improve contrastive learning:")
print(f"  • More samples per class (~407 vs ~31)")
print(f"  • Simpler decision boundaries (16 vs 211 classes)")
print(f"  • Better alignment with medical groupings")

print("\n" + "="*80)
print("NEXT STEPS")
print("="*80)
print("\n1. Train embedder with coarse categories:")
print("   uv run python train_embedder_coarse.py")
print("\n2. Compare clustering: Vanilla vs Fine-tuned (211) vs Fine-tuned (16)")
print("\n3. Evaluate if coarse categories improve embedding quality")
