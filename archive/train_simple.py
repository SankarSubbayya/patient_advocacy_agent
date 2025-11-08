#!/usr/bin/env python
"""
Simplified training script with progress output.
"""

import torch
import pandas as pd
from pathlib import Path
import sys

print("Starting simplified training script...", flush=True)

# Force CPU for faster loading during testing
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}", flush=True)

# Load metadata
print("\n1. Loading labeled metadata...", flush=True)
metadata_path = Path('/home/sankar/data/scin/synthetic_labeled_metadata.csv')
df = pd.read_csv(metadata_path)

# Filter to only training data
train_df = df[df['split'] == 'train']
val_df = df[df['split'] == 'val']

print(f"   Train samples: {len(train_df)}", flush=True)
print(f"   Val samples: {len(val_df)}", flush=True)
print(f"   Conditions: {df['condition'].nunique()}", flush=True)

# Import after basic checks
print("\n2. Importing modules...", flush=True)
from patient_advocacy_agent import SigLIPEmbedder, SCINDataLoader, EmbedderTrainer

# Create data loader with explicit metadata
print("\n3. Creating data loader...", flush=True)
data_loader = SCINDataLoader(
    data_dir=Path('/home/sankar/data/scin'),
    batch_size=16,  # Smaller batch for testing
    num_workers=0
)

# Create dataloaders using labeled metadata
print("\n4. Creating dataloaders...", flush=True)
dataloaders = data_loader.create_dataloaders(
    metadata_path=metadata_path,
    images_dir='images'
)

print(f"   Train batches: {len(dataloaders['train'])}", flush=True)
print(f"   Val batches: {len(dataloaders['val'])}", flush=True)

# Get a sample batch to verify
print("\n5. Testing data loading...", flush=True)
for batch in dataloaders['train']:
    print(f"   Batch keys: {batch.keys()}", flush=True)
    print(f"   Image shape: {batch['image'].shape}", flush=True)
    print(f"   Conditions in batch: {batch['condition'][:3]}", flush=True)
    if 'description' in batch:
        print(f"   Descriptions: {batch['description'][:2]}", flush=True)
    break

# Create embedder
print("\n6. Creating SigLIP embedder...", flush=True)
embedder = SigLIPEmbedder(
    model_name='google/siglip-base-patch16-224',
    projection_dim=512,
    freeze_backbone=False
)
print("   ✓ Embedder created", flush=True)

# Move to device
print(f"\n7. Moving model to {device}...", flush=True)
embedder.to(device)

# Create trainer
print("\n8. Creating trainer...", flush=True)
trainer = EmbedderTrainer(
    embedder=embedder,
    device=device,
    learning_rate=1e-4
)

# Train for just 2 epochs to test
print("\n9. Starting training (2 epochs test)...", flush=True)
print("-" * 60, flush=True)

for epoch in range(2):
    print(f"\nEpoch {epoch + 1}/2", flush=True)

    # Train one epoch
    train_loss = trainer.train_epoch(dataloaders['train'])
    print(f"  Train loss: {train_loss:.4f}", flush=True)

    # Validate
    val_loss = trainer.validate(dataloaders['val'])
    print(f"  Val loss: {val_loss:.4f}", flush=True)

    # Check if loss is changing
    if epoch == 0:
        first_loss = train_loss
    else:
        loss_change = first_loss - train_loss
        if abs(loss_change) < 0.001:
            print("  ⚠️  WARNING: Loss is not changing! Check data/model", flush=True)
        else:
            print(f"  ✓ Loss changed by {loss_change:.4f}", flush=True)

print("\n" + "="*60, flush=True)
print("Training test complete!", flush=True)
print("="*60, flush=True)

if len(trainer.train_losses) > 1:
    print(f"\nLoss progression: {trainer.train_losses}", flush=True)
    if trainer.train_losses[0] > trainer.train_losses[-1]:
        print("✓ Model is LEARNING - loss decreased!", flush=True)
    else:
        print("⚠️  Model is NOT learning - loss did not decrease", flush=True)