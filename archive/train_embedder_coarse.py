#!/usr/bin/env python
"""
Train SigLIP embedder with COARSE-GRAINED labels (16 categories instead of 211).

This script uses coarse medical groupings like:
- Inflammatory Dermatitis (33.8% of data)
- Urticaria/Allergic (9.8%)
- Bacterial Infections (8.4%)
etc.

Expected improvement: Better contrastive learning with ~369 images/class vs ~31
"""

import torch
import pandas as pd
from pathlib import Path
import json
import matplotlib.pyplot as plt
from datetime import datetime

print("="*80)
print("SIGLIP TRAINING WITH COARSE-GRAINED MEDICAL CATEGORIES")
print("="*80)
print("Using: coarse_labeled_metadata_with_labels.csv")
print("Categories: 16 coarse medical groupings (vs 211 fine-grained)")
print("Images: 5,909 labeled (avg ~369 per category)")
print("="*80)

# Configuration
class Config:
    # Paths
    data_dir = Path('/home/sankar/data/scin')
    metadata_file = data_dir / 'coarse_labeled_metadata_with_labels.csv'
    images_dir = data_dir / 'images'

    # Model output
    model_dir = Path('/home/sankar/models/embedder_coarse_labels')
    checkpoint_dir = model_dir / 'checkpoints'
    final_dir = model_dir / 'final'

    # Training params
    batch_size = 32
    num_epochs = 20
    learning_rate = 5e-5
    weight_decay = 1e-5
    early_stopping_patience = 5

    # Model
    model_name = 'google/siglip-base-patch16-224'
    projection_dim = 512
    freeze_backbone = False  # Fine-tune entire model

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

config = Config()

# Create directories
config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
config.final_dir.mkdir(parents=True, exist_ok=True)

print(f"\nConfiguration:")
print(f"  Device: {config.device}")
print(f"  Batch size: {config.batch_size}")
print(f"  Learning rate: {config.learning_rate}")
print(f"  Epochs: {config.num_epochs}")
print(f"  Model output: {config.model_dir}")

# Verify coarse labeled data
print("\n" + "-"*80)
print("Verifying Coarse Labeled Data")
print("-"*80)

df = pd.read_csv(config.metadata_file)
print(f"Total samples: {len(df)}")
print(f"Unique coarse categories: {df['coarse_category'].nunique()}")

categories = df['coarse_category'].value_counts()
print("\nCoarse category distribution:")
for category, count in categories.items():
    pct = count / len(df) * 100
    print(f"  {category:30s}: {count:4d} ({pct:5.1f}%)")

# Check label distribution
print(f"\nNumeric labels: 0-{df['coarse_condition_label'].max()}")
print(f"Train samples: {(df['split'] == 'train').sum()}")
print(f"Val samples: {(df['split'] == 'val').sum()}")
print(f"Test samples: {(df['split'] == 'test').sum()}")

# Import modules
print("\n" + "-"*80)
print("Loading Modules")
print("-"*80)

from patient_advocacy_agent import (
    SCINDataLoader,
    SigLIPEmbedder,
    EmbedderTrainer
)

# Create data loader
print("\nCreating data loaders...")
data_loader = SCINDataLoader(
    data_dir=config.data_dir,
    batch_size=config.batch_size,
    num_workers=0
)

# Use coarse labeled metadata
dataloaders = data_loader.create_dataloaders(
    metadata_path=config.metadata_file,  # Pass Path object, not string
    images_dir='images'
)

print(f"  Train batches: {len(dataloaders['train'])}")
print(f"  Val batches: {len(dataloaders['val'])}")
print(f"  Test batches: {len(dataloaders['test'])}")

# Create model
print("\nCreating SigLIP embedder...")
embedder = SigLIPEmbedder(
    model_name=config.model_name,
    projection_dim=config.projection_dim,
    freeze_backbone=config.freeze_backbone
)
print(f"  ✓ Model created")
print(f"  Moving to {config.device}...")
embedder.to(config.device)

# Create trainer
print("\nCreating trainer...")
trainer = EmbedderTrainer(
    embedder=embedder,
    device=config.device,
    learning_rate=config.learning_rate,
    weight_decay=config.weight_decay
)

# Add learning rate scheduler
from torch.optim.lr_scheduler import CosineAnnealingLR
trainer.scheduler = CosineAnnealingLR(
    trainer.optimizer,
    T_max=config.num_epochs,
    eta_min=1e-6
)
print("  ✓ Trainer created with cosine annealing scheduler")

# Training loop
print("\n" + "="*80)
print("STARTING TRAINING WITH COARSE CATEGORIES")
print("="*80)

best_val_loss = float('inf')
patience_counter = 0
training_start = datetime.now()
train_losses = []
val_losses = []

for epoch in range(config.num_epochs):
    print(f"\n[Epoch {epoch+1}/{config.num_epochs}]")

    # Train
    train_loss = trainer.train_epoch(dataloaders['train'])
    train_losses.append(train_loss)
    print(f"  Train Loss: {train_loss:.4f}")

    # Validate
    val_loss = trainer.validate(dataloaders['val'])
    val_losses.append(val_loss)
    print(f"  Val Loss:   {val_loss:.4f}")

    # Learning rate schedule step
    if hasattr(trainer, 'scheduler'):
        trainer.scheduler.step()
        current_lr = trainer.scheduler.get_last_lr()[0]
        print(f"  Learning Rate: {current_lr:.2e}")

    # Save checkpoint
    is_best = val_loss < best_val_loss
    if is_best:
        best_val_loss = val_loss
        patience_counter = 0

        # Save best model
        checkpoint_path = config.checkpoint_dir / f'best_model_epoch_{epoch+1}.pt'
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': embedder.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'best_val_loss': best_val_loss,
        }, checkpoint_path)
        print(f"  ✓ Best model saved! (val_loss: {val_loss:.4f})")
    else:
        patience_counter += 1
        print(f"  No improvement ({patience_counter}/{config.early_stopping_patience})")

    # Early stopping
    if patience_counter >= config.early_stopping_patience:
        print(f"\n⚠ Early stopping triggered after {epoch+1} epochs")
        break

training_end = datetime.now()
training_duration = (training_end - training_start).total_seconds() / 60

# Final evaluation
print("\n" + "="*80)
print("FINAL EVALUATION")
print("="*80)

test_loss = trainer.validate(dataloaders['test'])
print(f"Test Loss: {test_loss:.4f}")

# Save final model
final_model_path = config.final_dir / 'embedder_coarse_labels.pt'
torch.save({
    'model_state_dict': embedder.state_dict(),
    'config': {
        'model_name': config.model_name,
        'projection_dim': config.projection_dim,
        'freeze_backbone': config.freeze_backbone,
        'num_categories': 16,
        'batch_size': config.batch_size,
        'learning_rate': config.learning_rate,
    },
    'training_history': {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'test_loss': test_loss,
        'num_epochs_trained': len(train_losses),
        'training_duration_minutes': training_duration,
    }
}, final_model_path)
print(f"✓ Final model saved to: {final_model_path}")

# Save training history
history_path = config.final_dir / 'training_history.json'
with open(history_path, 'w') as f:
    json.dump({
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'test_loss': test_loss,
        'num_epochs_trained': len(train_losses),
        'config': {
            'batch_size': config.batch_size,
            'learning_rate': config.learning_rate,
            'model_name': config.model_name,
            'num_categories': 16,
        }
    }, f, indent=2)
print(f"✓ Training history saved to: {history_path}")

# Plot training curves
plt.figure(figsize=(10, 6))
epochs = range(1, len(train_losses) + 1)
plt.plot(epochs, train_losses, 'b-o', label='Train Loss', markersize=4)
plt.plot(epochs, val_losses, 'r-s', label='Val Loss', markersize=4)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Progress: Coarse Categories (16 classes)')
plt.legend()
plt.grid(True, alpha=0.3)

plot_path = config.final_dir / 'training_curve.png'
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"✓ Training curve saved to: {plot_path}")

# Summary
print("\n" + "="*80)
print("TRAINING COMPLETE")
print("="*80)
print(f"\nBest validation loss: {best_val_loss:.4f}")
print(f"Final test loss: {test_loss:.4f}")
print(f"Epochs trained: {len(train_losses)}")
print(f"Training duration: {training_duration:.1f} minutes")
print(f"\nModel saved to: {final_model_path}")

print("\n" + "="*80)
print("NEXT STEPS")
print("="*80)
print("\n1. Compare clustering performance:")
print("   - Vanilla SigLIP (baseline)")
print("   - Fine-tuned with 211 classes (previous)")
print("   - Fine-tuned with 16 coarse classes (new)")
print("\n2. Run: python compare_embeddings_coarse.py")
print("\n3. Expected: Better clustering with coarse categories!")
